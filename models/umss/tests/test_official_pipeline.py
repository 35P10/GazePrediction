"""Checks for the training pipeline built on the official code."""
import contextlib
import csv
import importlib.util
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from models.umss.evaluation import saliency as S
from models.umss.training import s1 as S1, data_preparation as D
from models.umss.reference.setup_umss import setup
from models.umss.reference.verify import verify_sources

PACKAGE = Path(__file__).resolve().parents[1]
REFERENCE = S.DEFAULT_DESTINATION


class OfficialPipelineTests(unittest.TestCase):
    def test_docker_layout_runs_without_the_models_package(self):
        """The image copies reference/, evaluation/ and training/ only (see training/Dockerfile)."""
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / 'umss'
            for folder in ('reference', 'evaluation', 'training'):
                shutil.copytree(PACKAGE / folder, image / folder, ignore=shutil.ignore_patterns('__pycache__'))
            env = {**os.environ, 'PYTHONPATH': ''}
            scripts = [image / 'training/pipeline' / name
                       for name in ('download.py', 'prepare_data.py', 'train.py', 'predict.py')]
            for script in scripts + [image / 'reference/verify.py']:
                subprocess.run([sys.executable, str(script), '--help'], cwd=temporary, env=env,
                               check=True, capture_output=True)

    def test_s1_runs_the_notebook_recipe_with_our_overrides(self):
        self.assertEqual(S1.s1_defaults(REFERENCE)['batch_size'], 8)
        model = Mock()
        adam = Mock(return_value='notebook-Adam')
        data = {f'{kind}_files_{split}': [] for kind in ('img', 'map', 'fix') for split in ('train', 'val')}
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        bindings = dict(
            bp=temporary.name, ckpt_savedir=temporary.name, weightspath='initial.hdf5', n_epochs=6, batch_size=8,
            training_verbose=2, Adam=adam, ModelCheckpoint=Mock(return_value='checkpoint'),
            LearningRateScheduler=Mock(return_value='scheduler'), InteractivePlot=Mock(return_value='csv-log'),
            load_datasets_multiduration=Mock(return_value=data),
            get_model_by_name=Mock(return_value=(Mock(return_value=model), 'mode')),
            create_losses=Mock(return_value=(['kl', 'cc', 'nss', 'ccmatch'], [10, -5, -1, 3], 'fixture', 3)),
            MultidurationGenerator=lambda **kwargs: kwargs)
        with contextlib.redirect_stdout(io.StringIO()):
            state = S1.run_s1(REFERENCE, bindings)
            learning_rates = [state['step_decay'](epoch) for epoch in range(6)]
        adam.assert_called_once_with(lr=0.0001)
        model.load_weights.assert_called_with('initial.hdf5')
        model.fit_generator.assert_called_with(
            state['gen_train'], epochs=6, verbose=2, callbacks=['checkpoint', 'scheduler', 'csv-log'],
            validation_data=state['gen_val'], max_queue_size=10, workers=5)
        np.testing.assert_allclose(learning_rates, [1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-6])

    def test_verify_rejects_modified_official_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / 'upstream'
            with contextlib.redirect_stdout(io.StringIO()):
                setup(destination)  # copies the bundled sources, no download
            verify_sources(destination)
            notebook = destination / S1.S1_NOTEBOOK
            notebook.write_bytes(notebook.read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                verify_sources(destination)

    @unittest.skipUnless(importlib.util.find_spec('shapely'), 'Shapely is required by training')
    def test_official_efd_map(self):
        with tempfile.TemporaryDirectory() as temporary:
            annotations = Path(temporary)
            square = ''.join(f'1,some,title,{x},{y}\n' for x, y in [(1, 1), (1, 5), (5, 5), (5, 1)])
            (annotations / 'chart').write_text(square)
            expected = np.zeros((8, 8), dtype=np.uint8)
            expected[1:6, 1:6] = 255
            result = D.make_efdmap(REFERENCE, 8, 8, [(2, 2)], 'chart', annotations)
            np.testing.assert_array_equal(np.asarray(result), expected)
            self.assertFalse(np.asarray(D.make_efdmap(REFERENCE, 8, 8, [], 'chart', annotations)).any())

    def test_evaluation_reproduces_published_scores_for_released_maps(self):
        from reproductions.umss_massvis import replication as R
        from models.umss.training.pipeline import predict as P
        _, names, paths = R.images()
        if not names or not R.encoding_files(names[0]):
            self.skipTest('Local MASSVIS inputs are required')
        name = names[0]
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            (work / 'upstream').symlink_to(REFERENCE)
            images = work / 'downloads/GT_EFD_500_2000_5000/val/raw_img'
            images.mkdir(parents=True)
            (images / f'{name}.png').symlink_to(paths[name])
            (work / 'downloads/massvis').mkdir()
            (work / 'downloads/massvis/fixationsByVis').symlink_to(R.DATA / 'massvis/fixationsByVis')
            (work / 'downloads/official').mkdir()
            (work / 'downloads/official/UMSS').symlink_to(R.RELEASED)
            checkpoint = work / 'checkpoint.hdf5'
            checkpoint.write_bytes(b'unused: inference is mocked')
            # "Our" maps are the released ones, so both rows must match the published scores.
            maps = [np.asarray(Image.open(R.RELEASED / 'salmaps' / f'{name}_{time}.png'), dtype=float) / 255
                    for time in P.TIMES]
            with patch.object(P, 'load_model', return_value=(None, None, None)), \
                    patch.object(P, 'predict', return_value=maps), contextlib.redirect_stdout(io.StringIO()):
                P.evaluate(SimpleNamespace(work=str(work), checkpoint=str(checkpoint)))
            with (work / 'predictions/paper_saliency_per_image.csv').open() as stream:
                rows = [row for row in csv.DictReader(stream) if row['source'] == 'official']
            published = R.published_saliency()
            for row in rows:
                expected = published[f"{int(row['time']) // 1000}s"].iloc[0]
                for metric in P.SALIENCY_METRICS:
                    self.assertAlmostEqual(float(row[metric]), expected[metric], places=10)


if __name__ == '__main__':
    unittest.main()
