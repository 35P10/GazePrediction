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

    def test_export_writes_maps_that_umss_reads_and_reuses_them(self):
        from models.umss import UMSS, export_maps
        from models.umss.training.pipeline import predict as P
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            (work / 'images').mkdir()
            image = work / 'images/chart.png'
            Image.new('RGB', (80, 60)).save(image)
            (work / 'ckpt').mkdir()
            (work / 'ckpt/md_eam.hdf5').write_bytes(b'unused: inference is mocked')
            with contextlib.redirect_stdout(io.StringIO()):
                setup(work / 'upstream')
            maps = [np.full((60, 80), value, dtype=np.float32) for value in (0.2, 0.4, 0.6)]
            versions = {'keras': SimpleNamespace(__version__='mock'), 'tensorflow': SimpleNamespace(__version__='mock')}
            args = SimpleNamespace(work=str(work), checkpoint=str(work / 'ckpt/md_eam.hdf5'),
                                   images=str(work / 'images'), out=str(work / 'predictions/charts'))
            chart_maps = Mock()
            with patch.object(P, 'load_model', return_value=(None, None, None)), \
                    patch.object(P, 'predict', return_value=maps), patch.object(P, 'save_chart_maps', chart_maps), \
                    patch.dict('sys.modules', versions), contextlib.redirect_stdout(io.StringIO()):
                P.export(args)
            self.assertEqual(chart_maps.call_args.args[-1], Path(args.out) / 'png')
            saved = UMSS(maps_dir=args.out).predict_maps(image)
            self.assertEqual((saved['width'], saved['height']), (80, 60))
            np.testing.assert_allclose(saved['maps'][:, 0, 0], [0.2, 0.4, 0.6])
            with patch('subprocess.Popen', side_effect=AssertionError('the export should be reused')):
                self.assertEqual(export_maps([image], 'charts', work=work), Path(args.out).resolve())

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
            def released_maps(work, model, pre, post, image, folder):
                for time in P.TIMES:
                    shutil.copyfile(R.RELEASED / 'salmaps' / f'{name}_{time}.png', folder / f'{name}_{time}.png')

            with patch.object(P, 'load_model', return_value=(None, None, None)), \
                    patch.object(P, 'save_chart_maps', released_maps), contextlib.redirect_stdout(io.StringIO()):
                P.evaluate(SimpleNamespace(work=str(work), checkpoint=str(checkpoint)))
            with (work / 'predictions/paper_saliency_per_image.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            published = R.published_saliency()
            for row in rows:
                expected = published[f"{int(row['time']) // 1000}s"].iloc[0]
                for metric in P.SALIENCY_METRICS:
                    self.assertAlmostEqual(float(row[metric]), expected[metric], places=10)


if __name__ == '__main__':
    unittest.main()
