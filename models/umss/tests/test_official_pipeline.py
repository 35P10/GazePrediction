"""Regression checks for official preparation and the shared Table 2 protocol."""
import contextlib
import csv
import importlib.util
import io
import json
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
from models.umss.reference.setup_umss import MANIFEST, setup, verify_sources

ROOT = Path(__file__).resolve().parents[3]
REFERENCE = S.DEFAULT_DESTINATION


class OfficialPipelineTests(unittest.TestCase):
    def test_standalone_docker_layout_imports_without_the_models_package(self):
        package = ROOT / 'models/umss'
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / 'umss'
            image.mkdir()
            for folder in ('reference', 'evaluation'):
                shutil.copytree(package / folder, image / folder,
                                ignore=shutil.ignore_patterns('__pycache__'))
            (image / 'training/pipeline').mkdir(parents=True)
            for name in ('s1.py', 'data_preparation.py'):
                shutil.copyfile(package / 'training' / name, image / 'training' / name)
            for name in ('train.py', 'prepare_data.py', 'predict.py', 'download.py'):
                shutil.copyfile(package / 'training/pipeline' / name,
                                image / 'training/pipeline' / name)
            shutil.copyfile(package / 'training/training_sources.json',
                            image / 'training/training_sources.json')
            (image / 'training/official_reference').symlink_to(REFERENCE)
            script = (
                "import sys; sys.path.insert(0, sys.argv[1]); "
                "from reference.setup_umss import DEFAULT_DESTINATION; "
                "from evaluation import saliency; from training import s1, data_preparation; "
                "assert s1.s1_defaults(DEFAULT_DESTINATION)['batch_size'] == 8; "
                "assert callable(saliency.score_saliency); "
                "assert callable(data_preparation.make_efdmap); "
                "assert 'models' not in sys.modules"
            )
            env = {**os.environ, 'PYTHONPATH': ''}
            subprocess.run([sys.executable, '-c', script, str(image)], cwd=temporary,
                           env=env, check=True, capture_output=True)
            for name in ('train.py', 'prepare_data.py', 'predict.py', 'download.py'):
                subprocess.run([sys.executable, str(image / 'training/pipeline' / name), '--help'],
                               cwd=temporary, env=env, check=True, capture_output=True)

    def test_s1_executes_the_original_recipe_and_applies_runtime_overrides(self):
        defaults = S1.s1_defaults(REFERENCE)
        self.assertEqual(defaults['n_epochs'], 9)
        self.assertEqual(defaults['batch_size'], 8)
        model = Mock()
        constructor = Mock(return_value=model)
        optimizer = Mock(return_value='original-Adam')
        generator = Mock(side_effect=lambda **kwargs: kwargs)
        scheduler = Mock(return_value='scheduler')
        data = {f'{kind}_files_{split}': [f'{split}-{kind}']
                for kind in ('img', 'map', 'fix') for split in ('train', 'val')}
        with tempfile.TemporaryDirectory() as temporary:
            bindings = dict(
                bp=temporary, ckpt_savedir=temporary, weightspath='initial-weights.hdf5',
                n_epochs=6, training_verbose=2, Adam=optimizer,
                ModelCheckpoint=Mock(return_value='checkpoint'),
                LearningRateScheduler=scheduler, InteractivePlot=Mock(return_value='csv-log'),
                load_datasets_multiduration=Mock(return_value=data),
                get_model_by_name=Mock(return_value=(constructor, 'fixture-mode')),
                create_losses=Mock(return_value=(['kl', 'cc', 'nss', 'ccmatch'],
                                                [10, -5, -1, 3], 'fixture', 3)),
                MultidurationGenerator=generator)
            for micro_batch in (8, 4):
                optimizer.reset_mock()
                options = {**bindings, 'batch_size': micro_batch}
                wrapper = None
                if micro_batch == 4:
                    options['opt'] = 'accumulation-Adam'
                    wrapper = lambda gen: ('accumulated', gen)
                with contextlib.redirect_stdout(io.StringIO()):
                    state = S1.run_s1(REFERENCE, options, wrapper)
                    actual = [state['step_decay'](epoch) for epoch in range(9)]
                np.testing.assert_allclose(actual, [1e-4, 1e-4, 1e-5, 1e-5, 1e-5,
                                                   1e-6, 1e-6, 1e-6, 1e-7], rtol=1e-14)
                constructor.assert_called_with(input_shape=(240, 320, 3), n_outs=4, nb_timestep=3)
                model.load_weights.assert_called_with('initial-weights.hdf5')
                self.assertEqual(state['gen_val']['batch_size'], 1)
                self.assertFalse(state['gen_val']['shuffle'])
                train = state['gen_train'] if micro_batch == 8 else state['gen_train'][1]
                self.assertEqual(train['batch_size'], micro_batch)
                self.assertTrue(train['shuffle'])
                model.fit_generator.assert_called_with(
                    state['gen_train'], epochs=6, verbose=2,
                    callbacks=['checkpoint', 'scheduler', 'csv-log'],
                    validation_data=state['gen_val'], max_queue_size=10, workers=5)
                if micro_batch == 8:
                    optimizer.assert_called_once_with(lr=0.0001)
                else:
                    optimizer.assert_not_called()

    def test_modified_official_code_is_rejected_before_execution(self):
        for relative, loader in [('Code/scripts/saliency/eval.py', S.saliency_functions),
                                 (S1.S1_NOTEBOOK, S1.s1_defaults)]:
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / relative
                path.parent.mkdir(parents=True)
                path.write_bytes((REFERENCE / relative).read_bytes() + b'\n')
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    loader(temporary)

    def test_source_setup_reuses_verified_bundled_files_and_extends_old_subset(self):
        manifest = json.loads(MANIFEST.read_text())
        previous = {**manifest, 'files': manifest['files'][:-3]}
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / 'upstream'
            destination.mkdir()
            for entry in previous['files']:
                path = destination / entry['path']
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(REFERENCE / entry['path'], path)
            (destination / 'sources.json').write_text(json.dumps(previous))
            with patch('models.umss.reference.setup_umss.urlopen', side_effect=AssertionError('Unexpected download')):
                with contextlib.redirect_stdout(io.StringIO()):
                    setup(destination)
            self.assertEqual(json.loads((destination / 'sources.json').read_text()), manifest)
            verify_sources(destination, manifest)

    @unittest.skipUnless(importlib.util.find_spec('shapely'), 'Shapely is required by training')
    def test_original_efd_handles_boundaries_empty_buckets_and_description_commas(self):
        with tempfile.TemporaryDirectory() as temporary:
            name = 'chart'
            annotations = Path(temporary)
            text = ''.join(f'1,some,title,{x},{y}\n' for x, y in [(1, 1), (1, 5), (5, 5), (5, 1)])
            (annotations / name).write_text(text)
            result = D.make_efdmap(REFERENCE, 8, 8, [(2, 2)], name, annotations)
            expected = np.zeros((8, 8), dtype=np.uint8)
            expected[1:6, 1:6] = 255
            np.testing.assert_array_equal(np.asarray(result), expected)
            for coords in ([], [(1, 1)]):
                result = D.make_efdmap(REFERENCE, 8, 8, coords, name, annotations)
                self.assertFalse(np.asarray(result).any())
            fix, heat = S.make_fixmap_and_heatmap(REFERENCE, 8, 8, [])
            self.assertFalse(np.asarray(fix).any())
            self.assertFalse(np.asarray(heat).any())

    def test_training_evaluation_reproduces_published_scores_without_overwriting_old_reports(self):
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
            humans = work / 'downloads/massvis'
            humans.mkdir()
            (humans / 'fixationsByVis').symlink_to(R.DATA / 'massvis/fixationsByVis')
            released = work / 'downloads/official'
            released.mkdir()
            (released / 'UMSS').symlink_to(R.RELEASED)
            checkpoint = work / 'checkpoint.hdf5'
            checkpoint.write_bytes(b'fixture: neural inference is mocked')
            (work / 'predictions').mkdir()
            legacy = work / 'predictions/evaluation.csv'
            legacy.write_text('previous diagnostic\n')
            maps = [np.asarray(Image.open(R.RELEASED / 'salmaps' / f'{name}_{time}.png'),
                               dtype=float) / 255 for time in P.TIMES]
            args = SimpleNamespace(work=str(work), checkpoint=str(checkpoint))
            with patch.object(P, 'load_model', return_value=(None, None, None)), \
                    patch.object(P, 'predict', return_value=maps), \
                    contextlib.redirect_stdout(io.StringIO()):
                P.evaluate(args)
            with (work / 'predictions/paper_saliency_per_image.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4)
            published = R.published_saliency()
            for row in rows:
                if row['source'] == 'official':
                    duration = f"{int(row['time']) // 1000}s"
                    for metric in P.SALIENCY_METRICS:
                        self.assertAlmostEqual(float(row[metric]), published[duration].iloc[0][metric], places=10)
            metadata = json.loads((work / 'predictions/paper_saliency_summary.json').read_text())
            self.assertEqual(metadata['images'], 1)
            self.assertEqual(metadata['sigma_px'], 19)
            self.assertEqual(set(metadata['means']), {'3000', '5000'})
            self.assertEqual(legacy.read_text(), 'previous diagnostic\n')
            self.assertEqual(len(list((work / 'predictions/val').glob('*.png'))), 3)


if __name__ == '__main__':
    unittest.main()
