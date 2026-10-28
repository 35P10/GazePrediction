"""Adapter checks that need no map data; UMSS.run() is validated on real data in
reproductions/umss_massvis/01_reproduce_umss.ipynb."""
from pathlib import Path
import tempfile
import unittest

from models import load_predictor

ROOT = Path(__file__).resolve().parents[3]
SALTINET = ROOT / 'reproductions/umss_massvis/official_repository/Code/scripts/UMSS/saltinet_util.py'


class UMSSTests(unittest.TestCase):
    @unittest.skipUnless(SALTINET.is_file(), 'DaRUS download missing (reproductions/umss_massvis notebook)')
    def test_umss_s2_distributions_match_released_code(self):
        import ast
        from models import umss

        values = {}
        for function in ast.parse(SALTINET.read_text()).body:
            if isinstance(function, ast.FunctionDef):
                for node in function.body:
                    if isinstance(node, ast.Assign) and node.targets[0].id in ('p', 'bins'):
                        values[function.name, node.targets[0].id] = tuple(ast.literal_eval(node.value))
        self.assertEqual(values['get_number_fixations', 'p'], umss.S2_FIXATION_COUNT_P)
        self.assertEqual(tuple(int(b) for b in values['get_number_fixations', 'bins']), umss.S2_FIXATION_COUNTS)
        self.assertEqual(values['get_duration_fixation', 'p'], umss.S2_DURATION_P)
        self.assertEqual(values['get_duration_fixation', 'bins'], umss.S2_DURATIONS_S)

    def test_typed_run_preserves_maps_seeds_and_image_coordinates(self):
        import numpy as np
        import pandas as pd
        from PIL import Image
        from models.umss import UMSS, sha256
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            image=root/'fixture.png';Image.new('RGB',(768,768)).save(image)
            maps=np.zeros((3,480,640),dtype=np.float32)
            maps[:,100,200]=1
            np.savez(root/'fixture.npz',maps=maps,width=768,height=768,image_sha256=sha256(image))
            model=UMSS(maps_dir=root)
            inputs: list = [{'image_id':'fixture','image_path':image}]
            folder=model.run(inputs,root/'out',n_runs=2,n_fix=10,seed=17300)
            saved=pd.read_csv(folder/'scanpaths.csv',float_precision='round_trip')
            self.assertEqual(len(saved),20)
            self.assertEqual(set(saved.seed),{17300,17301})
            self.assertTrue((saved.x==240).all() and (saved.y==160).all())
            self.assertTrue((saved.grid_x==200).all() and (saved.grid_y==100).all())
            self.assertTrue(np.array_equal(np.load(folder/'maps/fixture.npy'),maps))
            repeated=model.predict(image,n_fix=10,seed=17300)
            self.assertTrue(np.array_equal(saved[saved.run==0].duration,repeated['durations']))
            model.run(inputs,folder,n_runs=1,n_fix=5,seed=17302)
            self.assertEqual(len(pd.read_csv(folder/'scanpaths.csv')),5)
            with self.assertRaisesRegex(ValueError,'different image'):
                Image.new('RGB',(768,768),'red').save(image)
                model.predict_maps(image)

    def test_preprocessing_preview_is_exact_upstream_without_tensorflow(self):
        import ast
        import sys
        import numpy as np
        import cv2
        from PIL import Image
        from models.umss import UMSSUtils
        source=ROOT/'reproductions/umss_massvis/official_repository/Code/scripts/saliency/sal_imp_utilities.py'
        if not source.is_file():self.skipTest('Verified upstream sources are not installed')
        tree=ast.parse(source.read_text())
        functions=[node for node in tree.body if isinstance(node,ast.FunctionDef)
                   and node.name in ('padding','preprocess_images')]
        namespace=dict(np=np,cv2=cv2)
        exec(compile(ast.Module(body=functions,type_ignores=[]),'upstream','exec'),namespace)
        with tempfile.TemporaryDirectory() as temporary:
            image=Path(temporary)/'fixture.png';Image.new('RGB',(768,768),(17,93,241)).save(image)
            had_tf='tensorflow' in sys.modules
            preview=UMSSUtils.inputs(image)
            expected=namespace['preprocess_images']([str(image)],240,320)
            self.assertTrue(np.array_equal(preview['tensor'],expected))
            self.assertEqual(preview['model image'].shape,(240,320,3))
            self.assertTrue(np.array_equal(preview['model image'][120,160],[17,93,241]))
            self.assertEqual('tensorflow' in sys.modules,had_tf)

    def test_missing_checkpoint_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = load_predictor('umss', checkpoint=Path(temporary) / 'missing.hdf5')
            with self.assertRaisesRegex(FileNotFoundError, 'maps_dir'):
                model.input_fingerprint('chart.png')

    def test_run_returns_maps_and_paths_without_exporting_files(self):
        import numpy as np
        import pandas as pd
        from unittest.mock import patch
        from models.umss import UMSS

        maps = np.zeros((3, 480, 640), dtype=np.float32)
        maps[:, 100, 200] = 1
        model = UMSS(maps_dir='unused')
        model.predict_maps = lambda path: dict(maps=maps, width=768, height=768)
        inputs = [{'image_id': 'fixture', 'image_path': 'unused.png'}]
        with patch('numpy.save', side_effect=AssertionError('Map export attempted')), patch.object(
                pd.DataFrame, 'to_csv', side_effect=AssertionError('CSV export attempted')):
            output = model.run(inputs, n_runs=2, n_fix=3, seed=17)

        np.testing.assert_array_equal(output['maps']['fixture'], maps)
        paths = output['scanpaths']
        self.assertEqual(paths.groupby('run').size().tolist(), [3, 3])
        self.assertEqual(paths.seed.unique().tolist(), [17, 18])
        self.assertTrue(paths.x.eq(240).all() and paths.y.eq(160).all())


if __name__ == '__main__':
    unittest.main()
