"""UMSS model checks without neural inference; UMSS.run() on real data is validated in
reproductions/umss_massvis/01_validate_s1_saliency.ipynb."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image

from models.umss import UMSS, UMSSUtils, S2_FIXATION_COUNTS, sample_scanpath


def single_point_maps():
    """All mass at grid (x=200, y=100), so every fixation lands there."""
    maps = np.zeros((3, 480, 640), dtype=np.float32)
    maps[:, 100, 200] = 1
    return maps


class UMSSTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.maps = single_point_maps()
        np.savez(self.folder / 'chart.npz', maps=self.maps, width=768, height=768)
        self.model = UMSS(maps_dir=self.folder)
        self.inputs = [{'image_id': 'chart', 'image_path': 'chart.png'}]

    def test_run_in_memory(self):
        output = self.model.run(self.inputs, n_runs=2, n_fix=3, seed=17)
        np.testing.assert_array_equal(output['maps']['chart'], self.maps)
        scanpaths = output['scanpaths']
        self.assertEqual(scanpaths.groupby('run').size().tolist(), [3, 3])
        self.assertEqual(scanpaths.seed.unique().tolist(), [17, 18])
        # Grid (200, 100) on a 640x480 grid is (240, 160) in a 768x768 image.
        self.assertTrue((scanpaths.grid_x == 200).all() and (scanpaths.grid_y == 100).all())
        self.assertTrue((scanpaths.x == 240).all() and (scanpaths.y == 160).all())

    def test_run_to_folder_matches_predict(self):
        folder = self.model.run(self.inputs, self.folder / 'out', n_runs=2, n_fix=3, seed=17300)
        saved = pd.read_csv(folder / 'scanpaths.csv', float_precision='round_trip')
        self.assertEqual(len(saved), 6)
        np.testing.assert_array_equal(np.load(folder / 'maps/chart.npy'), self.maps)
        single = self.model.predict('chart.png', n_fix=3, seed=17300)
        self.assertEqual(saved[saved.run == 0].duration.tolist(), single['durations'])

    def test_s2_is_reproducible_and_ordered(self):
        maps = np.random.default_rng(0).random((3, 480, 640)) ** 8
        scanpath = sample_scanpath(maps, n_fix=None, seed=4)
        self.assertEqual(scanpath, sample_scanpath(maps, n_fix=None, seed=4))
        self.assertIn(len(scanpath['grid_points']), S2_FIXATION_COUNTS)
        self.assertEqual(scanpath['slices'], sorted(scanpath['slices']))

    def test_preprocessing_preview_without_tensorflow(self):
        image = self.folder / 'color.png'
        Image.new('RGB', (768, 768), (17, 93, 241)).save(image)
        had_tensorflow = 'tensorflow' in sys.modules
        preview = UMSSUtils.inputs(image)
        self.assertEqual(preview['tensor'].shape, (1, 240, 320, 3))
        self.assertEqual(preview['model image'][120, 160].tolist(), [17, 93, 241])
        self.assertEqual('tensorflow' in sys.modules, had_tensorflow)


if __name__ == '__main__':
    unittest.main()
