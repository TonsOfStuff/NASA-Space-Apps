import unittest

import numpy as np

from scripts.blend_dem_overlap import blend_block


class BlendOverlapTests(unittest.TestCase):
    def setUp(self):
        self.base = np.zeros((80, 40), dtype=np.float32)
        self.base[:, 25:] = 100
        self.base[0, 0] = np.nan
        self.provenance = np.zeros(self.base.shape, dtype=np.uint8)
        self.provenance[0, 0] = 2
        self.earlier = np.zeros(self.base.shape, dtype=bool)
        self.earlier[:, :25] = True
        self.later = np.full(self.base.shape, 100, dtype=np.float32)
        self.later[:, :5] = np.nan
        self.excluded = np.zeros(self.base.shape, dtype=bool)
        self.excluded[35, 23] = True

    def test_blends_seam_without_changing_missing_or_excluded_cells(self):
        result, quality = blend_block(self.base, self.provenance, self.earlier, self.later, self.excluded, 8)
        self.assertTrue(np.isnan(result[0, 0]))
        self.assertEqual(quality[0, 0], 2)
        self.assertEqual(result[35, 23], self.base[35, 23])
        self.assertEqual(quality[35, 23], 0)
        self.assertEqual(result[40, 10], self.base[40, 10])
        self.assertGreater(result[40, 24], 0)
        self.assertEqual(quality[40, 24], 1)
        self.assertLess(abs(result[40, 25] - result[40, 24]), 100)

    def test_chunk_halos_match_whole_image(self):
        expected, expected_quality = blend_block(self.base, self.provenance, self.earlier, self.later, self.excluded, 8)
        for row in range(0, 80, 17):
            end = min(row + 17, 80)
            top, bottom = max(row - 8, 0), min(end + 8, 80)
            window = slice(top, bottom)
            core = slice(row - top, end - top)
            actual, quality = blend_block(
                self.base[window], self.provenance[window], self.earlier[window],
                self.later[window], self.excluded[window], 8,
            )
            np.testing.assert_allclose(actual[core], expected[row:end], equal_nan=True)
            np.testing.assert_array_equal(quality[core], expected_quality[row:end])


if __name__ == "__main__":
    unittest.main()
