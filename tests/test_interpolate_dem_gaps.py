import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from scripts.interpolate_dem_gaps import interpolate_gaps, process, smooth_valid_elevation


class InterpolateDemGapsTests(unittest.TestCase):
    def test_single_cell_uses_all_eight_neighbors_with_spacing_weights(self):
        source = np.array([[0, 2, 4], [6, np.nan, 10], [12, 14, 16]], dtype=np.float32)
        source[0, 0] = 20  # Make the diagonal contribution visible.
        result, provenance, stats = interpolate_gaps(source, x_spacing=1, y_spacing=2)
        weighted = (6 + 10) * 1 + (2 + 14) * .25 + (20 + 4 + 12 + 16) * .2
        expected = weighted / (2 + .5 + .8)
        self.assertAlmostEqual(float(result[1, 1]), expected, places=5)
        self.assertEqual(provenance[1, 1], 1)
        self.assertEqual(stats.estimated_cells, 1)
        np.testing.assert_array_equal(result[np.isfinite(source)], source[np.isfinite(source)])
        self.assertTrue(np.isnan(source[1, 1]))

    def test_edge_connected_narrow_seam_fills_but_raster_edges_remain_missing(self):
        rows, cols = np.indices((8, 8))
        source = (2 * rows + 3 * cols).astype(np.float32)
        source[:, 3] = np.nan
        result, provenance, stats = interpolate_gaps(source, x_spacing=1, y_spacing=1, max_gap_width=1)
        self.assertTrue(np.isnan(result[0, 3]))
        self.assertTrue(np.isnan(result[-1, 3]))
        # At the open ends the unavailable edge pixel contributes no invented
        # boundary; the estimate stays close to the planar surface.
        np.testing.assert_allclose(result[1:-1, 3], (2 * rows + 3 * cols)[1:-1, 3], atol=0.5)
        np.testing.assert_array_equal(provenance[1:-1, 3], 1)
        self.assertEqual(stats.estimated_cells, 6)

    def test_broad_and_one_sided_missing_area_remains_unknown(self):
        source = np.ones((12, 12), dtype=np.float32)
        source[:, :4] = np.nan  # one-sided exterior
        source[:, 6:9] = np.nan  # three-pixel gap, wider than the limit
        result, provenance, stats = interpolate_gaps(source, x_spacing=1, y_spacing=1, max_gap_width=2)
        self.assertEqual(stats.estimated_cells, 0)
        np.testing.assert_array_equal(np.isnan(result), np.isnan(source))
        self.assertTrue(np.all(provenance[np.isnan(source)] == 2))

    def test_component_size_limit_leaves_long_seam_untouched(self):
        source = np.ones((12, 12), dtype=np.float32)
        source[:, 5] = np.nan
        result, provenance, stats = interpolate_gaps(source, x_spacing=1, y_spacing=1, max_component_cells=4)
        self.assertEqual(stats.skipped_regions, 1)
        self.assertEqual(stats.estimated_cells, 0)
        self.assertTrue(np.isnan(result[:, 5]).all())
        self.assertTrue((provenance[:, 5] == 2).all())

    def test_smoothing_preserves_unknowns_and_does_not_mutate_input(self):
        source = np.array([[0, 0, np.nan], [0, 10, np.nan], [0, 0, np.nan]], dtype=np.float32)
        before = source.copy()
        result = smooth_valid_elevation(source, sigma=1)
        np.testing.assert_equal(source, before)
        np.testing.assert_array_equal(np.isnan(result), np.isnan(source))
        self.assertLess(result[1, 1], source[1, 1])
        self.assertGreater(result[0, 0], 0)

    def test_geotiffs_preserve_reference_and_provenance(self):
        source = np.arange(49, dtype=np.float32).reshape(7, 7)
        source[3, 3] = np.nan
        source[:, 0] = np.nan
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            original, output, provenance, smooth = [base / name for name in ("original.tif", "derived.tif", "provenance.tif", "smooth.tif")]
            transform = from_origin(100, 200, 20, 20)
            with rasterio.open(original, "w", driver="GTiff", width=7, height=7, count=1, dtype="float32", crs="EPSG:32610", transform=transform, nodata=float("nan")) as dst:
                dst.write(source, 1)
            stats = process(original, output, provenance, max_gap_width=2, max_component_cells=4096, smooth_output=smooth, smooth_sigma=1)
            self.assertEqual(stats.estimated_cells, 1)
            with rasterio.open(output) as derived, rasterio.open(provenance) as mask, rasterio.open(smooth) as softened:
                self.assertEqual(derived.transform, transform)
                self.assertEqual(mask.transform, transform)
                self.assertEqual(derived.crs, rasterio.crs.CRS.from_epsg(32610))
                self.assertEqual(mask.read(1)[3, 3], 1)
                self.assertTrue(np.all(mask.read(1)[:, 0] == 2))
                self.assertTrue(np.isnan(softened.read(1)[:, 0]).all())
                self.assertEqual(softened.tags()["SIGMA_PIXELS"], "1")
            with self.assertRaises(FileExistsError):
                process(original, output, provenance, max_gap_width=2, max_component_cells=4096)

    def test_numeric_nodata_and_scale_metadata_are_preserved(self):
        source = np.arange(49, dtype=np.float32).reshape(7, 7)
        source[3, 3] = -9999
        source[:, 0] = -9999
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            original, output, provenance = [base / name for name in ("original.tif", "derived.tif", "provenance.tif")]
            with rasterio.open(original, "w", driver="GTiff", width=7, height=7, count=1, dtype="float32", crs="EPSG:32610", transform=from_origin(0, 140, 20, 20), nodata=-9999) as dst:
                dst.write(source, 1)
                dst.scales = (2.0,)
                dst.offsets = (5.0,)
            process(original, output, provenance, max_gap_width=2, max_component_cells=4096)
            with rasterio.open(output) as derived, rasterio.open(provenance) as mask:
                self.assertEqual(derived.nodata, -9999)
                self.assertEqual(derived.scales, (2.0,))
                self.assertEqual(derived.offsets, (5.0,))
                self.assertTrue(np.all(derived.read(1)[:, 0] == -9999))
                self.assertEqual(mask.read(1)[3, 3], 1)


if __name__ == "__main__":
    unittest.main()
