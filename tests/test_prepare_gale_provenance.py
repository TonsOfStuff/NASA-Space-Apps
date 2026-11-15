import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_origin

from scripts.prepare_gale import (
    ESTIMATED_RGBA,
    MISSING_RGBA,
    TILE_SIZE,
    downsample_children,
    elevation_stats,
    resolve_provenance_path,
    save_terrain,
    tile_path,
    validate_provenance_metadata,
    write_derived_images,
)


class GaleProvenanceTests(unittest.TestCase):
    def test_native_colors_and_conservative_parent_priority(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            n = TILE_SIZE
            for y in range(2):
                for x in range(2):
                    terrain = np.full((n, n), 10, dtype=np.float32)
                    estimated = np.zeros((n, n), dtype=bool)
                    if (x, y) == (0, 0):
                        terrain[0, 0] = np.nan
                        estimated[0, 1] = True  # Same parent as a missing child.
                        estimated[2, 2] = True  # All four children known.
                    save_terrain(output, 6, x, y, terrain)
                    write_derived_images(output, 6, x, y, terrain, estimated, n * 2, n * 2, 0, 20)
            native = np.asarray(Image.open(tile_path(output, "coverage", 6, 0, 0, "png")))
            np.testing.assert_array_equal(native[0, 0], MISSING_RGBA)
            np.testing.assert_array_equal(native[0, 1], ESTIMATED_RGBA)
            np.testing.assert_array_equal(native[5, 5], [0, 0, 0, 0])
            native_estimated = np.asarray(Image.open(tile_path(output, "coverage_estimated", 6, 0, 0, "png")))
            native_missing = np.asarray(Image.open(tile_path(output, "coverage_missing", 6, 0, 0, "png")))
            np.testing.assert_array_equal(native_estimated[0, 1], ESTIMATED_RGBA)
            np.testing.assert_array_equal(native_estimated[0, 0], [0, 0, 0, 0])
            np.testing.assert_array_equal(native_missing[0, 0], MISSING_RGBA)
            np.testing.assert_array_equal(native_missing[0, 1], [0, 0, 0, 0])
            parent, _, estimated_parent = downsample_children(output, 5, 0, 0)
            self.assertTrue(np.isnan(parent[0, 0]))
            self.assertFalse(estimated_parent[0, 0])  # Missing outranks estimated.
            self.assertTrue(estimated_parent[1, 1])
            write_derived_images(output, 5, 0, 0, parent, estimated_parent, n, n, 0, 20)
            parent_coverage = np.asarray(Image.open(tile_path(output, "coverage", 5, 0, 0, "png")))
            np.testing.assert_array_equal(parent_coverage[0, 0], MISSING_RGBA)
            np.testing.assert_array_equal(parent_coverage[1, 1], ESTIMATED_RGBA)
            np.testing.assert_array_equal(parent_coverage[2, 2], [0, 0, 0, 0])
            parent_estimated = np.asarray(Image.open(tile_path(output, "coverage_estimated", 5, 0, 0, "png")))
            parent_missing = np.asarray(Image.open(tile_path(output, "coverage_missing", 5, 0, 0, "png")))
            np.testing.assert_array_equal(parent_estimated[0, 0], [0, 0, 0, 0])
            np.testing.assert_array_equal(parent_missing[0, 0], MISSING_RGBA)
            np.testing.assert_array_equal(parent_estimated[1, 1], ESTIMATED_RGBA)
            np.testing.assert_array_equal(parent_missing[1, 1], [0, 0, 0, 0])

    def test_manifest_source_resolution_and_provenance_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dem_path, provenance_path, bad_path = (root / name for name in ("dem.tif", "p.tif", "bad.tif"))
            dem = np.ones((4, 4), dtype=np.float32)
            dem[0, 0] = np.nan
            classes = np.zeros((4, 4), dtype=np.uint8)
            classes[0, 0] = 2
            classes[1, 1] = 1
            kwargs = dict(driver="GTiff", width=4, height=4, count=1, crs="EPSG:32610", transform=from_origin(0, 80, 20, 20))
            with rasterio.open(dem_path, "w", **kwargs, dtype="float32", nodata=float("nan")) as writer:
                writer.write(dem, 1)
                writer.update_tags(PROVENANCE=provenance_path.name)
            with rasterio.open(provenance_path, "w", **kwargs, dtype="uint8") as writer:
                writer.write(classes, 1)
            with rasterio.open(bad_path, "w", **{**kwargs, "transform": from_origin(10, 80, 20, 20)}, dtype="uint8") as writer:
                writer.write(classes, 1)
            with rasterio.open(dem_path) as source, rasterio.open(provenance_path) as provenance, rasterio.open(bad_path) as bad:
                self.assertEqual(resolve_provenance_path(dem_path, source, None), provenance_path.resolve())
                self.assertEqual(resolve_provenance_path(dem_path, source, bad_path), bad_path)
                validate_provenance_metadata(source, provenance)
                with self.assertRaises(ValueError):
                    validate_provenance_metadata(source, bad)
                self.assertEqual(elevation_stats(source, provenance)[2:], (15, 1))
                self.assertEqual(elevation_stats(source, None)[2:], (15, 0))
            classes[0, 0] = 0  # NoData cell must never be labeled as original.
            with rasterio.open(provenance_path, "r+") as writer:
                writer.write(classes, 1)
            with rasterio.open(dem_path) as source, rasterio.open(provenance_path) as provenance:
                with self.assertRaises(ValueError):
                    elevation_stats(source, provenance)


if __name__ == "__main__":
    unittest.main()
