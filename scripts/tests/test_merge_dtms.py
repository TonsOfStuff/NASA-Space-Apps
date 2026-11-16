import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from scripts.merge_dtms import DEFAULT_OUTPUT, DEFAULT_TILES
from scripts.prepare_gale import DEFAULT_DEM, DEFAULT_IMAGERY, DEFAULT_OUTPUT as DEFAULT_TILE_OUTPUT


ROOT = Path(__file__).resolve().parents[2]


class MergeDtmTests(unittest.TestCase):
    def test_default_paths_follow_repository_layout(self):
        self.assertEqual(DEFAULT_TILES, ROOT / "backend/data/DTMs")
        self.assertEqual(DEFAULT_OUTPUT, ROOT / "backend/data/gale_20m_mosaic.tif")
        self.assertEqual(DEFAULT_DEM, ROOT / "backend/data/gale_20m_interpolated_feathered.tif")
        self.assertEqual(DEFAULT_IMAGERY, ROOT / "backend/data/curiosity_ctx_mosaic.tif")
        self.assertEqual(DEFAULT_TILE_OUTPUT, ROOT / "frontend/public/data/gale")

    def test_cli_merges_tiles_and_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            tile_dir = directory / "tiles"
            tile_dir.mkdir()
            for index in range(2):
                with rasterio.open(
                    tile_dir / f"tile_{index}_20m.tif", "w", driver="GTiff",
                    width=2, height=2, count=1, dtype="float32", crs="EPSG:32610",
                    transform=from_origin(index * 2, 2, 1, 1), nodata=-9999,
                ) as tile:
                    tile.write(np.full((2, 2), index + 1, dtype=np.float32), 1)

            output = directory / "mosaic.tif"
            command = [sys.executable, str(ROOT / "scripts/merge_dtms.py"), "--tiles", str(tile_dir), "--output", str(output)]
            first = subprocess.run(command, cwd=directory, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            with rasterio.open(output) as mosaic:
                np.testing.assert_array_equal(mosaic.read(1), [[1, 1, 2, 2], [1, 1, 2, 2]])

            second = subprocess.run(command, cwd=directory, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("Output already exists", second.stderr)
            with rasterio.open(output) as mosaic:
                np.testing.assert_array_equal(mosaic.read(1), [[1, 1, 2, 2], [1, 1, 2, 2]])


if __name__ == "__main__":
    unittest.main()
