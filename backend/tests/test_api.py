import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import rasterio
from rasterio.transform import from_origin

from spaceapps.api import DEFAULT_DEM, DEFAULT_MANIFEST, DEFAULT_TERRAIN, DEFAULT_THERMAL, RouteServer


ROOT = Path(__file__).resolve().parents[2]


class RouteApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        directory = Path(cls.temp.name)
        manifest = {
            "id": "test-region", "extent": [500000, 0, 500320, 320],
            "width": 16, "height": 16, "tileSize": 16,
            "minZoom": 0, "maxZoom": 0, "resolutions": [20],
        }
        manifest_path = directory / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        terrain_dir = directory / "terrain"
        (terrain_dir / "0/0").mkdir(parents=True)
        np.zeros((16, 16), dtype="<f4").tofile(terrain_dir / "0/0/0.f32")
        dem_path = directory / "dem.tif"
        with rasterio.open(
            dem_path, "w", driver="GTiff", width=16, height=16, count=1,
            dtype="float32", crs="EPSG:32610", transform=from_origin(500000, 320, 20, 20),
        ) as dem:
            dem.write(np.zeros((16, 16), dtype=np.float32), 1)
        thermal_path = directory / "thermal.cub"
        thermal_path.touch()
        cls.server = RouteServer(("127.0.0.1", 0), manifest_path, terrain_dir, dem_path, thermal_path, set())
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()
        cls.temp.cleanup()

    def test_default_paths_follow_repository_layout(self):
        self.assertEqual(DEFAULT_MANIFEST, ROOT / "frontend/public/data/gale/manifest.json")
        self.assertEqual(DEFAULT_TERRAIN, ROOT / "frontend/public/data/gale/tiles/terrain")
        self.assertEqual(DEFAULT_DEM, ROOT / "backend/data/gale_20m_interpolated_feathered.tif")
        self.assertEqual(DEFAULT_THERMAL, ROOT / "backend/data/thermal.cub")

    def test_health_and_route_over_http(self):
        with urlopen(f"{self.url}/api/health", timeout=30) as response:
            self.assertEqual(json.load(response), {"status": "ready"})

        payload = {
            "regionId": "test-region", "start": [500050, 270], "destination": [500250, 70],
            "rover": {"maxSlopeDeg": 15}, "weights": {"thermal": 0, "shade": 0},
        }
        request = Request(
            f"{self.url}/api/route", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urlopen(request, timeout=60) as response:
            result = json.load(response)
        self.assertEqual(result["status"], "reached")
        self.assertEqual(result["regionId"], "test-region")
        self.assertEqual(result["path"][0], [500050, 270])
        self.assertEqual(result["path"][-1], [500250, 70])


if __name__ == "__main__":
    unittest.main()
