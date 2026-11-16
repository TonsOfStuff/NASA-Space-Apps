"""Local HTTP API connecting the Mars map to the team's terrain router.

Run with: python -m spaceapps.api
The frontend sends projected Mars coordinates to POST /api/route.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import threading
from contextlib import suppress
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.errors import WindowError
from rasterio.transform import array_bounds, from_origin
from rasterio.warp import reproject, transform as transform_coordinates, transform_bounds
from rasterio.windows import Window, from_bounds

# new_algo also includes an interactive plot; an API should not open a window.
os.environ.setdefault("MPLBACKEND", "Agg")
from .new_algo import CostLayer, TerrainRouter, compute_illumination  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "web/public/data/gale/manifest.json"
DEFAULT_TERRAIN = ROOT / "web/public/data/gale/tiles/terrain"
DEFAULT_DEM = ROOT / "src/spaceapps/data/gale_20m_interpolated_feathered.tif"
DEFAULT_THERMAL = ROOT / "src/spaceapps/data/thermal.cub"
MAX_CELLS = 10_000_000

# Numba's workqueue threading layer aborts the process if parallel kernels
# overlap across Python threads. Both route construction and shade tiles call
# the same parallel shadow kernel, so they must share one lock.
SHADOW_KERNEL_LOCK = threading.Lock()


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def _point(value, name: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value):
        raise ApiError(400, "invalid_coordinate", f"{name} must contain two finite projected Mars coordinates.")
    return float(value[0]), float(value[1])


def _solar_inputs(mars_time, latitude: float) -> tuple[float, float, float, list[str]]:
    """Estimate a solar direction from local time, assuming an equinox.

    A mission-relative sol has no absolute season without the mission's epoch.
    """
    if mars_time is None:
        return 270.0, 30.0, 0.0, ["Sunlight was not included: no Mars date and time was supplied."]
    if not isinstance(mars_time, dict):
        raise ApiError(400, "invalid_mars_time", "marsDateTime must be an object.")
    sol, clock = mars_time.get("sol"), mars_time.get("localSolarTime")
    if isinstance(sol, bool) or not isinstance(sol, int) or sol < 0 or mars_time.get("solReference") != "mission" or not isinstance(clock, str) or not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", clock):
        raise ApiError(400, "invalid_mars_time", "Provide a non-negative mission sol and local solar time in HH:MM format.")
    hour, minute = map(int, clock.split(":"))
    hour_angle = math.radians(15 * (hour + minute / 60 - 12))
    lat = math.radians(latitude)
    east = -math.sin(hour_angle)
    north = -math.sin(lat) * math.cos(hour_angle)
    up = math.cos(lat) * math.cos(hour_angle)
    altitude = math.degrees(math.asin(max(-1.0, min(1.0, up))))
    azimuth = math.degrees(math.atan2(east, north)) % 360
    warnings = ["Sun direction is an equinox approximation; the mission sol has no configured epoch or seasonal model."]
    if altitude <= 0:
        warnings.append("The approximate Sun is below the horizon; sunlight was not used as a routing preference.")
        return azimuth, 1.0, 0.0, warnings
    return azimuth, altitude, 2.0, warnings


def _terrain_window(manifest: dict, terrain_dir: Path, start, destination):
    """Read a bounded route area from the map's conservative terrain pyramid."""
    left, bottom, right, top = map(float, manifest["extent"])
    for label, (x, y) in (("start", start), ("destination", destination)):
        if not left <= x < right or not bottom < y <= top:
            raise ApiError(422, "outside_region", f"The {label} is outside the available region.")

    distance = math.hypot(destination[0] - start[0], destination[1] - start[1])
    margin = max(2000.0, distance * 0.4)
    bounds = (max(left, min(start[0], destination[0]) - margin),
              max(bottom, min(start[1], destination[1]) - margin),
              min(right, max(start[0], destination[0]) + margin),
              min(top, max(start[1], destination[1]) + margin))
    max_zoom, tile_size = int(manifest["maxZoom"]), int(manifest["tileSize"])
    resolutions = manifest["resolutions"]
    selected = None
    for zoom in range(max_zoom, int(manifest["minZoom"]) - 1, -1):
        resolution = float(resolutions[zoom])
        width = math.ceil((right - left) / resolution)
        height = math.ceil((top - bottom) / resolution)
        col0 = max(0, math.floor((bounds[0] - left) / resolution))
        row0 = max(0, math.floor((top - bounds[3]) / resolution))
        col1 = min(width, math.ceil((bounds[2] - left) / resolution))
        row1 = min(height, math.ceil((top - bounds[1]) / resolution))
        if (col1 - col0) * (row1 - row0) <= MAX_CELLS:
            selected = zoom, resolution, col0, row0, col1, row1
            break
    if selected is None:
        raise ApiError(422, "route_area_too_large", "The selected area is too large for the available terrain levels.")

    zoom, resolution, col0, row0, col1, row1 = selected
    elevation = np.full((row1 - row0, col1 - col0), np.nan, dtype=np.float32)
    for tile_y in range(row0 // tile_size, (row1 - 1) // tile_size + 1):
        for tile_x in range(col0 // tile_size, (col1 - 1) // tile_size + 1):
            path = terrain_dir / str(zoom) / str(tile_x) / f"{tile_y}.f32"
            try:
                values = np.fromfile(path, dtype="<f4")
            except OSError as exc:
                raise ApiError(503, "terrain_unavailable", f"Terrain tile is unavailable: {path.name}.") from exc
            if values.size != tile_size * tile_size:
                raise ApiError(503, "terrain_unavailable", f"Terrain tile has an invalid size: {path.name}.")
            tile = values.reshape(tile_size, tile_size)
            x0, y0 = max(col0, tile_x * tile_size), max(row0, tile_y * tile_size)
            x1, y1 = min(col1, (tile_x + 1) * tile_size), min(row1, (tile_y + 1) * tile_size)
            elevation[y0 - row0:y1 - row0, x0 - col0:x1 - col0] = tile[y0 - tile_y * tile_size:y1 - tile_y * tile_size, x0 - tile_x * tile_size:x1 - tile_x * tile_size]
    transform = from_origin(left + col0 * resolution, top - row0 * resolution, resolution, resolution)
    return elevation, transform, resolution, zoom


def _thermal_window(path: Path, dem_crs, transform, shape) -> np.ndarray:
    """Crop the ISIS cube before warping; preserve special pixels as NoData."""
    try:
        with rasterio.open(path) as src:
            if src.crs is None:
                raise ApiError(503, "thermal_unavailable", "The thermal dataset has no coordinate system.")
            bounds = transform_bounds(dem_crs, src.crs, *array_bounds(shape[0], shape[1], transform), densify_pts=21)
            win = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
            win = Window(win.col_off - 2, win.row_off - 2, win.width + 4, win.height + 4)
            try:
                win = win.intersection(Window(0, 0, src.width, src.height))
            except WindowError:
                return np.full(shape, np.nan, dtype=np.float32)
            raw = src.read(1, window=win, masked=True).astype(np.float32).filled(np.nan)
            raw[raw <= -1e30] = np.nan
            raw = raw * src.scales[0] + src.offsets[0]
            thermal = np.full(shape, np.nan, dtype=np.float32)
            reproject(source=raw, destination=thermal, src_transform=src.window_transform(win), src_crs=src.crs,
                      src_nodata=np.nan, dst_transform=transform, dst_crs=dem_crs, dst_nodata=np.nan,
                      resampling=Resampling.bilinear)
            return thermal
    except ApiError:
        raise
    except (OSError, rasterio.errors.RasterioError, ValueError) as exc:
        raise ApiError(503, "thermal_unavailable", f"Could not read thermal inertia data: {exc}") from exc


def _tile_terrain(manifest: dict, terrain_dir: Path, zoom: int, tile_x: int, tile_y: int, halo: int) -> np.ndarray:
    """Read one terrain tile and its neighbours, with NaN outside the dataset."""
    size = int(manifest["tileSize"])
    result = np.full((size + 2 * halo, size + 2 * halo), np.nan, dtype=np.float32)
    col0, row0 = tile_x * size - halo, tile_y * size - halo
    col1, row1 = col0 + result.shape[1], row0 + result.shape[0]
    scale = 2 ** (int(manifest["maxZoom"]) - zoom)
    cols = math.ceil(int(manifest["width"]) / scale)
    rows = math.ceil(int(manifest["height"]) / scale)
    for ty in range(max(0, row0 // size), min((rows - 1) // size, (row1 - 1) // size) + 1):
        for tx in range(max(0, col0 // size), min((cols - 1) // size, (col1 - 1) // size) + 1):
            path = terrain_dir / str(zoom) / str(tx) / f"{ty}.f32"
            try:
                values = np.fromfile(path, dtype="<f4")
            except OSError as exc:
                raise ApiError(503, "terrain_unavailable", "A terrain tile could not be loaded.") from exc
            if values.size != size * size:
                raise ApiError(503, "terrain_unavailable", "A terrain tile has an invalid size.")
            x0, y0 = max(col0, tx * size), max(row0, ty * size)
            x1, y1 = min(col1, (tx + 1) * size), min(row1, (ty + 1) * size)
            result[y0 - row0:y1 - row0, x0 - col0:x1 - col0] = values.reshape(size, size)[y0 - ty * size:y1 - ty * size, x0 - tx * size:x1 - tx * size]
    return result


@lru_cache(maxsize=128)
def _map_tile(manifest_path: str, manifest_version: int, terrain_dir: str, dem_path: str,
              thermal_path: str, layer: str, zoom: int, tile_x: int, tile_y: int, clock: str) -> bytes:
    """Generate compressed tiles in memory; do not duplicate the source rasters on disk."""
    del manifest_version  # Included in the cache key so regenerated maps invalidate old tiles.
    manifest = json.loads(Path(manifest_path).read_text())
    size = int(manifest["tileSize"])
    if zoom < int(manifest["minZoom"]) or zoom > int(manifest["maxZoom"]):
        raise ApiError(404, "tile_not_found", "Tile is outside the available zoom levels.")
    scale = 2 ** (int(manifest["maxZoom"]) - zoom)
    cols, rows = math.ceil(int(manifest["width"]) / scale), math.ceil(int(manifest["height"]) / scale)
    if tile_x < 0 or tile_y < 0 or tile_x * size >= cols or tile_y * size >= rows:
        raise ApiError(404, "tile_not_found", "Tile is outside the available region.")
    resolution = float(manifest["resolutions"][zoom])
    left, _, _, top = map(float, manifest["extent"])
    transform = from_origin(left + tile_x * size * resolution, top - tile_y * size * resolution, resolution, resolution)
    with rasterio.open(dem_path) as dem:
        if dem.crs is None:
            raise ApiError(503, "terrain_unavailable", "The terrain dataset has no coordinate system.")
        dem_crs = dem.crs
    rgba = np.zeros((size, size, 4), dtype=np.uint8)

    if layer == "thermal":
        values = _thermal_window(Path(thermal_path), dem_crs, transform, (size, size))
        valid = np.isfinite(values)
        # Fixed range makes the same physical value use the same color on every tile.
        stops = np.array([100., 200., 300., 450., 600.])
        colors = np.array([[54, 33, 112], [47, 103, 142], [35, 159, 137], [159, 211, 92], [250, 226, 72]])
        safe = np.where(valid, values, stops[0])
        for channel in range(3):
            rgba[..., channel] = np.interp(safe, stops, colors[:, channel]).astype(np.uint8)
        rgba[..., 3] = np.where(valid, 200, 0).astype(np.uint8)
    elif layer == "shade":
        halo = max(2, math.ceil(500 / resolution) + 1)
        elevation = _tile_terrain(manifest, Path(terrain_dir), zoom, tile_x, tile_y, halo)
        center_x = left + (tile_x * size + size / 2) * resolution
        center_y = top - (tile_y * size + size / 2) * resolution
        _, latitude = transform_coordinates(dem_crs, dem_crs.geodetic_crs, [center_x], [center_y])
        azimuth, altitude, sun_weight, _ = _solar_inputs({"sol": 0, "localSolarTime": clock, "solReference": "mission"}, latitude[0])
        if sun_weight and np.isfinite(elevation).any():
            xres = resolution * max(math.cos(math.radians(latitude[0])), 1e-6)
            gr, gc = np.gradient(elevation, resolution, xres)
            with SHADOW_KERNEL_LOCK:
                light = compute_illumination(elevation, gr, gc, azimuth, altitude, xres, resolution, 500)
            valid = np.isfinite(elevation) & np.isfinite(gr) & np.isfinite(gc)
            shade = np.clip(1.0 - light, 0, 1)
        elif sun_weight:
            valid = np.zeros(elevation.shape, dtype=bool)
            shade = np.zeros(elevation.shape, dtype=np.float32)
        else:
            valid = np.isfinite(elevation)
            shade = np.ones(elevation.shape, dtype=np.float32)
        crop = np.s_[halo:halo + size, halo:halo + size]
        rgba[..., :3] = (22, 28, 51)
        # Boost partial shade for readability without darkening fully lit cells.
        rgba[..., 3] = np.where(valid[crop], np.sqrt(shade[crop]) * 225, 0).astype(np.uint8)
    else:
        raise ApiError(404, "tile_not_found", "Unknown map layer.")
    in_region = ((tile_y * size + np.arange(size))[:, None] < rows) & ((tile_x * size + np.arange(size))[None, :] < cols)
    rgba[~in_region] = 0
    output = io.BytesIO()
    Image.fromarray(rgba).save(output, format="PNG", compress_level=6)
    return output.getvalue()


def calculate_route(payload, manifest: dict, terrain_dir: Path, dem_path: Path, thermal_path: Path) -> dict:
    if not isinstance(payload, dict) or payload.get("regionId") != manifest.get("id"):
        raise ApiError(400, "invalid_region", "Select the region supported by this backend.")
    start, destination = _point(payload.get("start"), "start"), _point(payload.get("destination"), "destination")
    rover = payload.get("rover")
    slope = rover.get("maxSlopeDeg") if isinstance(rover, dict) else None
    if isinstance(slope, bool) or not isinstance(slope, (int, float)) or not math.isfinite(slope) or not 0 < slope < 90:
        raise ApiError(400, "invalid_rover", "Maximum slope must be between 0 and 90 degrees.")
    raw_weights = payload.get("weights", {})
    if not isinstance(raw_weights, dict):
        raise ApiError(400, "invalid_weights", "Route weights must be an object.")
    weights = {}
    for key in ("slope", "shade", "thermal"):
        value = raw_weights.get(key, 1.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 2:
            raise ApiError(400, "invalid_weights", f"{key.capitalize()} weight must be between 0 and 2.")
        weights[key] = float(value)
    elevation, transform, resolution, zoom = _terrain_window(manifest, terrain_dir, start, destination)
    with rasterio.open(dem_path) as dem:
        if dem.crs is None or dem.width != manifest["width"] or dem.height != manifest["height"] or not np.allclose(dem.bounds, manifest["extent"], rtol=0, atol=0.01):
            raise ApiError(503, "terrain_mismatch", "The backend elevation raster does not match the map region.")
        dem_crs = dem.crs
        _, latitudes = transform_coordinates(dem_crs, dem_crs.geodetic_crs, [start[0]], [start[1]])
    azimuth, altitude, sun_weight, warnings = _solar_inputs(payload.get("marsDateTime"), latitudes[0])
    sun_weight *= weights["shade"]
    if weights["shade"] == 0:
        warnings = ["Shade was excluded from the route."]
    thermal = None
    layers = []
    if weights["thermal"] > 0:
        thermal = _thermal_window(thermal_path, dem_crs, transform, elevation.shape)
        layers.append(CostLayer("Thermal inertia", thermal, weight=3.0 * weights["thermal"], lo=250, hi=300,
                                block_below=200, invert=True, units="TIU"))
    else:
        warnings.append("Thermal inertia was excluded from the route.")
    try:
        with SHADOW_KERNEL_LOCK:
            router = TerrainRouter(elevation, resolution, resolution, max_slope_deg=float(slope),
                                   max_roll_deg=10, rover_radius_m=1, smooth_sigma=1,
                                   slope_weight=10 * weights["slope"], roll_weight=10 * weights["slope"], sun_az_deg=azimuth,
                                   sun_alt_deg=altitude, sun_weight=sun_weight,
                                   shadow_dist_m=500 if sun_weight else 0, layers=layers)
        start_rc = tuple(map(int, rasterio.transform.rowcol(transform, *start)))
        destination_rc = tuple(map(int, rasterio.transform.rowcol(transform, *destination)))
        for label, (row, col) in (("start", start_rc), ("destination", destination_rc)):
            if not (0 <= row < elevation.shape[0] and 0 <= col < elevation.shape[1]) or not np.isfinite(elevation[row, col]):
                raise ApiError(422, "missing_elevation", f"The {label} has no elevation data at the selected route resolution.")
        if router.blocked[start_rc]:
            raise ApiError(422, "blocked_start", "The start is blocked by the current terrain or rover settings.")
        evaluated_target = destination_rc
        if router.blocked[destination_rc]:
            evaluated_target = router.snap(destination_rc)
            warnings.append("The science site is blocked by the rover model; the nearest traversable cell was used as the target.")
        result = router.find_route(start_rc, evaluated_target)
    except ApiError:
        raise
    except (ValueError, RuntimeError) as exc:
        raise ApiError(422, "route_unavailable", str(exc)) from exc

    def coord(row_col):
        row, col = map(int, row_col)
        x, y = rasterio.transform.xy(transform, row, col)
        return [float(x), float(y)]

    path = [coord(cell) for cell in result.path]
    thermal_info = None
    if thermal is not None:
        path_thermal = thermal[result.path[:, 0], result.path[:, 1]]
        valid_thermal = np.isfinite(path_thermal)
        if not valid_thermal.all():
            warnings.append(f"Thermal inertia is missing for {np.count_nonzero(~valid_thermal)} of {len(path_thermal)} route cells; those cells have no thermal penalty.")
        warnings.append("Thermal inertia is approximately 100 m per pixel and is used as a coarse ground-condition estimate.")
        thermal_info = {"coverageFraction": float(valid_thermal.mean()),
                        "meanAlongRoute": float(path_thermal[valid_thermal].mean()) if valid_thermal.any() else None,
                        "units": "TIU"}
    warnings.append("Parking is the model's route endpoint, not a separately validated parking site; Marswalk planning is not yet included.")
    if zoom < manifest["maxZoom"]:
        warnings.append(f"This large route uses a coarser terrain level: approximately {resolution:.0f} m per grid cell.")
    warnings.append("The search is limited to the selected area plus a detour margin.")
    reached = result.status == "Target reached" and evaluated_target == destination_rc
    return {
        "regionId": manifest["id"], "status": "reached" if reached else "partial", "path": path,
        "stop": path[-1], "distanceM": float(result.length_grid), "evaluatedStart": coord(start_rc),
        "evaluatedDestination": coord(evaluated_target), "resolutionM": [resolution, resolution],
        "warnings": warnings, "weights": weights, "thermalInertia": thermal_info,
    }


class RouteServer(ThreadingHTTPServer):
    def __init__(self, address, manifest_path: Path, terrain_dir: Path, dem_path: Path, thermal_path: Path, origins: set[str]):
        super().__init__(address, RouteHandler)
        self.manifest_path, self.terrain_dir, self.dem_path, self.thermal_path = manifest_path, terrain_dir, dem_path, thermal_path
        self.origins = origins
        self.route_lock = threading.BoundedSemaphore(1)
        self.tile_lock = threading.BoundedSemaphore(2)


class RouteHandler(BaseHTTPRequestHandler):
    server: RouteServer

    def _send(self, status: int, data: dict):
        body = json.dumps(data, allow_nan=False).encode("utf-8")
        self.send_response(status)
        origin = self.headers.get("Origin")
        if origin in self.server.origins:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        with suppress(BrokenPipeError, ConnectionResetError):
            self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        origin = self.headers.get("Origin")
        if origin in self.server.origins:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        request = urlsplit(self.path)
        match = re.fullmatch(r"/api/tiles/(thermal|shade)/(\d+)/(\d+)/(\d+)\.png", request.path)
        if match:
            layer, zoom, tile_x, tile_y = match.groups()
            clock = parse_qs(request.query).get("localSolarTime", ["12:00"])[0]
            if not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", clock):
                self._send(400, {"error": {"code": "invalid_mars_time", "message": "Local solar time must be HH:MM."}})
                return
            try:
                with self.server.tile_lock:
                    tile = _map_tile(str(self.server.manifest_path), self.server.manifest_path.stat().st_mtime_ns,
                                     str(self.server.terrain_dir), str(self.server.dem_path), str(self.server.thermal_path),
                                     layer, int(zoom), int(tile_x), int(tile_y), clock if layer == "shade" else "")
                self.send_response(200)
                origin = self.headers.get("Origin")
                if origin in self.server.origins:
                    self.send_header("Access-Control-Allow-Origin", origin)
                    self.send_header("Vary", "Origin")
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(tile)))
                self.send_header("Cache-Control", "public, max-age=300")
                self.end_headers()
                with suppress(BrokenPipeError, ConnectionResetError):
                    self.wfile.write(tile)
            except ApiError as exc:
                self._send(exc.status, {"error": {"code": exc.code, "message": exc.message}})
            except (OSError, rasterio.errors.RasterioError, ValueError) as exc:
                self._send(503, {"error": {"code": "data_unavailable", "message": str(exc)}})
            except Exception:
                import traceback
                traceback.print_exc()
                self._send(500, {"error": {"code": "internal_error", "message": "The map tile could not be generated. See the backend log for details."}})
            return
        if request.path != "/api/health":
            self._send(404, {"error": {"code": "not_found", "message": "Unknown endpoint."}})
            return
        ready = all(path.exists() for path in (self.server.manifest_path, self.server.terrain_dir, self.server.dem_path, self.server.thermal_path))
        self._send(200 if ready else 503, {"status": "ready" if ready else "data_unavailable"})

    def do_POST(self):
        if self.path != "/api/route":
            self._send(404, {"error": {"code": "not_found", "message": "Unknown endpoint."}})
            return
        if not self.server.route_lock.acquire(blocking=False):
            self._send(503, {"error": {"code": "busy", "message": "The route service is calculating another route. Try again shortly."}})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 32768:
                raise ApiError(400, "invalid_request", "Provide a JSON request of at most 32 KiB.")
            try:
                payload = json.loads(self.rfile.read(length))
            except (ValueError, UnicodeDecodeError) as exc:
                raise ApiError(400, "invalid_json", "The request must contain valid JSON.") from exc
            try:
                manifest = json.loads(self.server.manifest_path.read_text())
                result = calculate_route(payload, manifest, self.server.terrain_dir, self.server.dem_path, self.server.thermal_path)
            except (OSError, rasterio.errors.RasterioError) as exc:
                raise ApiError(503, "data_unavailable", f"A required dataset could not be opened: {exc}") from exc
            self._send(200, result)
        except ApiError as exc:
            self._send(exc.status, {"error": {"code": exc.code, "message": exc.message}})
        except Exception:
            import traceback
            traceback.print_exc()
            self._send(500, {"error": {"code": "internal_error", "message": "The route could not be calculated. See the backend log for details."}})
        finally:
            self.server.route_lock.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--terrain-dir", type=Path, default=DEFAULT_TERRAIN)
    parser.add_argument("--dem", type=Path, default=DEFAULT_DEM)
    parser.add_argument("--thermal", type=Path, default=DEFAULT_THERMAL)
    parser.add_argument("--allow-origin", action="append", default=[], help="Additional frontend origin permitted by CORS")
    args = parser.parse_args()
    origins = {"http://localhost:5173", "http://127.0.0.1:5173", *args.allow_origin}
    server = RouteServer((args.host, args.port), args.manifest, args.terrain_dir, args.dem, args.thermal, origins)
    print(f"Mars route API listening at http://{args.host}:{args.port}/api/route", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
