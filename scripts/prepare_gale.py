#!/usr/bin/env python3
"""Prepare a native-resolution, tiled Gale region for the web map.

Run from the repository root with ``backend/.venv/bin/python scripts/prepare_gale.py``.
The data are exploratory: image/DEM registration and the elevation vertical
datum/units must be independently verified before use for decisions.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import re
import resource
import time

import numpy as np
from PIL import Image
import rasterio
from rasterio.enums import MaskFlags, Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEM = ROOT / "backend/data/gale_20m_interpolated_feathered.tif"
DEFAULT_IMAGERY = ROOT / "backend/data/curiosity_ctx_mosaic.tif"
DEFAULT_OUTPUT = ROOT / "frontend/public/data/gale"
TILE_SIZE = 256
IMAGERY_RATIO = 2
IMAGERY_TILE_SIZE = TILE_SIZE * IMAGERY_RATIO
MAX_ZOOM = 6
TRANSPARENT_IMAGERY = np.zeros((IMAGERY_TILE_SIZE, IMAGERY_TILE_SIZE, 4), dtype=np.uint8)
ESTIMATED_RGBA = (73, 144, 216, 200)
MISSING_RGBA = (231, 148, 47, 190)
ELEV_STOPS = [(0.0, (42, 71, 91)), (0.25, (106, 118, 109)),
              (0.5, (184, 135, 100)), (0.75, (224, 186, 135)),
              (1.0, (255, 234, 192))]
SLOPE_STOPS = [(0.0, (34, 157, 140)), (1 / 3, (239, 198, 100)),
               (2 / 3, (230, 117, 67)), (1.0, (195, 59, 54))]


def color_lut(stops: list[tuple[float, tuple[int, int, int]]]) -> np.ndarray:
    positions = [stop[0] for stop in stops]
    colors = np.asarray([stop[1] for stop in stops], dtype=np.float32)
    values = np.linspace(0, 1, 1024)
    return np.stack([np.interp(values, positions, colors[:, i]) for i in range(3)], axis=-1).astype(np.uint8)


ELEV_LUT = color_lut(ELEV_STOPS)
SLOPE_LUT = color_lut(SLOPE_STOPS)


def colored(values: np.ndarray, valid: np.ndarray, lut: np.ndarray, opacity: int = 255) -> np.ndarray:
    indices = (np.clip(np.where(valid, values, 0.0), 0, 1) * (len(lut) - 1)).astype(np.int16)
    rgba = np.empty((*values.shape, 4), dtype=np.uint8)
    rgba[..., :3] = lut[indices]
    rgba[..., 3] = np.where(valid, opacity, 0).astype(np.uint8)
    return rgba


def tile_path(output: Path, layer: str, z: int, x: int, y: int, extension: str) -> Path:
    return output / "tiles" / layer / str(z) / str(x) / f"{y}.{extension}"


def save_image(output: Path, layer: str, z: int, x: int, y: int, rgba: np.ndarray) -> None:
    path = tile_path(output, layer, z, x, y, "png")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba).save(path, compress_level=3)


def save_terrain(output: Path, z: int, x: int, y: int, terrain: np.ndarray) -> None:
    path = tile_path(output, "terrain", z, x, y, "f32")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(terrain, dtype="<f4").tofile(path)


def load_terrain(output: Path, z: int, x: int, y: int) -> np.ndarray:
    path = tile_path(output, "terrain", z, x, y, "f32")
    if not path.is_file():
        return np.full((TILE_SIZE, TILE_SIZE), np.nan, dtype=np.float32)
    return np.fromfile(path, dtype="<f4").reshape(TILE_SIZE, TILE_SIZE)


def load_imagery(output: Path, z: int, x: int, y: int) -> np.ndarray:
    path = tile_path(output, "imagery", z, x, y, "png")
    if not path.is_file():
        return TRANSPARENT_IMAGERY
    with Image.open(path) as image:
        return np.asarray(image.convert("RGBA"))


def load_estimated(output: Path, z: int, x: int, y: int) -> np.ndarray:
    """Read the generated categorical coverage tile, preserving estimated cells."""
    path = tile_path(output, "coverage", z, x, y, "png")
    if not path.is_file():
        return np.zeros((TILE_SIZE, TILE_SIZE), dtype=bool)
    with Image.open(path) as image:
        rgba = np.asarray(image.convert("RGBA"))
        return np.all(rgba == np.asarray(ESTIMATED_RGBA, dtype=np.uint8), axis=2)


def grid_size(width: int, height: int, zoom: int) -> tuple[int, int, int, int]:
    """Return visible rows/columns and tile rows/columns at this zoom."""
    factor = 2 ** (MAX_ZOOM - zoom)
    cols = (width + factor - 1) // factor
    rows = (height + factor - 1) // factor
    return rows, cols, math.ceil(rows / TILE_SIZE), math.ceil(cols / TILE_SIZE)


def downsample_children(output: Path, z: int, x: int, y: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a parent from four child tiles without filling any coverage gaps."""
    n = TILE_SIZE
    m = IMAGERY_TILE_SIZE
    terrain = np.full((n * 2, n * 2), np.nan, dtype=np.float32)
    imagery = np.zeros((m * 2, m * 2, 4), dtype=np.uint8)
    estimated = np.zeros((n * 2, n * 2), dtype=bool)
    for dy in range(2):
        for dx in range(2):
            sy, sx = slice(dy * n, (dy + 1) * n), slice(dx * n, (dx + 1) * n)
            terrain[sy, sx] = load_terrain(output, z + 1, x * 2 + dx, y * 2 + dy)
            estimated[sy, sx] = load_estimated(output, z + 1, x * 2 + dx, y * 2 + dy)
            iy, ix = slice(dy * m, (dy + 1) * m), slice(dx * m, (dx + 1) * m)
            imagery[iy, ix] = load_imagery(output, z + 1, x * 2 + dx, y * 2 + dy)
    cells = terrain.reshape(n, 2, n, 2)
    known = np.isfinite(cells).all(axis=(1, 3))
    # Replacing NaNs only for the calculation does not mark them as known.
    averaged = (np.where(np.isfinite(cells), cells, 0.0).sum(axis=(1, 3), dtype=np.float32) / 4.0)
    averaged[~known] = np.nan
    pixels = imagery.reshape(m, 2, m, 2, 4)
    # Image alpha describes the source footprint, not confirmed usable imagery.
    image_known = (pixels[..., 3] == 255).all(axis=(1, 3))
    rgb = (pixels[..., :3].astype(np.uint16).sum(axis=(1, 3)) // 4).astype(np.uint8)
    rgba = np.zeros((m, m, 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = np.where(image_known, 255, 0).astype(np.uint8)
    # A parent pixel is estimated if any child was estimated, except that any
    # missing child takes priority and is represented by NaN in the terrain.
    estimated_parent = estimated.reshape(n, 2, n, 2).any(axis=(1, 3)) & known
    return averaged, rgba, estimated_parent


def read_native(dem: rasterio.io.DatasetReader, image: WarpedVRT, provenance: rasterio.io.DatasetReader | None,
                x: int, y: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = TILE_SIZE
    left, top = x * n, y * n
    width, height = min(n, dem.width - left), min(n, dem.height - top)
    window = Window(left, top, width, height)
    raw = dem.read(1, window=window, masked=True).astype(np.float32)
    z = (raw * dem.scales[0] + dem.offsets[0]).filled(np.nan).astype(np.float32)
    z[~np.isfinite(z)] = np.nan
    terrain = np.full((n, n), np.nan, dtype=np.float32)
    terrain[:height, :width] = z
    estimated = np.zeros((n, n), dtype=bool)
    if provenance is not None:
        estimated[:height, :width] = provenance.read(1, window=window) == 1
    image_window = Window(left * IMAGERY_RATIO, top * IMAGERY_RATIO,
                          width * IMAGERY_RATIO, height * IMAGERY_RATIO)
    bands = image.read(list(range(1, image.count)), window=image_window)
    alpha = image.read(image.count, window=image_window)
    rgb = np.repeat(bands, 3, axis=0) if bands.shape[0] == 1 else bands
    rgba = np.zeros((IMAGERY_TILE_SIZE, IMAGERY_TILE_SIZE, 4), dtype=np.uint8)
    rgba[:height * IMAGERY_RATIO, :width * IMAGERY_RATIO, :3] = np.moveaxis(rgb, 0, 2)
    rgba[:height * IMAGERY_RATIO, :width * IMAGERY_RATIO, 3] = alpha
    return terrain, rgba, estimated


def resolve_provenance_path(dem_path: Path, dem: rasterio.io.DatasetReader, override: Path | None) -> Path | None:
    """Prefer an explicit mask; otherwise resolve the DEM's provenance tag, if any."""
    if override is not None:
        return override
    tagged = dem.tags().get("PROVENANCE")
    if not tagged:
        return None
    path = Path(tagged)
    return path if path.is_absolute() else dem_path.resolve().parent / path


def validate_provenance_metadata(dem: rasterio.io.DatasetReader, provenance: rasterio.io.DatasetReader) -> None:
    if (provenance.count != 1 or provenance.dtypes[0] != "uint8" or
            provenance.crs != dem.crs or provenance.transform != dem.transform or
            provenance.width != dem.width or provenance.height != dem.height):
        raise ValueError("Provenance must be one-band uint8 with the exact DEM dimensions, CRS and transform.")


def elevation_stats(dem: rasterio.io.DatasetReader, provenance: rasterio.io.DatasetReader | None = None) -> tuple[float, float, int, int]:
    minimum, maximum, valid_count, estimated_count = np.inf, -np.inf, 0, 0
    for row in range(0, dem.height, 512):
        height = min(512, dem.height - row)
        window = Window(0, row, dem.width, height)
        raw = dem.read(1, window=window, masked=True).astype(np.float32)
        data = (raw * dem.scales[0] + dem.offsets[0]).filled(np.nan)
        valid = np.isfinite(data)
        if provenance is not None:
            classes = provenance.read(1, window=window, masked=True)
            if np.ma.getmaskarray(classes).any() or not np.isin(classes.data, (0, 1, 2)).all():
                raise ValueError("Provenance must contain only unmasked classes 0, 1 and 2.")
            if not np.array_equal(classes.data == 2, ~valid):
                raise ValueError("Provenance class 2 must exactly match DEM NoData; classes 0 and 1 must have valid elevation.")
            estimated_count += int(np.count_nonzero(classes.data == 1))
        if valid.any():
            minimum = min(minimum, float(data[valid].min()))
            maximum = max(maximum, float(data[valid].max()))
            valid_count += int(valid.sum())
    if not valid_count:
        raise ValueError("The elevation raster has no valid pixels.")
    return minimum, maximum, valid_count, estimated_count


def write_derived_images(output: Path, z: int, x: int, y: int, terrain: np.ndarray, estimated: np.ndarray,
                         rows: int, cols: int, min_z: float, max_z: float) -> None:
    valid = np.isfinite(terrain)
    normal = (terrain - min_z) / (max_z - min_z) if max_z > min_z else np.zeros_like(terrain)
    save_image(output, "elevation", z, x, y, colored(normal, valid, ELEV_LUT))
    # The outer tile's pixels beyond the region bounds remain transparent.
    in_region = ((y * TILE_SIZE + np.arange(TILE_SIZE))[:, None] < rows) & (
        (x * TILE_SIZE + np.arange(TILE_SIZE))[None, :] < cols)
    coverage = np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8)
    coverage[in_region & valid & estimated] = ESTIMATED_RGBA
    coverage[in_region & ~valid] = MISSING_RGBA
    save_image(output, "coverage", z, x, y, coverage)
    estimated_mask = np.zeros_like(coverage)
    estimated_mask[in_region & valid & estimated] = ESTIMATED_RGBA
    missing_mask = np.zeros_like(coverage)
    missing_mask[in_region & ~valid] = MISSING_RGBA
    save_image(output, "coverage_estimated", z, x, y, estimated_mask)
    save_image(output, "coverage_missing", z, x, y, missing_mask)


def write_slopes(output: Path, dem: rasterio.io.DatasetReader) -> None:
    """Compute derivatives from a 1-pixel halo to avoid seams between tiles."""
    n = TILE_SIZE
    for z in range(MAX_ZOOM, -1, -1):
        _, _, tile_rows, tile_cols = grid_size(dem.width, dem.height, z)
        resolution = float(dem.res[0]) * 2 ** (MAX_ZOOM - z)
        for y in range(tile_rows):
            # Equirectangular x units need local cos(latitude) ground correction.
            lat = (dem.bounds.top - (y * n + np.arange(n) + 0.5) * resolution) / 3396190.0
            xdist = 2 * resolution * np.cos(lat)[:, None]
            for x in range(tile_cols):
                halo = np.full((n + 2, n + 2), np.nan, dtype=np.float32)
                center = load_terrain(output, z, x, y)
                halo[1:-1, 1:-1] = center
                if x > 0:
                    halo[1:-1, 0] = load_terrain(output, z, x - 1, y)[:, -1]
                if x + 1 < tile_cols:
                    halo[1:-1, -1] = load_terrain(output, z, x + 1, y)[:, 0]
                if y > 0:
                    halo[0, 1:-1] = load_terrain(output, z, x, y - 1)[-1, :]
                if y + 1 < tile_rows:
                    halo[-1, 1:-1] = load_terrain(output, z, x, y + 1)[0, :]
                gradient_x = (halo[1:-1, 2:] - halo[1:-1, :-2]) / xdist
                gradient_y = (halo[2:, 1:-1] - halo[:-2, 1:-1]) / (2 * resolution)
                slope = np.degrees(np.arctan(np.hypot(gradient_x, gradient_y)))
                valid = np.isfinite(center) & np.isfinite(slope)
                save_image(output, "slope", z, x, y, colored(slope / 30.0, valid, SLOPE_LUT, 200))
        print(f"z{z}: slope tiles complete ({tile_cols} × {tile_rows})", flush=True)


def prepare(args: argparse.Namespace) -> None:
    start = time.perf_counter()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args.id):
        raise ValueError("Region id must use lowercase letters, digits, underscores, or hyphens.")
    with ExitStack() as stack:
        dem = stack.enter_context(rasterio.open(args.dem))
        image = stack.enter_context(rasterio.open(args.imagery))
        if dem.crs is None or not dem.crs.is_projected or dem.crs.linear_units != "metre":
            raise ValueError("Elevation must have a projected CRS with metre units.")
        t = dem.transform
        if t.b != 0 or t.d != 0 or t.a <= 0 or t.e >= 0 or not math.isclose(dem.res[0], dem.res[1], rel_tol=1e-8):
            raise ValueError("Elevation must have square pixels in a north-up, unrotated projection.")
        if dem.count != 1 or MaskFlags.all_valid in dem.mask_flag_enums[0]:
            raise ValueError("Elevation must be a single band with an explicit NoData mask.")
        if image.crs is None or image.count not in (1, 3) or any(dtype != "uint8" for dtype in image.dtypes):
            raise ValueError("Imagery must be georeferenced, 8-bit, and grayscale or RGB.")
        proj4 = dem.crs.to_proj4().replace("+no_defs=True", "+no_defs")
        if not all(p in proj4.split() for p in ("+proj=eqc", "+lat_ts=0", "+lat_0=0", "+y_0=0", "+R=3396190")):
            raise ValueError("Slope preparation currently requires the spherical Gale Mars equirectangular projection.")
        provenance_path = resolve_provenance_path(args.dem, dem, getattr(args, "provenance", None))
        provenance = stack.enter_context(rasterio.open(provenance_path)) if provenance_path is not None else None
        if provenance is not None:
            validate_provenance_metadata(dem, provenance)
        min_z, max_z, count, estimated_count = elevation_stats(dem, provenance)
        print(f"Elevation: {count:,} / {dem.width * dem.height:,} native cells, including {estimated_count:,} estimated; range {min_z:.3f} to {max_z:.3f}", flush=True)
        print(f"Provenance: {provenance_path if provenance_path is not None else 'not supplied'}", flush=True)
        print("Preparing terrain, imagery, elevation and coverage tiles...", flush=True)
        image_transform = dem.transform * rasterio.Affine.scale(1 / IMAGERY_RATIO, 1 / IMAGERY_RATIO)
        with WarpedVRT(image, crs=dem.crs, transform=image_transform,
                       width=dem.width * IMAGERY_RATIO, height=dem.height * IMAGERY_RATIO,
                       resampling=Resampling.bilinear, add_alpha=True) as aligned:
            for z in range(MAX_ZOOM, -1, -1):
                rows, cols, tile_rows, tile_cols = grid_size(dem.width, dem.height, z)
                for y in range(tile_rows):
                    for x in range(tile_cols):
                        terrain, imagery, estimated = (read_native(dem, aligned, provenance, x, y) if z == MAX_ZOOM
                                                       else downsample_children(args.output, z, x, y))
                        save_terrain(args.output, z, x, y, terrain)
                        save_image(args.output, "imagery", z, x, y, imagery)
                        write_derived_images(args.output, z, x, y, terrain, estimated, rows, cols, min_z, max_z)
                    if z == MAX_ZOOM and y % 5 == 4:
                        print(f"z{z}: finished row {y + 1}/{tile_rows}", flush=True)
                print(f"z{z}: base tiles complete ({tile_cols} × {tile_rows})", flush=True)
        print("Preparing seam-aware slope tiles...", flush=True)
        write_slopes(args.output, dem)
        base = args.asset_base.rstrip("/")
        image_has_mask = all(MaskFlags.all_valid not in flags for flags in image.mask_flag_enums)
        image_note = ("Imagery has a declared NoData mask; source coverage and registration still require verification."
                      if image_has_mask else "Imagery has no declared NoData mask; alpha identifies its footprint only, not confirmed valid imagery.")
        resolutions = [float(dem.res[0]) * 2 ** (MAX_ZOOM - z) for z in range(MAX_ZOOM + 1)]
        manifest = {
            "id": args.id, "name": args.name, "projection": f"MARS:{args.id.upper().replace('-', '_')}", "proj4": proj4,
            "extent": list(map(float, dem.bounds)), "width": dem.width, "height": dem.height,
            "pixelSize": list(map(float, dem.res)), "tileSize": TILE_SIZE, "minZoom": 0,
            "maxZoom": MAX_ZOOM, "resolutions": resolutions, "imageryPixelRatio": IMAGERY_RATIO,
            "layers": {layer: f"{base}/tiles/{layer}/{{z}}/{{x}}/{{y}}.png"
                       for layer in ("imagery", "elevation", "slope", "coverage")},
            "coverageLayers": {category: f"{base}/tiles/coverage_{category}/{{z}}/{{x}}/{{y}}.png"
                               for category in ("estimated", "missing")},
            "terrain": f"{base}/tiles/terrain/{{z}}/{{x}}/{{y}}.f32",
            "terrainEncoding": {"dataType": "float32", "byteOrder": "little", "rowOrder": "north-to-south",
                                "noData": "NaN", "verticalUnits": "Unverified; source documentation required"},
            "elevationRange": [min_z, max_z],
            "slopeLegend": {"minimumDegrees": 0, "maximumDegrees": 30, "maximumIsClamped": True},
            "coverage": {"validElevationPixels": count, "originalElevationPixels": count - estimated_count if provenance is not None else None,
                         "estimatedElevationPixels": estimated_count if provenance is not None else None, "totalPixels": dem.width * dem.height,
                         "imageryNoDataMaskPresent": image_has_mask,
                         "legend": {"source": "transparent" if provenance is not None else "transparent; provenance unavailable",
                                    "estimated": list(ESTIMATED_RGBA), "missing": list(MISSING_RGBA)}},
            "sources": {"elevation": args.dem.name, "imagery": args.imagery.name,
                        "provenance": provenance_path.name if provenance_path is not None else None},
            "notes": [
                f"Native elevation spacing is approximately {dem.res[0]:.2f} projected metres per pixel; zoomed-out levels are coarser averages and cannot resolve smaller hazards.",
                "Low-resolution cells are unknown if any contributing elevation cell is missing; otherwise they are marked estimated if any contributor was estimated. Coverage outside the available DEM extent is unknown.",
                "The imagery and elevation use different Mars reference systems; their feature registration has not been verified.",
                image_note,
                "Elevation vertical units and datum require verification against source documentation.",
                "Slope is an approximate derivative of the current zoom level and does not establish traversability.",
                ("Blue indicates an interpolated or overlap-blended elevation estimate; amber indicates missing elevation. Transparent cells use original valid source elevation."
                 if provenance is not None else
                 "Amber indicates missing elevation. No provenance was supplied, so valid cells cannot be classified as original versus interpolated."),
            ],
        }
        args.output.mkdir(parents=True, exist_ok=True)
        tmp = args.output / "manifest.json.tmp"
        tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        tmp.replace(args.output / "manifest.json")
        elapsed = time.perf_counter() - start
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(f"Finished in {elapsed:.1f} seconds; max RSS (platform units) {peak:,}; output: {args.output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dem", type=Path, default=DEFAULT_DEM)
    parser.add_argument("--provenance", type=Path, help="Optional 0=source, 1=estimated, 2=missing GeoTIFF; defaults to the DEM's PROVENANCE tag when present")
    parser.add_argument("--imagery", type=Path, default=DEFAULT_IMAGERY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--asset-base", default="/data/gale")
    parser.add_argument("--id", default="gale", help="Lowercase identifier for the region")
    parser.add_argument("--name", default="Gale region", help="Display name for the region")
    args = parser.parse_args()
    try:
        prepare(args)
    except (ValueError, OSError, rasterio.errors.RasterioError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
