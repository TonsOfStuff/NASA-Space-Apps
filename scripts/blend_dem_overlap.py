#!/usr/bin/env python3
"""Feather a known DTM overlap into a separate DEM and provenance raster.

The earlier tile is the one selected by the original first-valid merge. Near
its edge, blend towards the later tile where both contain elevation data. The
original DEM, its existing interpolations, and unsupported cells stay intact.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from scipy.ndimage import distance_transform_edt


def blend_block(
    base: np.ndarray,
    provenance: np.ndarray,
    earlier_valid: np.ndarray,
    later: np.ndarray,
    excluded: np.ndarray,
    feather_pixels: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend only measured cells within a limited, mutually supported overlap."""
    if feather_pixels < 1:
        raise ValueError("Feather width must be at least one pixel.")
    if not (base.shape == provenance.shape == earlier_valid.shape == later.shape == excluded.shape):
        raise ValueError("All block inputs must have the same shape.")
    later_valid = np.isfinite(later)
    # A zero border makes distance to data edges well defined. The caller
    # supplies a halo at least as wide as the feather around each output block.
    early_distance = distance_transform_edt(np.pad(earlier_valid, 1))[1:-1, 1:-1]
    later_distance = distance_transform_edt(np.pad(later_valid, 1))[1:-1, 1:-1]
    weight = np.clip(1.0 - early_distance / feather_pixels, 0, 1)
    weight *= np.minimum(later_distance / feather_pixels, 1.0)
    eligible = (
        np.isfinite(base) & (provenance == 0) & earlier_valid & later_valid & ~excluded & (weight > 0)
    )
    result = base.copy()
    result[eligible] = base[eligible] + weight[eligible] * (later[eligible] - base[eligible])
    changed = eligible & (result != base)
    result_provenance = provenance.copy()
    result_provenance[changed] = 1
    return result, result_provenance


def process(args: argparse.Namespace) -> int:
    targets = [args.output.resolve(), args.output_provenance.resolve()]
    sources = [args.input.resolve(), args.provenance.resolve(), args.earlier.resolve(), args.later.resolve()]
    sources.extend(path.resolve() for path in args.exclude)
    if len(set(targets + sources)) != len(targets) + len(set(sources)):
        raise ValueError("Output paths must be distinct from each other and the inputs.")
    if any(path.exists() for path in targets):
        raise FileExistsError("An output already exists; choose fresh output paths.")
    if not 1 <= args.feather_pixels <= 1024 or args.block_rows < 1:
        raise ValueError("Feather width must be 1–1024 pixels and block rows must be positive.")

    with ExitStack() as stack:
        dem = stack.enter_context(rasterio.open(args.input))
        origin = stack.enter_context(rasterio.open(args.provenance))
        if dem.count != 1 or dem.crs is None or dem.scales[0] != 1 or dem.offsets[0] != 0 or dem.crs != origin.crs or dem.transform != origin.transform or (dem.width, dem.height) != (origin.width, origin.height):
            raise ValueError("The DEM and provenance must be aligned single-band rasters.")

        def aligned(path: Path):
            source = stack.enter_context(rasterio.open(path))
            if source.crs != dem.crs or source.count != 1 or source.scales[0] != 1 or source.offsets[0] != 0:
                raise ValueError(f"Source tile requires compatible CRS, one band and unscaled heights: {path}")
            return stack.enter_context(WarpedVRT(
                source, crs=dem.crs, transform=dem.transform, width=dem.width,
                height=dem.height, resampling=Resampling.nearest,
            ))

        early = aligned(args.earlier)
        late = aligned(args.later)
        excluded_sources = [aligned(path) for path in args.exclude]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output_provenance.parent.mkdir(parents=True, exist_ok=True)
        profile = dem.profile.copy()
        profile.update(dtype="float32", nodata=float("nan"), compress="deflate")
        out = stack.enter_context(rasterio.open(args.output, "w", **profile))
        profile.update(dtype="uint8", nodata=None)
        out_provenance = stack.enter_context(rasterio.open(args.output_provenance, "w", **profile))

        changed_total = 0
        for row in range(0, dem.height, args.block_rows):
            rows = min(args.block_rows, dem.height - row)
            top = max(0, row - args.feather_pixels)
            bottom = min(dem.height, row + rows + args.feather_pixels)
            halo = Window(0, top, dem.width, bottom - top)
            core = slice(row - top, row - top + rows)
            original = dem.read(1, window=halo, masked=True).astype(np.float32).filled(np.nan)
            provenance = origin.read(1, window=halo)
            if np.any((provenance > 2) | ((provenance == 2) != ~np.isfinite(original))):
                raise ValueError("DEM and provenance disagree on missing or estimated cells.")
            earlier = early.read(1, window=halo, masked=True)
            later = late.read(1, window=halo, masked=True)
            earlier_valid = ~np.ma.getmaskarray(earlier) & np.isfinite(earlier.filled(np.nan))
            later_values = later.astype(np.float32).filled(np.nan)
            excluded = np.zeros(original.shape, dtype=bool)
            for source in excluded_sources:
                sample = source.read(1, window=halo, masked=True)
                excluded |= ~np.ma.getmaskarray(sample) & np.isfinite(sample.filled(np.nan))
            values, quality = blend_block(original, provenance, earlier_valid, later_values, excluded, args.feather_pixels)
            write_window = Window(0, row, dem.width, rows)
            out.write(values[core], 1, window=write_window)
            out_provenance.write(quality[core], 1, window=write_window)
            changed_total += int(np.count_nonzero((quality[core] == 1) & (provenance[core] == 0)))
            print(f"Rows {row + rows:,}/{dem.height:,}; blended cells {changed_total:,}", flush=True)

        out.update_tags(
            SOURCE=args.input.name,
            PROVENANCE=args.output_provenance.name,
            BLEND=f"Overlap feather between {args.earlier.name} and {args.later.name}",
            FEATHER_PIXELS=str(args.feather_pixels),
            BLENDED_CELLS=str(changed_total),
            WARNING="Blended and interpolated heights are estimates; consult provenance.",
        )
        out_provenance.set_band_description(1, "Elevation provenance: 0 source, 1 interpolated or overlap-blended, 2 missing")
        out_provenance.update_tags(
            SOURCE=args.input.name,
            LEGEND="0=original valid source; 1=interpolated or overlap-blended estimate; 2=missing",
        )
        return changed_total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--earlier", type=Path, required=True)
    parser.add_argument("--later", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", default=[], help="Earlier-priority source tile; may be repeated")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-provenance", type=Path, required=True)
    parser.add_argument("--feather-pixels", type=int, default=128)
    parser.add_argument("--block-rows", type=int, default=256)
    args = parser.parse_args()
    try:
        changed = process(args)
    except (ValueError, OSError, rasterio.errors.RasterioError) as exc:
        parser.error(str(exc))
    print(f"Blended {changed:,} cells. Source and unsupported NoData cells were preserved.")


if __name__ == "__main__":
    main()
