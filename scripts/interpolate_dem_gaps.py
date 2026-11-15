#!/usr/bin/env python3
"""Create an estimated DEM and provenance mask for narrow, locally bracketed gaps.

This never changes the source DEM. A gap cell is eligible only when original,
finite source elevations lie on *both* sides along one of four axes and
the intervening run is at most --max-gap-width pixels. Raster-edge cells and
unbounded/broad missing regions remain missing. Eligible 8-connected regions
are solved by a distance-weighted eight-neighbour discrete harmonic equation;
unknown neighbours outside the eligible region provide no invented boundary.

These values are interpolated estimates, not measurements or proof of safe
terrain. This does not correct offsets between already-valid overlapping DTMs.
The companion provenance raster distinguishes observed, estimated and missing
cells, and should accompany any use of the derived DEM.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg


@dataclass(frozen=True)
class InterpolationStats:
    candidate_cells: int
    estimated_cells: int
    missing_cells: int
    estimated_regions: int
    skipped_regions: int


def bracketed_candidates(valid: np.ndarray, max_gap_width: int) -> np.ndarray:
    """Return missing cells bracketed by original source values within the width limit.

    Width is the number of pixels between opposing valid source pixels along a
    horizontal, vertical or diagonal grid line. Boundary cells are never filled.
    Bracketing is tested solely against the original valid-data mask, so a filled
    estimate cannot make a further unsupported cell eligible.
    """
    if valid.ndim != 2:
        raise ValueError("The valid-data mask must be two-dimensional.")
    if not 1 <= max_gap_width <= 8:
        raise ValueError("max_gap_width must be between 1 and 8 pixels.")
    height, width = valid.shape
    missing = ~valid
    candidates = np.zeros(valid.shape, dtype=bool)
    for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
        for gap_width in range(1, max_gap_width + 1):
            for index in range(1, gap_width + 1):
                before_r, before_c = -dr * index, -dc * index
                after_r, after_c = dr * (gap_width - index + 1), dc * (gap_width - index + 1)
                r0 = max(0, -before_r, -after_r)
                r1 = min(height, height - before_r, height - after_r)
                c0 = max(0, -before_c, -after_c)
                c1 = min(width, width - before_c, width - after_c)
                if r0 >= r1 or c0 >= c1:
                    continue
                candidates[r0:r1, c0:c1] |= (
                    missing[r0:r1, c0:c1]
                    & valid[r0 + before_r:r1 + before_r, c0 + before_c:c1 + before_c]
                    & valid[r0 + after_r:r1 + after_r, c0 + after_c:c1 + after_c]
                )
    if height and width:
        candidates[0, :] = candidates[-1, :] = False
        candidates[:, 0] = candidates[:, -1] = False
    return candidates


def _solve_region(
    values: np.ndarray,
    original_valid: np.ndarray,
    labels: np.ndarray,
    label: int,
    bounds: tuple[slice, slice],
    x_spacing: float,
    y_spacing: float,
) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray] | None:
    """Solve one bounded component; return absolute pixel positions and estimates."""
    rs, cs = bounds
    row_slice = slice(max(0, rs.start - 1), min(values.shape[0], rs.stop + 1))
    col_slice = slice(max(0, cs.start - 1), min(values.shape[1], cs.stop + 1))
    local_labels = labels[row_slice, col_slice]
    local_valid = original_valid[row_slice, col_slice]
    local_values = values[row_slice, col_slice]
    rr, cc = np.nonzero(local_labels == label)
    n = len(rr)
    ids = np.full(local_labels.shape, -1, dtype=np.int32)
    ids[rr, cc] = np.arange(n, dtype=np.int32)
    diagonal = np.zeros(n, dtype=np.float64)
    rhs = np.zeros(n, dtype=np.float64)
    row_indices: list[np.ndarray] = []
    col_indices: list[np.ndarray] = []
    coefficients: list[np.ndarray] = []
    observed_min, observed_max = np.inf, -np.inf
    for dr, dc in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
        nr, nc = rr + dr, cc + dc
        weight = 1.0 / ((dc * x_spacing) ** 2 + (dr * y_spacing) ** 2)
        adjacent_ids = ids[nr, nc]
        unknown = adjacent_ids >= 0
        observed = local_valid[nr, nc]
        usable = unknown | observed
        diagonal[usable] += weight
        if np.any(unknown):
            row_indices.append(np.flatnonzero(unknown))
            col_indices.append(adjacent_ids[unknown])
            coefficients.append(np.full(int(unknown.sum()), -weight, dtype=np.float64))
        if np.any(observed):
            neighbours = local_values[nr[observed], nc[observed]].astype(np.float64)
            rhs[observed] += weight * neighbours
            observed_min = min(observed_min, float(neighbours.min()))
            observed_max = max(observed_max, float(neighbours.max()))
    if not np.isfinite(observed_min) or np.any(diagonal <= 0):
        return None
    row_indices.append(np.arange(n))
    col_indices.append(np.arange(n))
    coefficients.append(diagonal)
    matrix = coo_matrix(
        (np.concatenate(coefficients), (np.concatenate(row_indices), np.concatenate(col_indices))),
        shape=(n, n),
    ).tocsr()
    initial = np.full(n, (observed_min + observed_max) / 2, dtype=np.float64)
    estimates, info = cg(matrix, rhs, x0=initial, rtol=1e-9, atol=1e-10, maxiter=max(200, min(5000, n * 2)))
    # The weighted harmonic solution must stay within its valid boundary range.
    tolerance = max(1e-5, (observed_max - observed_min) * 1e-6)
    if info != 0 or not np.isfinite(estimates).all() or estimates.min() < observed_min - tolerance or estimates.max() > observed_max + tolerance:
        return None
    return (rr + row_slice.start, cc + col_slice.start), estimates


def interpolate_gaps(
    elevation: np.ndarray,
    *,
    x_spacing: float,
    y_spacing: float,
    max_gap_width: int = 2,
    max_component_cells: int = 4096,
    copy: bool = True,
) -> tuple[np.ndarray, np.ndarray, InterpolationStats]:
    """Return derived elevation, provenance (0 source/1 estimated/2 missing), stats."""
    if elevation.ndim != 2 or not np.issubdtype(elevation.dtype, np.floating):
        raise ValueError("Elevation must be a 2D floating-point array with NaN for missing values.")
    if not np.isfinite([x_spacing, y_spacing]).all() or x_spacing <= 0 or y_spacing <= 0:
        raise ValueError("Pixel spacing must be finite and positive.")
    if not 1 <= max_component_cells <= 100_000:
        raise ValueError("max_component_cells must be between 1 and 100000.")
    result = np.array(elevation, copy=copy)
    original_valid = np.isfinite(result)
    candidates = bracketed_candidates(original_valid, max_gap_width)
    labels, region_count = ndimage.label(candidates, structure=np.ones((3, 3), dtype=np.uint8))
    counts = np.bincount(labels.ravel())
    slices = ndimage.find_objects(labels)
    provenance = np.full(result.shape, 2, dtype=np.uint8)
    provenance[original_valid] = 0
    estimated_regions = skipped_regions = 0
    for region_id in range(1, region_count + 1):
        if counts[region_id] > max_component_cells:
            skipped_regions += 1
            continue
        bounds = slices[region_id - 1]
        if bounds is None:
            continue
        solution = _solve_region(result, original_valid, labels, region_id, bounds, x_spacing, y_spacing)
        if solution is None:
            skipped_regions += 1
            continue
        indexes, estimates = solution
        result[indexes] = estimates
        provenance[indexes] = 1
        estimated_regions += 1
    stats = InterpolationStats(
        candidate_cells=int(candidates.sum()),
        estimated_cells=int(np.count_nonzero(provenance == 1)),
        missing_cells=int(np.count_nonzero(provenance == 2)),
        estimated_regions=estimated_regions,
        skipped_regions=skipped_regions,
    )
    return result, provenance, stats


def smooth_valid_elevation(elevation: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing that leaves missing cells missing (even at array edges)."""
    if not np.isfinite(sigma) or not 0 < sigma <= 10:
        raise ValueError("Smoothing sigma must be greater than 0 and at most 10 pixels.")
    valid = np.isfinite(elevation)
    numerator = np.where(valid, elevation, 0).astype(elevation.dtype, copy=False)
    weights = valid.astype(elevation.dtype)
    ndimage.gaussian_filter(numerator, sigma=sigma, mode="constant", cval=0, output=numerator)
    ndimage.gaussian_filter(weights, sigma=sigma, mode="constant", cval=0, output=weights)
    np.divide(numerator, weights, out=numerator, where=valid & (weights > 0))
    numerator[~valid] = np.nan
    return numerator


def process(input_path: Path, output_path: Path, provenance_path: Path, *, max_gap_width: int, max_component_cells: int,
            smooth_output: Path | None = None, smooth_sigma: float = 0) -> InterpolationStats:
    if (smooth_output is None) != (smooth_sigma == 0):
        raise ValueError("Provide both --smooth-output and a positive --smooth-sigma, or neither.")
    if smooth_sigma and (not np.isfinite(smooth_sigma) or not 0 < smooth_sigma <= 10):
        raise ValueError("Smoothing sigma must be greater than 0 and at most 10 pixels.")
    outputs = [output_path, provenance_path] + ([smooth_output] if smooth_output else [])
    paths = [input_path.resolve()] + [path.resolve() for path in outputs]
    if len(set(paths)) != len(paths):
        raise ValueError("Input and all output paths must be distinct.")
    if any(path.exists() for path in outputs):
        raise FileExistsError("Output already exists; choose new output paths to preserve existing data.")
    with rasterio.open(input_path) as src:
        if src.count != 1 or not np.issubdtype(np.dtype(src.dtypes[0]), np.floating):
            raise ValueError("Expected a one-band floating-point elevation raster.")
        if src.crs is None or not src.crs.is_projected or src.crs.linear_units != "metre":
            raise ValueError("Expected a projected raster with metre coordinates.")
        transform = src.transform
        if transform.b != 0 or transform.d != 0 or transform.a <= 0 or transform.e >= 0:
            raise ValueError("Expected north-up, unrotated source pixels.")
        masked = src.read(1, masked=True)
        elevation = masked.filled(np.nan)
        del masked
        elevation[~np.isfinite(elevation)] = np.nan
        derived, provenance, stats = interpolate_gaps(
            elevation, x_spacing=src.res[0], y_spacing=src.res[1],
            max_gap_width=max_gap_width, max_component_cells=max_component_cells, copy=False,
        )
        profile = src.profile.copy()
        nodata = src.nodata if src.nodata is not None else float("nan")
        profile.update(nodata=nodata, compress="deflate")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        provenance_path.parent.mkdir(parents=True, exist_ok=True)
        if smooth_output:
            smooth_output.parent.mkdir(parents=True, exist_ok=True)
        # Keep the in-memory representation as NaN for the smoothing pass.
        write_derived = np.where(np.isfinite(derived), derived, nodata) if np.isfinite(nodata) else derived
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(write_derived, 1)
            dst.scales = src.scales
            dst.offsets = src.offsets
            if src.units[0]:
                dst.set_band_unit(1, src.units[0])
            dst.update_tags(
                SOURCE=input_path.name,
                INTERPOLATION="8-neighbor inverse-square weighted harmonic; locally bracketed gaps only",
                MAX_GAP_WIDTH_PIXELS=str(max_gap_width),
                MAX_COMPONENT_CELLS=str(max_component_cells),
                ESTIMATED_CELLS=str(stats.estimated_cells),
                PROVENANCE=provenance_path.name,
                WARNING="Interpolated values are estimates, not confirmed safe terrain; consult provenance raster.",
            )
        if smooth_output:
            smoothed = smooth_valid_elevation(derived, smooth_sigma)
            if np.isfinite(nodata):
                smoothed[~np.isfinite(smoothed)] = nodata
            with rasterio.open(smooth_output, "w", **profile) as dst:
                dst.write(smoothed, 1)
                dst.scales = src.scales
                dst.offsets = src.offsets
                if src.units[0]:
                    dst.set_band_unit(1, src.units[0])
                dst.update_tags(
                    SOURCE=output_path.name,
                    SMOOTHING="NaN-aware normalized Gaussian; original missing cells remain missing",
                    SIGMA_PIXELS=str(smooth_sigma),
                    PROVENANCE=provenance_path.name,
                    WARNING="All valid heights may be smoothed; provenance records original source/estimated origin, not unchanged values.",
                )
        profile.update(dtype="uint8", nodata=None)
        with rasterio.open(provenance_path, "w", **profile) as dst:
            dst.write(provenance, 1)
            dst.set_band_description(1, "Elevation provenance: 0 original valid source, 1 interpolated estimate, 2 missing")
            dst.update_tags(
                SOURCE=input_path.name,
                LEGEND="0=original valid source; 1=interpolated estimate; 2=still missing",
                WARNING="Class 1 is inferred and must not be treated as measured or verified safe terrain.",
            )
        return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="Original projected one-band elevation GeoTIFF")
    parser.add_argument("--output", type=Path, required=True, help="New derived elevation GeoTIFF; never overwrites")
    parser.add_argument("--provenance", type=Path, required=True, help="New 0=source, 1=estimated, 2=missing GeoTIFF")
    parser.add_argument("--max-gap-width", type=int, default=2, help="Maximum number of missing pixels between valid source cells (default: 2; max: 8)")
    parser.add_argument("--max-component-cells", type=int, default=4096, help="Maximum eligible connected component size (default: 4096; max: 100000)")
    parser.add_argument("--smooth-output", type=Path, help="Optional separate smoothed GeoTIFF (requires --smooth-sigma)")
    parser.add_argument("--smooth-sigma", type=float, default=0, help="Optional Gaussian standard deviation in pixels (0 disables smoothing; max: 10)")
    args = parser.parse_args()
    try:
        stats = process(args.input, args.output, args.provenance, max_gap_width=args.max_gap_width, max_component_cells=args.max_component_cells,
                        smooth_output=args.smooth_output, smooth_sigma=args.smooth_sigma)
    except (ValueError, OSError, rasterio.errors.RasterioError) as exc:
        parser.error(str(exc))
    print(f"Candidates: {stats.candidate_cells:,}; interpolated: {stats.estimated_cells:,} cells in {stats.estimated_regions:,} regions")
    print(f"Still missing: {stats.missing_cells:,} cells; skipped regions: {stats.skipped_regions:,}")
    print("Interpolated values are estimates, not verified terrain. Keep the provenance raster with the derived DEM.")
    if args.smooth_output:
        print(f"The optional smoothed output uses sigma {args.smooth_sigma:g} pixels and blurs real terrain as well as seams.")


if __name__ == "__main__":
    main()
