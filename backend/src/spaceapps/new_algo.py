"""Visualize a Gale Crater elevation patch and a slope-limited, sun-aware rover route.

Controls (interactive mode):
    left click                  set start
    right click / shift+click   set target
    Shade checkbox / key i      toggle the shadow layer
    Blocked checkbox / key b    toggle the red blocked zones (NoData, too steep, safety margin, too-soft ground)
    Thermal inertia checkbox / key 1  toggle the thermal inertia layer (when --thermal-inertia is given)
"""

import argparse
import heapq
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.cm import ScalarMappable
from matplotlib.widgets import CheckButtons
from matplotlib.colors import ListedColormap, LightSource, Normalize
from numba import njit, prange
from rasterio.crs import CRS
from rasterio.errors import WindowError
from rasterio.transform import array_bounds, rowcol, xy
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window, from_bounds
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates

NBRS = np.array([(-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1)], dtype=np.int64)


# --------------------------------------------------------------------------- #
# Numba kernels: implicit grid (no edge lists), A*, line-of-sight, path smoothing
# --------------------------------------------------------------------------- #

@njit(cache=True, parallel=True)
def _cast_shadows(elev, xres, yres, az, alt, max_dist):
    """1.0 where the sun is visible, 0.0 where terrain blocks it (ray march toward the sun)."""
    rows, cols = elev.shape
    lit = np.ones((rows, cols), np.float32)
    sx = math.sin(az)   # east component of the direction toward the sun
    sy = math.cos(az)   # north component
    step = min(xres, yres)
    tan_alt = math.tan(alt)
    zmax = np.nanmax(elev)
    nsteps = int(max_dist / step)
    for r in prange(rows):
        for c in range(cols):
            z0 = elev[r, c]
            if np.isnan(z0):
                continue
            for k in range(1, nsteps + 1):
                t = k * step
                z_ray = z0 + t * tan_alt
                if z_ray > zmax:
                    break  # ray is above all terrain: lit
                rr = r + int(math.floor(-sy * t / yres + 0.5))
                cc = c + int(math.floor(sx * t / xres + 0.5))
                if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                    break  # left the patch: assume lit
                if elev[rr, cc] > z_ray:
                    lit[r, c] = 0.0
                    break
    return lit

@njit(cache=True)
def _edge_cost(elev, gr, gc, blocked, r, c, r2, c2,
               xres, yres, max_grad, max_roll, w_slope, w_roll, penalty):
    """Cost of stepping (r, c) -> (r2, c2), or -1.0 if the step is not allowed."""
    if blocked[r2, c2]:
        return -1.0
    dr = r2 - r
    dc = c2 - c
    if dr != 0 and dc != 0:
        # Do not cut diagonally across a blocked corner.
        if blocked[r, c2] or blocked[r2, c]:
            return -1.0
    dy = dr * yres
    dx = dc * xres
    d = math.sqrt(dx * dx + dy * dy)
    grad = abs(np.float64(elev[r2, c2]) - np.float64(elev[r, c])) / d
    if not grad <= max_grad:  # also rejects NaN
        return -1.0
    # Side (roll) slope: terrain gradient component perpendicular to the heading.
    mr = 0.5 * (np.float64(gr[r, c]) + np.float64(gr[r2, c2]))
    mc = 0.5 * (np.float64(gc[r, c]) + np.float64(gc[r2, c2]))
    cross = abs(-mr * dx / d + mc * dy / d)
    if not cross <= max_roll:  # also rejects NaN
        return -1.0
    # Cost is always >= d, so the Euclidean heuristic stays admissible.
    # `penalty` (shade, thermal inertia, ...) is >= 0, so cost stays >= d and the A* heuristic stays admissible.
    return d * (1.0 + w_slope * (grad / max_grad) ** 2 + w_roll * (cross / max_roll) ** 2
                + np.float64(penalty[r2, c2]))


@njit(cache=True)
def _astar(elev, gr, gc, blocked, sr, sc, tr, tc,
           xres, yres, max_grad, max_roll, w_slope, w_roll, penalty):
    rows, cols = elev.shape
    g = np.full((rows, cols), np.inf)
    closed = np.zeros((rows, cols), np.bool_)
    parent = np.full(rows * cols, -1, np.int32)
    g[sr, sc] = 0.0
    heap = [(0.0, np.int64(sr * cols + sc))]
    found = False
    expanded = 0
    while len(heap) > 0:
        _, node = heapq.heappop(heap)
        r = node // cols
        c = node % cols
        if closed[r, c]:
            continue
        closed[r, c] = True
        expanded += 1
        if r == tr and c == tc:
            found = True
            break
        for k in range(8):
            r2 = r + NBRS[k, 0]
            c2 = c + NBRS[k, 1]
            if r2 < 0 or r2 >= rows or c2 < 0 or c2 >= cols or closed[r2, c2]:
                continue
            cost = _edge_cost(elev, gr, gc, blocked, r, c, r2, c2,
                              xres, yres, max_grad, max_roll, w_slope, w_roll, penalty)
            if cost < 0.0:
                continue
            ng = g[r, c] + cost
            if ng < g[r2, c2]:
                g[r2, c2] = ng
                parent[r2 * cols + c2] = np.int32(node)
                h = math.sqrt(((r2 - tr) * yres) ** 2 + ((c2 - tc) * xres) ** 2)
                heapq.heappush(heap, (ng + h, np.int64(r2 * cols + c2)))
    return found, closed, parent, expanded


@njit(cache=True)
def _los(elev, gr, gc, blocked, r0, c0, r1, c1,
         xres, yres, max_grad, max_roll, w_slope, w_roll, penalty):
    """Total cost of the Bresenham line between two cells, or -1.0 if any step is not allowed."""
    r, c = r0, c0
    sr = 1 if r1 > r0 else -1
    sc = 1 if c1 > c0 else -1
    dcx = abs(c1 - c0)
    dry = -abs(r1 - r0)
    err = dcx + dry
    total = 0.0
    while not (r == r1 and c == c1):
        e2 = 2 * err
        nr, nc = r, c
        if e2 >= dry:
            err += dry
            nc += sc
        if e2 <= dcx:
            err += dcx
            nr += sr
        cost = _edge_cost(elev, gr, gc, blocked, r, c, nr, nc,
                          xres, yres, max_grad, max_roll, w_slope, w_roll, penalty)
        if cost < 0.0:
            return -1.0
        total += cost
        r, c = nr, nc
    return total


@njit(cache=True)
def _string_pull(path, elev, gr, gc, blocked,
                 xres, yres, max_grad, max_roll, w_slope, w_roll, penalty):
    """Greedy any-angle smoothing. A shortcut is only taken if it is allowed AND no more
    costly than the grid-path segment it replaces, so sun/slope preferences are kept."""
    n = path.shape[0]
    cum = np.zeros(n)
    for k in range(1, n):
        cum[k] = cum[k - 1] + _edge_cost(elev, gr, gc, blocked, path[k - 1, 0], path[k - 1, 1],
                                         path[k, 0], path[k, 1], xres, yres, max_grad, max_roll,
                                         w_slope, w_roll, penalty)
    out = np.empty((n, 2), np.int64)
    out[0, 0] = path[0, 0]
    out[0, 1] = path[0, 1]
    m = 1
    i = 0
    while i < n - 1:
        j = n - 1
        while j > i + 1:
            line = _los(elev, gr, gc, blocked, path[i, 0], path[i, 1], path[j, 0], path[j, 1],
                        xres, yres, max_grad, max_roll, w_slope, w_roll, penalty)
            if line >= 0.0 and line <= 1.02 * (cum[j] - cum[i]):
                break
            j -= 1
        out[m, 0] = path[j, 0]
        out[m, 1] = path[j, 1]
        m += 1
        i = j
    return out[:m]


# --------------------------------------------------------------------------- #
# Terrain preparation and routing
# --------------------------------------------------------------------------- #
def smooth_nan_aware(dem, sigma):
    """Gaussian smoothing that ignores NoData (normalized convolution)."""
    if sigma <= 0:
        return dem.copy()
    valid = np.isfinite(dem)
    num = gaussian_filter(np.where(valid, dem, 0.0).astype(np.float32), sigma)
    den = gaussian_filter(valid.astype(np.float32), sigma)
    out = np.full_like(dem, np.nan)
    ok = valid & (den > 1e-3)
    out[ok] = num[ok] / den[ok]
    return out



def compute_illumination(elev, gr, gc, az_deg, alt_deg, xres, yres, shadow_dist_m=500.0):
    """Illumination in [0, 1]: fraction of flat-ground sunlight, 0 in cast or self shadow.

    az_deg is the sun azimuth clockwise from north (90 = east, 180 = south, 270 = west);
    alt_deg is its elevation above the horizon. gr/gc are dz/d(row m) and dz/d(col m).
    """
    if not 0 < alt_deg <= 90:
        raise ValueError("Sun altitude must be in (0, 90] degrees.")
    az, alt = np.deg2rad(az_deg), np.deg2rad(alt_deg)
    sx, sy, sz = np.cos(alt) * np.sin(az), np.cos(alt) * np.cos(az), np.sin(alt)
    # Rows grow southward, so dz/dnorth = -gr and dz/deast = gc.
    cos_inc = (-gc * sx + gr * sy + sz) / np.sqrt(1.0 + gc ** 2 + gr ** 2)
    illum = np.clip(cos_inc / sz, 0.0, 1.0)
    if shadow_dist_m > 0:
        illum = illum * _cast_shadows(elev, xres, yres, az, alt, float(shadow_dist_m))
    illum = np.where(np.isfinite(illum), illum, 0.0)
    return np.ascontiguousarray(illum, dtype=np.float32)



@dataclass
class CostLayer:
    """A map layer (thermal inertia, ...) already resampled onto the DEM patch grid.

    The cost penalty per cell is weight * clip((value - lo) / (hi - lo), 0, 1), or 1 minus that
    if invert=True (use invert for "high is good" layers). Cells with value > block_above or
    < block_below are impassable. NaN (no coverage) means no penalty and no block.
    """
    name: str
    data: np.ndarray
    weight: float = 1.0
    lo: float = 0.0
    hi: float = 1.0
    block_above: float = None
    block_below: float = None
    invert: bool = False
    units: str = ""
    cmap: str = "viridis"

    def is_bad(self, value):
        """True if a value is on the undesirable side of the scale (used for the parking warning)."""
        return bool(value < self.lo) if self.invert else bool(value > self.hi)

    def penalty(self):
        norm = np.clip((self.data - self.lo) / max(self.hi - self.lo, 1e-12), 0.0, 1.0)
        if self.invert:
            norm = 1.0 - norm
        return np.where(np.isfinite(norm), self.weight * norm, 0.0).astype(np.float32)

    def blocked(self):
        mask = np.zeros(self.data.shape, bool)
        with np.errstate(invalid="ignore"):
            if self.block_above is not None:
                mask |= self.data > self.block_above
            if self.block_below is not None:
                mask |= self.data < self.block_below
        return mask



def describe_overlap_problem(path, src, crs, dem_crs, transform, shape):
    """Human-readable diagnostics for a layer that doesn't line up with the DEM patch."""
    lines = [f"{Path(path).name} does not overlap the DEM patch. Diagnostics:"]
    try:
        patch = array_bounds(shape[0], shape[1], transform)
        lines.append(f"  Patch bounds in DEM CRS (left, bottom, right, top): {tuple(round(v, 1) for v in patch)}")
        geo = dem_crs.geodetic_crs
        lon_lat = transform_bounds(dem_crs, geo, *patch, densify_pts=21)
        lines.append(f"  Patch in lon/lat (west, south, east, north): {tuple(round(v, 4) for v in lon_lat)}")
        lines.append("    (planetographic/planetocentric differences are small here; the patch center is what matters)")
    except Exception as exc:  # diagnostics must never hide the real error
        lines.append(f"  (could not convert patch bounds: {exc})")
    try:
        lines.append(f"  File CRS: {(crs.to_string() if crs else 'none')[:110]}")
        lines.append(f"  File bounds in its own CRS (left, bottom, right, top): {tuple(round(v, 4) for v in src.bounds)}")
        if crs and crs.is_geographic:
            lines.append("  File is geographic (degrees): longitudes above 180 mean a 0-360 file; the patch is in -180..180 if its west value is negative.")
        else:
            pb = transform_bounds(dem_crs, crs, *array_bounds(shape[0], shape[1], transform), densify_pts=21)
            lines.append(f"  Patch bounds converted into the file's CRS: {tuple(round(v, 1) for v in pb)}")
    except Exception as exc:
        lines.append(f"  (could not describe the file: {exc})")
    lines.append("  Fixes: choose a patch inside the file with --center E N (DEM CRS coordinates), "
                 "or pass the right projection with --layer-crs, or use a file that covers this location.")
    return "\n".join(lines)


def load_global_layer(path, dem_crs, transform, shape, src_crs=None, resampling="bilinear", band=1):
    """Resample a (possibly planet-wide, coarser, differently projected) raster onto the DEM patch grid.

    GDAL's warper reads only the part of the source that overlaps the patch, reprojects it, and
    interpolates it to the DEM's pixel grid. Returns (data float32 patch, lo, hi) where lo/hi are the
    2nd/98th percentiles of the WHOLE source raster, so penalties use a planet-wide scale instead of
    being stretched across the small patch.
    """
    with rasterio.open(path) as src:
        crs = CRS.from_user_input(src_crs) if src_crs else src.crs
        if crs is None:
            # Common for planetary products: no CRS tag. Assume plain lon/lat on the DEM's own body.
            crs = dem_crs.geodetic_crs
            print(f"{Path(path).name}: no CRS in file, assuming lon/lat on the DEM's body (override with --layer-crs).")
        dest = np.full(shape, np.nan, dtype=np.float32)
        # ISIS3 (.cub) / PDS products mark bad pixels with huge negative special values that are not always
        # declared as NoData, so read those through memory where we can mask them ourselves.
        if src_crs or src.crs is None or src.nodata is None or src.driver in ("ISIS3", "ISIS2", "PDS", "PDS4", "VICAR"):
            # GDAL ignores a CRS override on an untagged file, so crop the needed window ourselves
            # (patch bounds converted into the source CRS, plus a small pad) and warp from memory.
            bounds = transform_bounds(dem_crs, crs, *array_bounds(shape[0], shape[1], transform), densify_pts=21)
            win = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
            win = Window(win.col_off - 2, win.row_off - 2, win.width + 4, win.height + 4)
            try:
                win = win.intersection(Window(0, 0, src.width, src.height))
            except WindowError:
                raise ValueError(describe_overlap_problem(path, src, crs, dem_crs, transform, shape)) from None
            arr = src.read(band, window=win).astype(np.float32)
            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan
            arr[arr <= -1e30] = np.nan  # ISIS special pixels (NULL, LRS, LIS, HIS, HRS) are all about -3.4e38
            reproject(source=arr, destination=dest, src_transform=src.window_transform(win), src_crs=crs,
                      src_nodata=np.nan, dst_transform=transform, dst_crs=dem_crs, dst_nodata=np.nan,
                      resampling=Resampling[resampling])
        else:
            reproject(source=rasterio.band(src, band), destination=dest, src_crs=crs,
                      dst_transform=transform, dst_crs=dem_crs, src_nodata=src.nodata, dst_nodata=np.nan,
                      resampling=Resampling[resampling])
        if src.scales[band - 1] != 1 or src.offsets[band - 1] != 0:
            dest = dest * src.scales[band - 1] + src.offsets[band - 1]
        # Planet-wide scale from a decimated read.
        h, w = src.height, src.width
        coarse = src.read(band, out_shape=(min(h, 1024), min(w, 2048)), masked=True).astype(np.float64)
        coarse = np.ma.masked_less_equal(coarse, -1e30)  # ISIS special pixels
        coarse = coarse * src.scales[band - 1] + src.offsets[band - 1]
        vals = coarse.compressed() if np.ma.isMaskedArray(coarse) else coarse[np.isfinite(coarse)]
        lo, hi = (float(np.percentile(vals, 2)), float(np.percentile(vals, 98))) if vals.size else (0.0, 1.0)
    covered = np.isfinite(dest).mean()
    if covered == 0:
        with rasterio.open(path) as src:
            msg = describe_overlap_problem(path, src, CRS.from_user_input(src_crs) if src_crs else (src.crs or dem_crs.geodetic_crs),
                                           dem_crs, transform, shape)
        raise ValueError(msg.replace("does not overlap the DEM patch.", "has no valid data over the DEM patch (all NoData or outside the file)."))
    if covered < 0.999:
        print(f"Warning: {Path(path).name} covers only {covered:.0%} of the patch; uncovered cells get no penalty.")
    return dest, lo, hi


@dataclass
class RouteResult:
    path: np.ndarray        # raw A* grid path, (N, 2) of (row, col)
    smooth: np.ndarray      # string-pulled any-angle waypoints
    status: str
    length_grid: float      # 3D length of the raw path (m)
    length_smooth: float    # 3D length of the smoothed path (m)
    expanded: int           # number of cells A* expanded
    explored: np.ndarray    # boolean mask of expanded cells
    sun_fraction: float     # mean illumination along the grid path (1 = full sun)
    layer_stats: dict       # name -> (mean value along path, value at path end, units)


class TerrainRouter:
    """Precompute slope and traversability once, then answer A* queries cheaply."""

    def __init__(self, elevation, x_resolution, y_resolution, max_slope_deg=15.0,
                 max_roll_deg=10.0, rover_radius_m=1.0, smooth_sigma=1.0,
                 slope_weight=1.0, roll_weight=1.0, sun_az_deg=270.0, sun_alt_deg=30.0,
                 sun_weight=2.0, shadow_dist_m=500.0, layers=()):
        if elevation.ndim != 2:
            raise ValueError("Elevation must be a 2D array.")
        if not 0 < max_slope_deg < 90 or not 0 < max_roll_deg < 90:
            raise ValueError("Maximum slope and roll must be between 0 and 90 degrees.")
        if x_resolution <= 0 or y_resolution <= 0:
            raise ValueError("Pixel dimensions must be positive.")
        self.rows, self.cols = elevation.shape
        self.xres, self.yres = float(x_resolution), float(y_resolution)
        self.max_grad = float(np.tan(np.deg2rad(max_slope_deg)))
        self.max_roll = float(np.tan(np.deg2rad(max_roll_deg)))
        self.w_slope, self.w_roll = float(slope_weight), float(roll_weight)
        self.layers = list(layers)
        self.layers_by_name = {l.name: l for l in self.layers}

        # Slope over a rover-sized baseline instead of between adjacent 1 m pixels.
        self.elev = np.ascontiguousarray(smooth_nan_aware(elevation, smooth_sigma), dtype=np.float32)
        gr, gc = np.gradient(self.elev, self.yres, self.xres)
        self.gr = np.ascontiguousarray(gr, dtype=np.float32)
        self.gc = np.ascontiguousarray(gc, dtype=np.float32)
        self.slope_deg = np.degrees(np.arctan(np.hypot(self.gr, self.gc)))
        self.illum = compute_illumination(self.elev, self.gr, self.gc, sun_az_deg, sun_alt_deg,
                                          self.xres, self.yres, shadow_dist_m)

        # Blocked = NoData or too steep, inflated by the rover radius.
        bad = ~np.isfinite(self.elev) | (self.slope_deg > max_slope_deg)
        if rover_radius_m > 0 and bad.any():
            dist = distance_transform_edt(~bad, sampling=(self.yres, self.xres))
            bad = dist <= rover_radius_m
        # Extra layers: hard blocks and soft penalties, added on top of shade.
        penalty = float(sun_weight) * (1.0 - self.illum)
        for layer in self.layers:
            if layer.data.shape != bad.shape:
                raise ValueError(f"Layer '{layer.name}' does not match the DEM patch shape.")
            bad = bad | layer.blocked()
            penalty = penalty + layer.penalty()
        self.penalty = np.ascontiguousarray(np.maximum(penalty, 0.0), dtype=np.float32)
        self.blocked = np.ascontiguousarray(bad)
        self._filled = np.where(np.isfinite(self.elev), self.elev, np.nanmean(self.elev)).astype(np.float32)
        if self.blocked.all():
            raise ValueError("Every cell is blocked; relax the slope limits or rover radius.")

    def snap(self, rc):
        """Nearest traversable cell to (row, col)."""
        row = min(max(int(rc[0]), 0), self.rows - 1)
        col = min(max(int(rc[1]), 0), self.cols - 1)
        if not self.blocked[row, col]:
            return row, col
        _, (ri, ci) = distance_transform_edt(self.blocked, return_indices=True)
        return int(ri[row, col]), int(ci[row, col])

    def _length_3d(self, pts):
        total = 0.0
        for a, b in zip(pts[:-1], pts[1:]):
            n = int(np.abs(b - a).max()) + 1
            t = np.linspace(0.0, 1.0, n + 1)
            rr = a[0] + (b[0] - a[0]) * t
            cc = a[1] + (b[1] - a[1]) * t
            z = map_coordinates(self._filled, [rr, cc], order=1, mode="nearest")
            horiz = np.hypot(np.diff(cc) * self.xres, np.diff(rr) * self.yres)
            total += float(np.hypot(horiz, np.diff(z)).sum())
        return total

    def find_route(self, start, target):
        for name, (row, col) in (("start", start), ("target", target)):
            if not (0 <= row < self.rows and 0 <= col < self.cols):
                raise ValueError(f"The {name} point is outside the elevation patch.")
            if self.blocked[row, col]:
                raise ValueError(f"The {name} point is blocked (NoData, too steep, or inside the safety margin).")
        found, closed, parent, expanded = _astar(
            self.elev, self.gr, self.gc, self.blocked, start[0], start[1], target[0], target[1],
            self.xres, self.yres, self.max_grad, self.max_roll, self.w_slope, self.w_roll, self.penalty)

        if found:
            end, status = tuple(target), "Target reached"
        else:
            # A* ran out of cells, so `closed` is exactly the reachable set.
            r, c = np.nonzero(closed)
            best = np.argmin(np.hypot((r - target[0]) * self.yres, (c - target[1]) * self.xres))
            end, status = (int(r[best]), int(c[best])), "Closest reachable cell"

        start_node = start[0] * self.cols + start[1]
        current = end[0] * self.cols + end[1]
        path = []
        while current != start_node:
            path.append(divmod(current, self.cols))
            current = int(parent[current])
            if current < 0:
                raise RuntimeError("Could not reconstruct the route.")
        path.append(tuple(start))
        path = np.asarray(path[::-1], dtype=np.int64)

        smooth = _string_pull(path, self.elev, self.gr, self.gc, self.blocked,
                              self.xres, self.yres, self.max_grad, self.max_roll,
                              self.w_slope, self.w_roll, self.penalty)
        return RouteResult(path, smooth, status, self._length_3d(path),
                           self._length_3d(smooth), int(expanded), closed,
                           float(self.illum[path[:, 0], path[:, 1]].mean()),
                           {l.name: (float(np.nanmean(l.data[path[:, 0], path[:, 1]])) if np.isfinite(l.data[path[:, 0], path[:, 1]]).any() else float('nan'),
                                     float(l.data[path[-1, 0], path[-1, 1]]), l.units) for l in self.layers})


def find_rover_route(elevation, start, target, x_resolution, y_resolution, max_slope_deg=15.0):
    """Backward-compatible wrapper returning (path, status, 3D length in meters)."""
    result = TerrainRouter(elevation, x_resolution, y_resolution, max_slope_deg).find_route(start, target)
    return result.path, result.status, result.length_grid


# --------------------------------------------------------------------------- #
# Interactive plot
# --------------------------------------------------------------------------- #
class RouteSelector:
    """Interactive start/target selection with toggleable shade and blocked-cell layers."""

    def __init__(self, fig, ax, router, transform, extent, start, target, show_smoothed=False):
        self.fig, self.ax, self.router, self.transform = fig, ax, router, transform
        self.show_smoothed = show_smoothed
        self.start, self.target = router.snap(start), router.snap(target)

        # Shade layer: dark navy, opacity proportional to how shaded each cell is.
        shade_rgba = np.zeros(router.blocked.shape + (4,), np.float32)
        shade_rgba[..., 2] = 0.25
        shade_rgba[..., 3] = 0.8 * (1.0 - router.illum)
        self.shade_img = ax.imshow(shade_rgba, extent=extent, zorder=2)
        # Blocked layer: red where the rover can't go (NoData, too steep, safety margin).
        self.blocked_img = ax.imshow(np.ma.masked_where(~router.blocked, np.ones(router.blocked.shape)),
                                     cmap=ListedColormap(["red"]), alpha=0.4, extent=extent,
                                     vmin=0, vmax=1, zorder=3)
        self.layers = {"Shade": self.shade_img, "Blocked": self.blocked_img}
        self.keys = ["i", "b"]
        self.colorbars = {}
        # One overlay per extra layer (thermal inertia, ...), off by default; checkbox / number key toggles it.
        for n, layer in enumerate(router.layers, start=1):
            img = ax.imshow(layer.data, cmap=layer.cmap, alpha=0.5, extent=extent, zorder=2,
                            vmin=layer.lo, vmax=layer.hi, visible=False)
            self.layers[layer.name] = img
            self.keys.append(str(n))
            cbar = self.ax.figure.colorbar(img, ax=ax, label=f"{layer.name} {layer.units}".strip(), shrink=0.5, pad=0.02)
            cbar.ax.set_visible(False)
            self.colorbars[layer.name] = cbar
        self.line, = ax.plot([], [], color="cyan", linewidth=2.5, label="Route", zorder=4)
        self.smooth_line, = ax.plot([], [], color="white", linewidth=1.5, linestyle="--",
                                    label="Smoothed route" if show_smoothed else "_nolegend_", zorder=4, visible=show_smoothed)
        marker = dict(edgecolors="black", s=90, zorder=5)
        self.start_marker = ax.scatter([], [], color="lime", marker="o", label="Start", **marker)
        self.target_marker = ax.scatter([], [], color="red", marker="X", label="Target", **marker)
        self.stop_marker = ax.scatter([], [], color="orange", marker="s", label="Closest reachable cell", **marker)
        self.message = fig.text(0.5, 0.17, "", ha="center", va="center", fontsize=10)
        ax.legend(loc="upper right", fontsize=8)

        # Clickable checkboxes (keys i / b do the same).
        n = len(self.layers)
        self.checks = CheckButtons(fig.add_axes([0.02, 0.01, 0.2, 0.035 * n]), list(self.layers),
                                   [layer.get_visible() for layer in self.layers.values()])
        self.checks.on_clicked(self.toggle_layer)
        fig.canvas.mpl_connect("button_press_event", self.on_click)
        fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.update_route()

    def toggle_layer(self, label):
        layer = self.layers[label]
        layer.set_visible(not layer.get_visible())
        if label in self.colorbars:
            self.colorbars[label].ax.set_visible(layer.get_visible())
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        # Keep the checkbox in sync with the key shortcut (i, b, then 1, 2, ... for extra layers).
        if event.key in self.keys:
            self.checks.set_active(self.keys.index(event.key))

    def on_click(self, event):
        # Let the toolbar use mouse clicks while zoom or pan is active.
        toolbar = getattr(self.fig.canvas, "toolbar", None)
        if toolbar is not None and toolbar.mode:
            return
        if event.inaxes is not self.ax or event.button not in (1, 3) or event.xdata is None:
            return
        row, col = rowcol(self.transform, event.xdata, event.ydata)
        if not (0 <= row < self.router.rows and 0 <= col < self.router.cols):
            return
        snapped = self.router.snap((row, col))
        if event.button == 3 or event.key == "shift":
            self.target = snapped
        else:
            self.start = snapped
        self.update_route(note="" if snapped == (row, col) else " (snapped to nearest traversable cell)")

    def update_route(self, note=""):
        self.message.set_text("Computing…")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        try:
            res = self.router.find_route(self.start, self.target)
        except (ValueError, RuntimeError) as exc:
            self.message.set_text(f"Error: {exc}")
            self.fig.canvas.draw_idle()
            return
        t = self.transform
        px, py = xy(t, res.path[:, 0], res.path[:, 1])
        self.line.set_data(px, py)
        if self.show_smoothed:
            sx, sy = xy(t, res.smooth[:, 0], res.smooth[:, 1])
            self.smooth_line.set_data(sx, sy)
        self.start_marker.set_offsets([xy(t, *self.start)])
        self.target_marker.set_offsets([xy(t, *self.target)])
        self.stop_marker.set_offsets([xy(t, *res.path[-1])] if res.status != "Target reached" else np.empty((0, 2)))
        self.ax.set_title(f"Gale Crater: slope-limited route ({res.status})")
        self.message.set_text(
            f"{res.status} · {res.length_grid:.0f} m (3D) · {res.sun_fraction:.0%} sunlit{note}\n"
            + "".join(f"{k}: route avg {m:.3g}, at stop {e:.3g} {u}"
                      f"{'  <-- poor ground to park on' if self.router.layers_by_name[k].is_bad(e) else ''}\n"
                      for k, (m, e, u) in res.layer_stats.items())
            + "Left click: start · Right/shift click: target · keys: " + ", ".join(
                f"{k}={name.lower()}" for k, name in zip(self.keys, self.layers)))
        self.fig.canvas.draw_idle()
        print(f"{res.status}; 3D length {res.length_grid:.1f} m; sunlit {res.sun_fraction:.0%}; "
              f"stopping pixel: {tuple(map(int, res.path[-1]))}")


def load_patch(tif, size_m, center, center_lonlat=None):
    with rasterio.open(tif) as src:
        if src.crs is None or not src.crs.is_projected or src.crs.linear_units != "metre":
            raise ValueError("A projected raster with meter units is required.")
        if src.transform.b != 0 or src.transform.d != 0 or src.transform.a <= 0 or src.transform.e >= 0:
            raise ValueError("Expected a north-up raster without rotation.")
        xres, yres = src.res
        width = min(src.width, max(2, round(size_m / xres)))
        height = min(src.height, max(2, round(size_m / yres)))
        if center_lonlat:
            xs, ys = warp_transform(src.crs.geodetic_crs, src.crs, [center_lonlat[0]], [center_lonlat[1]])
            center = (xs[0], ys[0])
        if center:
            crow, ccol = src.index(*center)
            if not (0 <= crow < src.height and 0 <= ccol < src.width):
                raise ValueError("--center is outside the raster.")
        else:
            crow, ccol = src.height // 2, src.width // 2
        col_off = int(min(max(0, ccol - width // 2), src.width - width))
        row_off = int(min(max(0, crow - height // 2), src.height - height))
        window = Window(col_off, row_off, width, height)
        dem = src.read(1, window=window, masked=True).astype(np.float64)
        dem = (dem * src.scales[0] + src.offsets[0]).filled(np.nan)
        dem[~np.isfinite(dem)] = np.nan
        return dem.astype(np.float32), src.window_transform(window), src.window_bounds(window), xres, yres, src.crs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tif", type=Path, default=Path(__file__).resolve().parents[2] / "data/gale_20m_mosaic.tif")
    parser.add_argument("--size-m", type=float, default=1000, help="Approximate patch width and height in meters (default: %(default)s)")
    parser.add_argument("--center", type=float, nargs=2, metavar=("E", "N"), help="Patch center in projecte --d coordinates (default: raster center)")
    parser.add_argument("--center-lonlat", type=float, nargs=2, metavar=("LON", "LAT"), help="Patch center as longitude latitude in degrees (alternative to --center)")
    parser.add_argument("--max-slope", type=float, default=12, help="Maximum slope along travel in degrees (default: %(default)s)")
    parser.add_argument("--max-roll", type=float, default=10, help="Maximum side slope in degrees (default: %(default)s)")
    parser.add_argument("--rover-radius", type=float, default=1.0, help="Safety margin around blocked cells in meters (default: %(default)s)")
    parser.add_argument("--smooth", type=float, default=1.0, help="Gaussian sigma in pixels applied before slope (default: %(default)s)")
    parser.add_argument("--slope-weight", type=float, default=10.0, help="Weight of the slope cost penalty (default: %(default)s)")
    parser.add_argument("--roll-weight", type=float, default=10.0, help="Weight of the side-slope cost penalty (default: %(default)s)")
    parser.add_argument("--sun-az", type=float, default=270, help="Sun azimuth in degrees clockwise from north: 90=E, 180=S, 270=W (default: %(default)s)")
    parser.add_argument("--sun-alt", type=float, default=30, help="Sun elevation above the horizon in degrees (default: %(default)s)")
    parser.add_argument("--sun-weight", type=float, default=2.0, help="Extra cost for shaded cells; 0 ignores sun, higher detours more (default: %(default)s)")
    parser.add_argument("--shadow-dist", type=float, default=500, help="Max distance in meters to look for shadow-casting terrain; 0 disables cast shadows (default: %(default)s)")
    parser.add_argument("--show-smoothed", action="store_true", help="Also draw the any-angle smoothed route as a dashed line")
    parser.add_argument("--thermal-inertia", type=Path, help="Thermal inertia GeoTIFF (global or regional; auto-cropped and resampled to the DEM patch)")
    parser.add_argument("--ti-weight", type=float, default=3.0, help="Cost penalty on the softest (low thermal inertia) ground (default: %(default)s)")
    parser.add_argument("--ti-poor", type=float, default=250, help="TI at or below this gets the full penalty: loose dust/sand (default: %(default)s)")
    parser.add_argument("--ti-good", type=float, default=300, help="TI at or above this gets no penalty: firm, rocky or cemented ground (default: %(default)s)")
    parser.add_argument("--ti-block", type=float, default=200,help="Hard limit: cells with TI below this value are impassable (e.g. 100 to avoid deep dust)")
    parser.add_argument("--ti-units", default="TIU", help="Units label (default: %(default)s; J m^-2 K^-1 s^-1/2)")
    parser.add_argument("--layer-band", type=int, default=1, help="Band of the layer file to use; ISIS .cub files can hold several (default: %(default)s)")
    parser.add_argument("--layer-crs", help="CRS to assume for global layers if the file has none or a wrong one (e.g. 'EPSG:4326' or a Mars WKT/PROJ string)")
    parser.add_argument("--layer-resampling", default="bilinear", choices=["nearest", "bilinear", "cubic"], help="How coarse layers are interpolated onto the 1 m grid (default: %(default)s)")
    parser.add_argument("--start", type=int, nargs=2, metavar=("ROW", "COL"), help="Start pixel within the patch")
    parser.add_argument("--target", type=int, nargs=2, metavar=("ROW", "COL"), help="Target pixel within the patch")
    parser.add_argument("--save", type=Path, help="Save the plot to this file instead of displaying it")
    args = parser.parse_args()
    if args.size_m <= 0:
        parser.error("--size-m must be positive")

    dem, transform, bounds, xres, yres, dem_crs = load_patch(args.tif, args.size_m, args.center, args.center_lonlat)
    rows, cols = dem.shape
    print(f"Patch: {cols} × {rows} pixels, {cols * xres:.1f} × {rows * yres:.1f} m")
    print(f"Pixel size: {xres:.2f} × {yres:.2f} m")

    layers = []
    if args.thermal_inertia:
        if args.ti_good <= args.ti_poor:
            parser.error("--ti-good must be greater than --ti-poor")
        data, p2, p98 = load_global_layer(args.thermal_inertia, dem_crs, transform, dem.shape,
                                          args.layer_crs, args.layer_resampling, args.layer_band)
        print(f"Thermal inertia over patch: {np.nanmin(data):.4g} to {np.nanmax(data):.4g} {args.ti_units} "
              f"(whole file, 2nd-98th pct: {p2:.4g} to {p98:.4g})")
        layers.append(CostLayer("Thermal inertia", data, args.ti_weight, lo=args.ti_poor, hi=args.ti_good,
                                block_below=args.ti_block, invert=True, units=args.ti_units, cmap="viridis"))

    router = TerrainRouter(dem, xres, yres, args.max_slope, args.max_roll, args.rover_radius,
                           args.smooth, args.slope_weight, args.roll_weight,
                           args.sun_az, args.sun_alt, args.sun_weight, args.shadow_dist, layers)
    start = tuple(args.start) if args.start else (rows // 20, cols // 20)
    target = tuple(args.target) if args.target else (rows - 1 - rows // 20, cols - 1 - cols // 20)

    fig, ax = plt.subplots(figsize=(9, 8))
    cmap = plt.colormaps["terrain"].copy()
    left, bottom, right, top = bounds
    extent = (left, right, bottom, top)

    # Hillshade so the relief is readable.
    valid = np.isfinite(dem)
    filled = np.where(valid, dem, np.nanmean(dem))
    norm = Normalize(np.nanmin(dem), np.nanmax(dem))
    rgb = LightSource(azdeg=args.sun_az, altdeg=args.sun_alt).shade(filled, cmap=cmap, norm=norm, blend_mode="overlay",
                                                  vert_exag=2, dx=xres, dy=yres)
    rgb[~valid] = (0.33, 0.33, 0.33, 1.0)
    ax.imshow(rgb, origin="upper", extent=extent, zorder=1)
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="Elevation (m)")
    ax.set(xlabel="Projected easting (m)", ylabel="Projected northing (m)")
    fig.tight_layout(rect=(0, 0.25, 1, 0.97))

    # Keep the selector alive for as long as the figure is open.
    selector = RouteSelector(fig, ax, router, transform, extent, start, target, args.show_smoothed)
    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"Saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
