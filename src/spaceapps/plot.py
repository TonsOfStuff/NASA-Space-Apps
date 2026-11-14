"""Visualize a Gale Crater elevation patch and a slope-limited example route."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Button
import numpy as np
import rasterio
from rasterio.transform import rowcol, xy
from rasterio.windows import Window
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


class TerrainRouter:
    """Cache terrain connectivity and the shortest-path tree for the current start."""

    def __init__(self, elevation, x_resolution, y_resolution, max_slope_deg=15.0):
        if elevation.ndim != 2:
            raise ValueError("Elevation must be a 2D array.")
        if not 0 < max_slope_deg < 90:
            raise ValueError("Maximum slope must be between 0 and 90 degrees.")
        if x_resolution <= 0 or y_resolution <= 0:
            raise ValueError("Pixel dimensions must be positive.")
        self.elevation = elevation
        self.rows, self.cols = elevation.shape
        self.x_resolution, self.y_resolution = x_resolution, y_resolution
        rows, cols = elevation.shape
        max_gradient = np.tan(np.deg2rad(max_slope_deg))
        node_ids = np.arange(rows * cols).reshape(rows, cols)
        edge_from, edge_to, weights = [], [], []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)):
            r1 = slice(max(0, -dr), min(rows, rows - dr))
            c1 = slice(max(0, -dc), min(cols, cols - dc))
            r2 = slice(max(0, dr), min(rows, rows + dr))
            c2 = slice(max(0, dc), min(cols, cols + dc))
            distance = np.hypot(dc * x_resolution, dr * y_resolution)
            gradient = np.abs(elevation[r2, c2] - elevation[r1, c1]) / distance
            valid = np.isfinite(gradient) & (gradient <= max_gradient)
            if dr and dc:
                # Do not cut diagonally across a missing-data corner.
                valid &= np.isfinite(elevation[r1, c2]) & np.isfinite(elevation[r2, c1])
            edge_from.append(node_ids[r1, c1][valid])
            edge_to.append(node_ids[r2, c2][valid])
            weights.append(distance * (1 + (gradient[valid] / max_gradient) ** 2))

        graph = csr_matrix((np.concatenate(weights),
                            (np.concatenate(edge_from), np.concatenate(edge_to))),
                           shape=(rows * cols, rows * cols))
        self.graph = graph
        self.cached_start = None
        self.costs = None
        self.predecessors = None

    def find_route(self, start, target):
        rows, cols = self.rows, self.cols
        elevation = self.elevation
        x_resolution, y_resolution = self.x_resolution, self.y_resolution
        for name, (row, col) in (("start", start), ("target", target)):
            if not (0 <= row < rows and 0 <= col < cols):
                raise ValueError(f"The {name} point is outside the elevation patch.")
            if not np.isfinite(elevation[row, col]):
                raise ValueError(f"The {name} point is in a NoData area; choose another point.")
        start_node = start[0] * cols + start[1]
        target_node = target[0] * cols + target[1]
        if self.cached_start != start_node:
            self.costs, self.predecessors = dijkstra(
                self.graph, directed=True, indices=start_node, return_predecessors=True
            )
            self.cached_start = start_node
        costs, predecessors = self.costs, self.predecessors
        if np.isfinite(costs[target_node]):
            best_node, status = target_node, "Target reached"
        else:
            reachable = np.flatnonzero(np.isfinite(costs))
            r, c = np.divmod(reachable, cols)
            distance_to_target = np.hypot((r - target[0]) * y_resolution,
                                          (c - target[1]) * x_resolution)
            best_node = reachable[np.argmin(distance_to_target)]
            status = "Closest reachable cell"

        path = []
        current = int(best_node)
        while current != start_node:
            path.append(divmod(current, cols))
            current = int(predecessors[current])
            if current < 0:
                raise RuntimeError("Could not reconstruct the route.")
        path.append(start)
        path = np.asarray(path[::-1])
        steps = np.diff(path, axis=0)
        distance_m = float(np.hypot(steps[:, 1] * x_resolution,
                                    steps[:, 0] * y_resolution).sum())
        return path, status, distance_m



def find_rover_route(elevation, start, target, x_resolution, y_resolution, max_slope_deg=15.0):
    """Find a route using the same rules as the interactive terrain router."""
    return TerrainRouter(elevation, x_resolution, y_resolution, max_slope_deg).find_route(start, target)


class RouteSelector:
    """Manage interactive start and target selection on a terrain plot."""

    def __init__(self, fig, ax, dem, transform, xres, yres, max_slope, start, target):
        self.fig, self.ax = fig, ax
        self.dem, self.transform = dem, transform
        self.xres, self.yres, self.max_slope = xres, yres, max_slope
        self.start, self.target = start, target
        self.router = TerrainRouter(dem, xres, yres, max_slope)
        self.mode = "start"
        self.unreachable_overlay = None
        self.line, = ax.plot([], [], color="cyan", linewidth=2.5, label="Example route")
        self.start_marker = ax.scatter([], [], color="lime", marker="o", edgecolors="black", s=90, label="Start", zorder=3)
        self.target_marker = ax.scatter([], [], color="red", marker="X", edgecolors="black", s=90, label="Target", zorder=3)
        self.stop_marker = ax.scatter([], [], color="orange", marker="s", edgecolors="black", s=90, label="Closest reachable cell", zorder=3)
        self.message = fig.text(0.5, 0.11, "", ha="center", va="center", fontsize=10)
        self.start_button = Button(fig.add_axes([0.14, 0.025, 0.2, 0.055]), "Set start")
        self.target_button = Button(fig.add_axes([0.38, 0.025, 0.2, 0.055]), "Set target")
        self.reachability_button = Button(fig.add_axes([0.62, 0.025, 0.25, 0.055]), "Show unreachable")
        self.start_button.on_clicked(lambda event: self.select("start"))
        self.target_button.on_clicked(lambda event: self.select("target"))
        self.reachability_button.on_clicked(self.toggle_reachability)
        self.connection = fig.canvas.mpl_connect("button_press_event", self.on_click)
        ax.legend(loc="best")
        self.update_route()

    def select(self, mode):
        self.mode = mode
        self.message.set_text(f"Click the map to set the {mode} point.")
        self.fig.canvas.draw_idle()

    def toggle_reachability(self, event):
        if self.unreachable_overlay is None:
            cmap = ListedColormap(["#d62728"])
            cmap.set_bad((0, 0, 0, 0))
            self.unreachable_overlay = self.ax.imshow(
                self.unreachable_mask(), cmap=cmap, vmin=0, vmax=1, alpha=0.4,
                origin="upper", extent=self.ax.images[0].get_extent(),
                interpolation="nearest", zorder=1.5,
            )
            self.reachability_button.label.set_text("Hide unreachable")
            self.message.set_text("Red = unreachable from start with current slope limit; gray = NoData")
        else:
            self.unreachable_overlay.remove()
            self.unreachable_overlay = None
            self.reachability_button.label.set_text("Show unreachable")
            self.message.set_text("Reachability overlay hidden. Select start or target, then click the map.")
        self.fig.canvas.draw_idle()

    def unreachable_mask(self):
        reachable = np.isfinite(self.router.costs).reshape(self.dem.shape)
        # Show valid but disconnected cells in red; leave NoData visible in gray.
        return np.ma.array(np.zeros(self.dem.shape, dtype=np.uint8),
                           mask=reachable | ~np.isfinite(self.dem))

    def on_click(self, event):
        # Let the toolbar use mouse clicks while zoom or pan is active.
        toolbar = getattr(self.fig.canvas, "toolbar", None)
        if toolbar is not None and toolbar.mode:
            return
        if event.inaxes is not self.ax or event.button != 1 or event.xdata is None or event.ydata is None:
            return
        row, col = rowcol(self.transform, event.xdata, event.ydata)
        if not (0 <= row < self.dem.shape[0] and 0 <= col < self.dem.shape[1]):
            return
        if not np.isfinite(self.dem[row, col]):
            self.message.set_text("No elevation data there. Choose another point.")
            self.fig.canvas.draw_idle()
            return
        if self.mode == "start":
            self.start = (row, col)
        else:
            self.target = (row, col)
        self.update_route()

    def update_route(self):
        path, status, distance = self.router.find_route(self.start, self.target)
        path_x, path_y = xy(self.transform, path[:, 0], path[:, 1])
        self.line.set_data(path_x, path_y)
        self.start_marker.set_offsets([xy(self.transform, *self.start)])
        self.target_marker.set_offsets([xy(self.transform, *self.target)])
        self.stop_marker.set_offsets([xy(self.transform, *path[-1])] if status != "Target reached" else np.empty((0, 2)))
        if self.unreachable_overlay is not None:
            self.unreachable_overlay.set_data(self.unreachable_mask())
        self.ax.set_title(f"Gale Crater: slope-limited route ({status})")
        overlay_note = " · Red = unreachable" if self.unreachable_overlay is not None else ""
        self.message.set_text(f"{status} · {distance:.0f} m · Max slope: {self.max_slope:g}°{overlay_note}")
        self.fig.canvas.draw_idle()
        print(f"{status}; horizontal route length: {distance:.1f} m; stopping pixel: {tuple(map(int, path[-1]))}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tif", type=Path, default=Path(__file__).resolve().parent / "gale_20m_mosaic.tif")
    parser.add_argument("--size-m", type=float, default=500, help="Approximate width and height in meters (default: 500)")
    parser.add_argument("--max-slope", type=float, default=15, help="Maximum slope in degrees (default: 15)")
    parser.add_argument("--start", type=int, nargs=2, metavar=("ROW", "COL"), help="Start pixel within the patch")
    parser.add_argument("--target", type=int, nargs=2, metavar=("ROW", "COL"), help="Target pixel within the patch")
    parser.add_argument("--save", type=Path, help="Save the plot to this file instead of displaying it")
    args = parser.parse_args()
    if args.size_m <= 0:
        parser.error("--size-m must be positive")

    with rasterio.open(args.tif) as src:
        if src.crs is None or not src.crs.is_projected or src.crs.linear_units != "metre":
            raise ValueError("A projected raster with meter units is required.")
        if src.transform.b != 0 or src.transform.d != 0 or src.transform.a <= 0 or src.transform.e >= 0:
            raise ValueError("Expected a north-up raster without rotation.")
        xres, yres = src.res
        width = min(src.width, max(2, round(args.size_m / xres)))
        height = min(src.height, max(2, round(args.size_m / yres)))
        window = Window((src.width - width) // 2, (src.height - height) // 2, width, height)
        dem = src.read(1, window=window, masked=True).astype(np.float64)
        dem = (dem * src.scales[0] + src.offsets[0]).filled(np.nan)
        dem[~np.isfinite(dem)] = np.nan
        transform = src.window_transform(window)
        bounds = src.window_bounds(window)

    rows, cols = dem.shape
    start = tuple(args.start) if args.start else (max(0, rows // 20), max(0, cols // 20))
    target = tuple(args.target) if args.target else (min(rows - 1, rows - 1 - rows // 20), min(cols - 1, cols - 1 - cols // 20))
    print(f"Patch: {cols} × {rows} pixels, {cols*xres:.1f} × {rows*yres:.1f} m")
    print(f"Pixel size: {xres:.2f} × {yres:.2f} m")

    fig, ax = plt.subplots(figsize=(9, 8))
    cmap = plt.colormaps["terrain"].copy()
    cmap.set_bad("#555555")
    left, bottom, right, top = bounds
    image = ax.imshow(dem, cmap=cmap, origin="upper", extent=(left, right, bottom, top), interpolation="nearest")
    fig.colorbar(image, ax=ax, label="Elevation (m)")
    ax.set(xlabel="Projected easting (m)", ylabel="Projected northing (m)")
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    # Keep the selector alive for as long as the figure is open.
    selector = RouteSelector(fig, ax, dem, transform, xres, yres, args.max_slope, start, target)
    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"Saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
