import { type Camera, type PerspectiveCamera, type Vector3 } from "three";
import type { RegionManifest } from "./types";

export type Tile = { x: number; y: number };
export type Selection = { z: number; tiles: Tile[]; key: string };

// A fixed detail area keeps close-up terrain bounded. Coarser terrain fills
// the rest of the region in the 3D scene.
export const MAX_TERRAIN_TILES = 25;
const DETAIL_BIAS = 0.2;
const ZOOM_HYSTERESIS = 0.05;

export function pickTiles(region: RegionManifest, camera: Camera, target: Vector3, heightPx: number, previousZoom?: number): Selection {
  const perspective = camera as PerspectiveCamera;
  const distance = camera.position.distanceTo(target);
  const groundPerPixel = 2 * distance * Math.tan((perspective.fov || 45) * Math.PI / 360) / Math.max(heightPx, 1);
  // Favor the next level before the current grid becomes visibly coarse.
  // Only camera distance, field of view, and viewport size affect detail.
  const idealZoom = Math.log2(region.resolutions[region.minZoom] / Math.max(groundPerPixel, 0.001)) + DETAIL_BIAS;
  // A small deadband prevents floating-point drift near a threshold from
  // changing detail repeatedly while the camera orbits at the same distance.
  const selectedZoom = previousZoom !== undefined && Math.abs(idealZoom - previousZoom) <= 0.5 + ZOOM_HYSTERESIS
    ? previousZoom : Math.round(idealZoom);
  const z = Math.max(region.minZoom, Math.min(region.maxZoom, selectedZoom));

  const centerX = (region.extent[0] + region.extent[2]) / 2;
  const centerY = (region.extent[1] + region.extent[3]) / 2;
  const span = region.tileSize * region.resolutions[z];
  const cols = Math.ceil((region.extent[2] - region.extent[0]) / span);
  const rows = Math.ceil((region.extent[3] - region.extent[1]) / span);
  const width = Math.min(cols, Math.sqrt(MAX_TERRAIN_TILES));
  const height = Math.min(rows, Math.sqrt(MAX_TERRAIN_TILES));
  const targetX = Math.floor((target.x + centerX - region.extent[0]) / span);
  const targetY = Math.floor((region.extent[3] - (centerY - target.z)) / span);
  const x0 = Math.max(0, Math.min(cols - width, targetX - Math.floor(width / 2)));
  const y0 = Math.max(0, Math.min(rows - height, targetY - Math.floor(height / 2)));
  const tiles: Tile[] = [];
  for (let y = y0; y < y0 + height; y++) for (let x = x0; x < x0 + width; x++) tiles.push({ x, y });
  return { z, tiles, key: `${z}:${tiles.map(({ x, y }) => `${x},${y}`).join(";")}` };
}

/** Pick a small, whole-region backdrop to display beyond the detailed area. */
export function overviewZoom(region: RegionManifest): number | null {
  let zoom = region.minZoom;
  const initialSpan = region.tileSize * region.resolutions[zoom];
  const initialCols = Math.ceil((region.extent[2] - region.extent[0]) / initialSpan);
  const initialRows = Math.ceil((region.extent[3] - region.extent[1]) / initialSpan);
  if (initialCols * initialRows > 4) return null;
  while (zoom < region.maxZoom) {
    const span = region.tileSize * region.resolutions[zoom + 1];
    const cols = Math.ceil((region.extent[2] - region.extent[0]) / span);
    const rows = Math.ceil((region.extent[3] - region.extent[1]) / span);
    if (cols * rows > 4) break;
    zoom++;
  }
  return zoom;
}
