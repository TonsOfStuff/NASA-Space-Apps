/** A point in the active region's projected Mars coordinate system, in metres. */
export type MarsCoordinate = [number, number];

export type MarkerKind = "start" | "destination";
export type LayerId = "imagery" | "elevation" | "slope" | "coverage";
export type LayerVisibility = Record<LayerId, boolean>;

/** Source bounds and native pixel dimensions in a Mars-projected coordinate system. */
export interface RegionManifest {
  id: string;
  name: string;
  projection: string;
  proj4: string;
  extent: [number, number, number, number];
  width: number;
  height: number;
  pixelSize: [number, number];
  elevationRange: [number, number];
  /** Top-left-origin tile grid; resolutions are projected metres per pixel. */
  tileSize: number;
  minZoom: number;
  maxZoom: number;
  resolutions: number[];
  /** Imagery can contain more physical pixels than the shared tile grid. */
  imageryPixelRatio?: number;
  layers: Record<LayerId, string>;
  /** URL template for little-endian Float32 terrain tiles; NaN means NoData. */
  terrain: string;
  notes: string[];
}

/** Shared interaction contract for the 2D and 3D maps. */
export interface MapViewProps {
  region: RegionManifest;
  layers: LayerVisibility;
  start: MarsCoordinate | null;
  destination: MarsCoordinate | null;
  stop: MarsCoordinate | null;
  route: MarsCoordinate[];
  activeTool: MarkerKind | null;
  onPlace: (kind: MarkerKind, coordinate: MarsCoordinate) => void;
}
