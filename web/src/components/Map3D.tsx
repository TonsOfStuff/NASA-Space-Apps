import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree, type ThreeEvent } from "@react-three/fiber";
import {
  BufferAttribute,
  BufferGeometry,
  CanvasTexture,
  DoubleSide,
  NearestFilter,
  Plane,
  SRGBColorSpace,
  Texture,
  Vector3,
} from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";
import { LineSegments2 } from "three/examples/jsm/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/examples/jsm/lines/LineSegmentsGeometry.js";
import { overviewZoom, pickTiles, type Selection, type Tile } from "../terrainLod";
import type { EnvironmentalLayerId, LayerId, MapViewProps, MarsCoordinate, RegionManifest } from "../types";

type Loaded = { z: number; tiles: Tile[]; data: Map<string, Float32Array | null> };
type LodStatus = { zoom: number; loading: boolean; error: string | null };
type HeightLookup = { current: ((coordinate: MarsCoordinate) => number | null) | null };
type Map3DProps = MapViewProps & { colorizeImagery: boolean };
type RenderLayer = LayerId | "estimated" | "missing";
type Edge = "north" | "south" | "west" | "east";
type TerrainBounds = { left: number; right: number; north: number; south: number };

const layerOrder: LayerId[] = ["imagery", "elevation", "shade", "thermal", "slope", "coverage"];
const terrainCache = new Map<string, Promise<Float32Array | null>>();
const imageCache = new Map<string, Promise<HTMLImageElement | null>>();

function cached<T>(cache: Map<string, Promise<T>>, key: string, limit: number, load: () => Promise<T>): Promise<T> {
  const present = cache.get(key);
  if (present) {
    cache.delete(key);
    cache.set(key, present);
    return present;
  }
  const promise = load().catch((error: unknown) => {
    cache.delete(key);
    throw error;
  });
  cache.set(key, promise);
  while (cache.size > limit) cache.delete(cache.keys().next().value as string);
  return promise;
}

function tileUrl(template: string, z: number, x: number, y: number): string {
  return template.replaceAll("{z}", String(z)).replaceAll("{x}", String(x)).replaceAll("{y}", String(y));
}

function inRegion(region: RegionManifest, z: number, x: number, y: number): boolean {
  const resolution = region.resolutions[z];
  const span = resolution * region.tileSize;
  return x >= 0 && y >= 0 && x < Math.ceil((region.extent[2] - region.extent[0]) / span)
    && y < Math.ceil((region.extent[3] - region.extent[1]) / span);
}

function loadTerrain(region: RegionManifest, z: number, x: number, y: number): Promise<Float32Array | null> {
  if (!inRegion(region, z, x, y)) return Promise.resolve(null);
  const url = tileUrl(region.terrain, z, x, y);
  return cached(terrainCache, url, 80, async () => {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Terrain tile could not be loaded (${response.status}).`);
    const buffer = await response.arrayBuffer();
    const count = region.tileSize * region.tileSize;
    if (buffer.byteLength !== count * 4) throw new Error("A terrain tile has an invalid size.");
    const view = new DataView(buffer);
    const result = new Float32Array(count);
    for (let i = 0; i < count; i++) result[i] = view.getFloat32(i * 4, true);
    return result;
  });
}

function loadImage(region: RegionManifest, layer: RenderLayer, z: number, x: number, y: number,
                   environmentalLayers?: Record<EnvironmentalLayerId, string>): Promise<HTMLImageElement | null> {
  if (!inRegion(region, z, x, y)) return Promise.resolve(null);
  const template = layer === "estimated" || layer === "missing" ? region.coverageLayers?.[layer]
    : layer === "shade" || layer === "thermal" ? environmentalLayers?.[layer] : region.layers[layer];
  if (!template) return Promise.resolve(null);
  const url = tileUrl(template, z, x, y);
  return cached(imageCache, url, 160, () => new Promise((resolve, reject) => {
    const image = new Image();
    image.crossOrigin = "anonymous";
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("A map layer tile could not be loaded."));
    image.src = url;
  }));
}

function tileKey(x: number, y: number): string { return `${x}/${y}`; }
function neighbours(tile: Tile): Tile[] {
  return [tile, { x: tile.x + 1, y: tile.y }, { x: tile.x, y: tile.y + 1 }, { x: tile.x + 1, y: tile.y + 1 }];
}

function sample(loaded: Loaded, region: RegionManifest, col: number, row: number): number {
  if (col < 0 || row < 0) return NaN;
  const size = region.tileSize;
  const tileX = Math.floor(col / size);
  const tileY = Math.floor(row / size);
  const tile = loaded.data.get(tileKey(tileX, tileY));
  return tile?.[(row % size) * size + (col % size)] ?? NaN;
}

function heightAt(loaded: Loaded, region: RegionManifest, [x, y]: MarsCoordinate): number | null {
  const resolution = region.resolutions[loaded.z];
  const col = (x - region.extent[0]) / resolution - 0.5;
  const row = (region.extent[3] - y) / resolution - 0.5;
  if (!Number.isFinite(col) || !Number.isFinite(row) || col < 0 || row < 0) return null;
  const c = Math.floor(col);
  const r = Math.floor(row);
  const fx = col - c;
  const fy = row - r;
  const a = sample(loaded, region, c, r);
  const b = sample(loaded, region, c + 1, r);
  const lower = sample(loaded, region, c, r + 1);
  const d = sample(loaded, region, c + 1, r + 1);
  if (fx + fy <= 1) return [a, b, lower].every(Number.isFinite) ? a + fx * (b - a) + fy * (lower - a) : null;
  return [b, lower, d].every(Number.isFinite) ? d + (1 - fx) * (lower - d) + (1 - fy) * (b - d) : null;
}

function toScene(region: RegionManifest, [x, y]: MarsCoordinate, height: number): [number, number, number] {
  return [x - (region.extent[0] + region.extent[2]) / 2, height - region.elevationRange[0],
    (region.extent[1] + region.extent[3]) / 2 - y];
}

function makeGeometry(region: RegionManifest, loaded: Loaded, tile: Tile, overview: Loaded | null, edges: Edge[]): BufferGeometry {
  const size = region.tileSize;
  const side = size + 1;
  const resolution = region.resolutions[loaded.z];
  const vertexCount = side * side + edges.length * side;
  const positions = new Float32Array(vertexCount * 3);
  const uvs = new Float32Array(vertexCount * 2);
  const normals = new Float32Array(vertexCount * 3);
  const valid = new Uint8Array(side * side);
  const indices: number[] = [];
  for (let row = 0; row < side; row++) {
    for (let col = 0; col < side; col++) {
      const i = row * side + col;
      const globalCol = tile.x * size + col;
      const globalRow = tile.y * size + row;
      const elevation = sample(loaded, region, globalCol, globalRow);
      valid[i] = Number.isFinite(elevation) ? 1 : 0;
      const x = region.extent[0] + (globalCol + 0.5) * resolution;
      const y = region.extent[3] - (globalRow + 0.5) * resolution;
      const point = toScene(region, [x, y], valid[i] ? elevation : region.elevationRange[0]);
      positions[i * 3] = point[0]; positions[i * 3 + 1] = point[1]; positions[i * 3 + 2] = point[2];
      uvs[i * 2] = (col + 0.5) / side; uvs[i * 2 + 1] = 1 - (row + 0.5) / side;
      if (valid[i]) {
        // Use the same nearby samples on both sides of a tile seam so that
        // lighting is continuous as well as the underlying geometry.
        const west = sample(loaded, region, globalCol - 1, globalRow);
        const east = sample(loaded, region, globalCol + 1, globalRow);
        const north = sample(loaded, region, globalCol, globalRow - 1);
        const south = sample(loaded, region, globalCol, globalRow + 1);
        const dx = Number.isFinite(west) && Number.isFinite(east) ? (east - west) / (2 * resolution)
          : Number.isFinite(east) ? (east - elevation) / resolution : Number.isFinite(west) ? (elevation - west) / resolution : 0;
        const dz = Number.isFinite(north) && Number.isFinite(south) ? (south - north) / (2 * resolution)
          : Number.isFinite(south) ? (south - elevation) / resolution : Number.isFinite(north) ? (elevation - north) / resolution : 0;
        const length = Math.hypot(dx, 1, dz);
        normals[i * 3] = -dx / length; normals[i * 3 + 1] = 1 / length; normals[i * 3 + 2] = -dz / length;
      }
    }
  }
  for (let row = 0; row < size; row++) for (let col = 0; col < size; col++) {
    const a = row * side + col, b = a + 1, c = a + side, d = c + 1;
    if (valid[a] && valid[b] && valid[c]) indices.push(a, c, b);
    if (valid[b] && valid[c] && valid[d]) indices.push(b, c, d);
  }
  // Join the detailed edge to the coarse surface at the same coordinates.
  // The two grids average heights differently, so a vertical strip prevents
  // cracks without overlapping the two terrain surfaces.
  if (overview) edges.forEach((edge, edgeIndex) => {
    const edgeValid = new Uint8Array(side);
    const base = side * side + edgeIndex * side;
    const edgeVertex = (offset: number) => edge === "north" ? offset : edge === "south" ? size * side + offset
      : edge === "west" ? offset * side : offset * side + size;
    for (let offset = 0; offset < side; offset++) {
      const source = edgeVertex(offset);
      const dest = base + offset;
      const col = source % side, row = Math.floor(source / side);
      const x = region.extent[0] + (tile.x * size + col + 0.5) * resolution;
      const y = region.extent[3] - (tile.y * size + row + 0.5) * resolution;
      const coarseHeight = valid[source] ? heightAt(overview, region, [x, y]) : null;
      edgeValid[offset] = coarseHeight === null ? 0 : 1;
      positions[dest * 3] = positions[source * 3];
      positions[dest * 3 + 1] = coarseHeight === null ? positions[source * 3 + 1] : coarseHeight - region.elevationRange[0];
      positions[dest * 3 + 2] = positions[source * 3 + 2];
      normals.set(normals.subarray(source * 3, source * 3 + 3), dest * 3);
      uvs.set(uvs.subarray(source * 2, source * 2 + 2), dest * 2);
    }
    for (let offset = 0; offset < size; offset++) {
      if (!edgeValid[offset] || !edgeValid[offset + 1]) continue;
      const a = edgeVertex(offset), b = edgeVertex(offset + 1), c = base + offset, d = c + 1;
      indices.push(a, c, b, b, c, d);
    }
  });
  const geometry = new BufferGeometry();
  geometry.setAttribute("position", new BufferAttribute(positions, 3));
  geometry.setAttribute("uv", new BufferAttribute(uvs, 2));
  geometry.setAttribute("normal", new BufferAttribute(normals, 3));
  geometry.setIndex(indices);
  return geometry;
}

function TileSurface({ region, loaded, tile, layers, coverage, environmentalLayers, activeTool, onPlace, onError, colorizeImagery, clipPlanes, stitchOverview = null }: Pick<MapViewProps, "region" | "layers" | "coverage" | "environmentalLayers" | "activeTool" | "onPlace"> & {
  loaded: Loaded; tile: Tile; onError: (message: string) => void; colorizeImagery: boolean; clipPlanes?: Plane[]; stitchOverview?: Loaded | null;
}) {
  const hasTile = (x: number, y: number) => loaded.tiles.some((candidate) => candidate.x === x && candidate.y === y);
  const stitchNorth = !!stitchOverview && !hasTile(tile.x, tile.y - 1);
  const stitchSouth = !!stitchOverview && !hasTile(tile.x, tile.y + 1);
  const stitchWest = !!stitchOverview && !hasTile(tile.x - 1, tile.y);
  const stitchEast = !!stitchOverview && !hasTile(tile.x + 1, tile.y);
  // A changed view can add or remove tiles at the same level. Keep the
  // geometry for tiles whose surrounding elevation samples are unchanged.
  const geometry = useMemo(() => makeGeometry(region, loaded, tile, stitchOverview,
    ([stitchNorth && "north", stitchSouth && "south", stitchWest && "west", stitchEast && "east"] as const)
      .filter((edge): edge is Edge => !!edge)), [
    region, loaded.z, tile.x, tile.y, stitchOverview, stitchNorth, stitchSouth, stitchWest, stitchEast,
    loaded.data.get(tileKey(tile.x - 1, tile.y - 1)), loaded.data.get(tileKey(tile.x, tile.y - 1)), loaded.data.get(tileKey(tile.x + 1, tile.y - 1)),
    loaded.data.get(tileKey(tile.x - 1, tile.y)), loaded.data.get(tileKey(tile.x, tile.y)), loaded.data.get(tileKey(tile.x + 1, tile.y)),
    loaded.data.get(tileKey(tile.x - 1, tile.y + 1)), loaded.data.get(tileKey(tile.x, tile.y + 1)), loaded.data.get(tileKey(tile.x + 1, tile.y + 1)),
  ]);
  const [map, setMap] = useState<CanvasTexture | null>(null);
  useEffect(() => () => geometry.dispose(), [geometry]);

  useEffect(() => {
    let cancelled = false;
    let created: CanvasTexture | null = null;
    const enabled: RenderLayer[] = layerOrder.filter((layer) => layers[layer] && layer !== "coverage");
    if (layers.coverage) {
      if (region.coverageLayers) {
        if (coverage.estimated) enabled.push("estimated");
        if (coverage.missing) enabled.push("missing");
      } else {
        enabled.push("coverage");
      }
    }
    Promise.all(enabled.map(async (layer) => ({
      layer,
      images: await Promise.all(neighbours(tile).map(({ x, y }) => loadImage(region, layer, loaded.z, x, y, environmentalLayers))),
    }))).then((entries) => {
      if (cancelled) return;
      const size = region.tileSize;
      // Imagery can contain more visual detail than the elevation grid. Keep
      // that extra resolution when compositing the layer textures.
      const imageryRatio = (region as RegionManifest & { imageryPixelRatio?: number }).imageryPixelRatio ?? 1;
      const ratio = Math.max(1, Math.min(4, Math.ceil(imageryRatio)));
      const outputSize = size * ratio;
      const canvas = document.createElement("canvas");
      canvas.width = outputSize + ratio; canvas.height = outputSize + ratio;
      const context = canvas.getContext("2d");
      if (!context) throw new Error("Could not prepare the terrain texture.");
      context.fillStyle = "#a9998e";
      context.fillRect(0, 0, canvas.width, canvas.height);
      entries.forEach(({ layer, images: [main, east, south, corner] }) => {
        context.globalAlpha = layer === "elevation" && layers.imagery ? 0.65 : 1;
        context.imageSmoothingEnabled = layer !== "coverage" && layer !== "estimated" && layer !== "missing";
        if (main) context.drawImage(main, 0, 0, outputSize, outputSize);
        if (east) {
          const inputRatio = east.naturalWidth / size;
          context.drawImage(east, 0, 0, inputRatio, east.naturalHeight, outputSize, 0, ratio, outputSize);
        }
        if (south) {
          const inputRatio = south.naturalHeight / size;
          context.drawImage(south, 0, 0, south.naturalWidth, inputRatio, 0, outputSize, outputSize, ratio);
        }
        if (corner) context.drawImage(corner, 0, 0, corner.naturalWidth / size, corner.naturalHeight / size,
          outputSize, outputSize, ratio, ratio);
        if (layer === "imagery" && colorizeImagery) {
          // CTX is monochrome. A soft-light tint keeps its shadows and surface
          // detail visible; later data layers retain their original colors.
          context.globalAlpha = 1;
          context.globalCompositeOperation = "soft-light";
          context.fillStyle = "#ee9d68";
          context.fillRect(0, 0, canvas.width, canvas.height);
          context.globalCompositeOperation = "source-over";
        }
      });
      created = new CanvasTexture(canvas);
      created.colorSpace = SRGBColorSpace;
      if (layers.coverage && (!region.coverageLayers || coverage.estimated || coverage.missing)) {
        // Coverage is categorical; interpolating it can suggest coverage where
        // the source reports unknown terrain.
        created.magFilter = NearestFilter;
        created.minFilter = NearestFilter;
      }
      setMap(created);
    }).catch(() => { if (!cancelled) onError("Some visible map layer tiles could not be loaded."); });
    return () => { cancelled = true; created?.dispose(); };
  }, [region, loaded.z, tile.x, tile.y, layers.imagery, layers.elevation, layers.shade, layers.thermal, layers.slope, layers.coverage, coverage.estimated, coverage.missing, environmentalLayers.shade, environmentalLayers.thermal, colorizeImagery, onError]);

  const onClick = (event: ThreeEvent<MouseEvent>) => {
    if (!activeTool) return;
    // Three's raycaster does not apply material clipping. Ignore backdrop hits
    // inside the clipped area so the detailed terrain can receive the click.
    if (clipPlanes?.every((plane) => plane.distanceToPoint(event.point) < 0)) return;
    event.stopPropagation();
    const [left, bottom, right, top] = region.extent;
    onPlace(activeTool, [event.point.x + (left + right) / 2, (top + bottom) / 2 - event.point.z]);
  };
  return <mesh geometry={geometry} onClick={onClick}>
    {/* A material first rendered without a map needs a new shader when the
        asynchronously loaded texture arrives. Remount it for that transition. */}
    <meshStandardMaterial key={map ? "textured" : "loading"} color={map ? "#ffffff" : "#a9998e"}
      map={map} roughness={1} metalness={0} side={DoubleSide} clippingPlanes={clipPlanes} clipIntersection />
  </mesh>;
}

/** Missing heights have no terrain mesh; mark their footprint on a flat plane. */
function MissingSurface({ region, loaded, tile, clipPlanes }: { region: RegionManifest; loaded: Loaded; tile: Tile; clipPlanes?: Plane[] }) {
  const [texture, setTexture] = useState<Texture | null>(null);
  useEffect(() => {
    let cancelled = false;
    let created: Texture | null = null;
    void loadImage(region, "missing", loaded.z, tile.x, tile.y).then((image) => {
      if (cancelled || !image) return;
      created = new Texture(image);
      created.colorSpace = SRGBColorSpace;
      created.magFilter = NearestFilter;
      created.minFilter = NearestFilter;
      created.needsUpdate = true;
      setTexture(created);
    }).catch(() => { if (!cancelled) setTexture(null); });
    return () => { cancelled = true; created?.dispose(); };
  }, [region, loaded.z, tile.x, tile.y]);
  if (!texture) return null;
  const span = region.resolutions[loaded.z] * region.tileSize;
  const x = region.extent[0] + (tile.x + .5) * span - (region.extent[0] + region.extent[2]) / 2;
  const z = (region.extent[1] + region.extent[3]) / 2 - region.extent[3] + (tile.y + .5) * span;
  return <mesh position={[x, -1, z]} rotation={[-Math.PI / 2, 0, 0]}>
    <planeGeometry args={[span, span]} />
    <meshBasicMaterial map={texture} transparent depthWrite={false} side={DoubleSide} clippingPlanes={clipPlanes} clipIntersection />
  </mesh>;
}

function Markers({ region, loaded, overview, start, destination, stop }: Pick<MapViewProps, "region" | "start" | "destination" | "stop"> & { loaded: Loaded; overview: Loaded | null }) {
  const resolution = region.resolutions[loaded.z];
  const radius = resolution * 2;
  return <>{([
    ["start", start, "#147bdb"], ["destination", destination, "#f15e4a"], ["stop", stop, "#eaa24d"],
  ] as const).map(([name, coordinate, color]) => {
    if (!coordinate) return null;
    const height = heightAt(loaded, region, coordinate) ?? (overview && heightAt(overview, region, coordinate));
    if (height === null) return null;
    const [x, y, z] = toScene(region, coordinate, height);
    return <mesh key={name} position={[x, y + radius, z]}>
      <sphereGeometry args={[radius, 14, 10]} /><meshStandardMaterial color={color} roughness={0.65} />
    </mesh>;
  })}</>;
}

function RouteLine({ region, loaded, overview, route, detailBounds, overviewVisible }: Pick<MapViewProps, "region" | "route"> & {
  loaded: Loaded; overview: Loaded | null; detailBounds: TerrainBounds | null; overviewVisible: boolean;
}) {
  const lines = useMemo(() => {
    const positions: number[] = [];
    const step = region.resolutions[loaded.z] / 2;
    const diagonal = Math.hypot(region.extent[2] - region.extent[0], region.extent[3] - region.extent[1]);
    let last: [number, number, number] | null = null;

    const addPoint = (coordinate: MarsCoordinate) => {
      const [x, , z] = toScene(region, coordinate, 0);
      const inDetail = detailBounds && x >= detailBounds.left && x <= detailBounds.right &&
        z >= detailBounds.north && z <= detailBounds.south;
      // The detailed tile data also contains a halo for interpolation. Outside
      // the rendered detail area, sample the visible overview instead.
      const surface = inDetail ? loaded : overviewVisible ? overview : null;
      const height = surface && heightAt(surface, region, coordinate);
      if (height === null) { last = null; return; }
      const point = toScene(region, coordinate, height + step * 0.2);
      if (last) positions.push(...last, ...point);
      last = point;
    };

    if (route.length) addPoint(route[0]);
    for (let i = 1; i < route.length; i++) {
      const from = route[i - 1], to = route[i];
      const distance = Math.hypot(to[0] - from[0], to[1] - from[1]);
      if (!Number.isFinite(distance) || distance > diagonal * 2) { last = null; continue; }
      if (distance === 0) continue;
      const count = Math.max(1, Math.ceil(distance / step));
      for (let n = 1; n <= count; n++) {
        const t = n / count;
        const coordinate: MarsCoordinate = [from[0] + (to[0] - from[0]) * t, from[1] + (to[1] - from[1]) * t];
        addPoint(coordinate);
      }
    }
    if (!positions.length) return null;
    const geometry = new LineSegmentsGeometry().setPositions(positions);
    // Screen-space widths stay legible as the camera zooms; the dark casing
    // keeps the cyan route distinct from both light and dark terrain. The
    // depth bias prevents flickering where the line lies close to the surface.
    const materialOptions = { depthWrite: false, toneMapped: false, polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -1 };
    const outline = new LineSegments2(geometry, new LineMaterial({ ...materialOptions, color: "#0e191f", linewidth: 9 }));
    const center = new LineSegments2(geometry, new LineMaterial({ ...materialOptions, color: "#59dbed", linewidth: 5 }));
    outline.renderOrder = 1;
    center.renderOrder = 2;
    return { geometry, outline, center };
  }, [region, loaded, overview, route, detailBounds, overviewVisible]);
  useEffect(() => () => {
    if (!lines) return;
    lines.geometry.dispose();
    lines.outline.material.dispose();
    lines.center.material.dispose();
  }, [lines]);
  return lines && <><primitive object={lines.outline} /><primitive object={lines.center} /></>;
}

function CameraControls({ region, onSelection, reset, heightLookup }: { region: RegionManifest; onSelection: (selection: Selection) => void; reset: number; heightLookup: HeightLookup }) {
  const { camera, gl, size } = useThree();
  const controlsRef = useRef<OrbitControls | null>(null);
  const lastKey = useRef("");
  const lastZoom = useRef<number | undefined>(undefined);
  const lastTime = useRef(0);
  useEffect(() => {
    const controls = new OrbitControls(camera, gl.domElement);
    controlsRef.current = controls;
    const span = Math.max(region.extent[2] - region.extent[0], region.extent[3] - region.extent[1]);
    const range = Math.max(0, region.elevationRange[1] - region.elevationRange[0]);
    camera.position.set(span * 0.1, span * 0.85 + range / 2, span * 0.9);
    camera.near = Math.max(0.1, region.resolutions[region.maxZoom] / 10);
    camera.far = span * 20 + range;
    camera.updateProjectionMatrix();
    controls.target.set(0, range / 2, 0);
    controls.enableDamping = true;
    controls.minDistance = region.resolutions[region.maxZoom] * 4;
    controls.maxDistance = span * 5;
    controls.maxPolarAngle = Math.PI / 2 - 0.05;
    controls.listenToKeyEvents(gl.domElement);
    controls.update();
    lastKey.current = "";
    lastZoom.current = undefined;
    return () => { controlsRef.current = null; controls.dispose(); };
  }, [camera, gl, region, reset]);
  useFrame((state) => {
    const controls = controlsRef.current;
    if (!controls) return;
    controls.update();
    // Keep the zoom target on the local terrain while panning. Otherwise a
    // full-region initial target could sit kilometres above or below a later
    // close-up, preventing a useful native-resolution view.
    const coordinate: MarsCoordinate = [
      controls.target.x + (region.extent[0] + region.extent[2]) / 2,
      (region.extent[1] + region.extent[3]) / 2 - controls.target.z,
    ];
    const elevation = heightLookup.current?.(coordinate);
    if (elevation !== undefined && elevation !== null) {
      const delta = (elevation - region.elevationRange[0] - controls.target.y) * 0.15;
      controls.target.y += delta;
      camera.position.y += delta;
      controls.update();
    }
    if (state.clock.elapsedTime - lastTime.current < 0.15) return;
    lastTime.current = state.clock.elapsedTime;
    const selection = pickTiles(region, camera, controls.target, size.height, lastZoom.current);
    lastZoom.current = selection.z;
    if (selection.key !== lastKey.current) {
      lastKey.current = selection.key;
      onSelection(selection);
    }
  });
  return null;
}

function TerrainTiles(props: Map3DProps & { selection: Selection | null; onStatus: (status: LodStatus) => void; heightLookup: HeightLookup }) {
  const { region, selection, onStatus } = props;
  const [loaded, setLoaded] = useState<Loaded | null>(null);
  const [overview, setOverview] = useState<Loaded | null>(null);
  const [imageError, setImageError] = useState<string | null>(null);
  const onError = useCallback((message: string) => setImageError(message), []);
  useEffect(() => {
    let cancelled = false;
    const z = overviewZoom(region);
    setOverview(null);
    // A custom region may not provide a sufficiently coarse overview level.
    if (z === null) return () => { cancelled = true; };
    const span = region.tileSize * region.resolutions[z];
    const cols = Math.ceil((region.extent[2] - region.extent[0]) / span);
    const rows = Math.ceil((region.extent[3] - region.extent[1]) / span);
    const tiles: Tile[] = [];
    for (let y = 0; y < rows; y++) for (let x = 0; x < cols; x++) tiles.push({ x, y });
    Promise.all(tiles.map(async (tile) => [tileKey(tile.x, tile.y), await loadTerrain(region, z, tile.x, tile.y)] as const))
      .then((entries) => { if (!cancelled) setOverview({ z, tiles, data: new Map(entries) }); })
      .catch(() => { if (!cancelled) onError("Overview terrain tiles could not be loaded."); });
    return () => { cancelled = true; };
  }, [region, onError]);
  useEffect(() => {
    if (!selection) return;
    let cancelled = false;
    setImageError(null);
    onStatus({ zoom: loaded?.z ?? selection.z, loading: true, error: null });
    const required = new Map<string, Tile>();
    // A one-tile halo supplies the neighbouring pixel centres required for
    // seamless geometry, smooth normals, and route interpolation.
    selection.tiles.forEach((tile) => {
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const adjacent = { x: tile.x + dx, y: tile.y + dy };
        if (inRegion(region, selection.z, adjacent.x, adjacent.y)) required.set(tileKey(adjacent.x, adjacent.y), adjacent);
      }
    });
    Promise.all([...required.entries()].map(async ([key, tile]) => [key, await loadTerrain(region, selection.z, tile.x, tile.y)] as const))
      .then((entries) => {
        if (cancelled) return;
        setLoaded({ z: selection.z, tiles: selection.tiles, data: new Map(entries) });
        onStatus({ zoom: selection.z, loading: false, error: null });
      })
      .catch(() => { if (!cancelled) onStatus({ zoom: loaded?.z ?? selection.z, loading: false, error: "Terrain tiles could not be loaded." }); });
    return () => { cancelled = true; };
    // Changing loaded while a request runs must not restart the request.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [region, selection, onStatus]);
  useEffect(() => {
    if (imageError) onStatus({ zoom: loaded?.z ?? region.minZoom, loading: false, error: imageError });
  }, [imageError, loaded, onStatus, region.minZoom]);
  useEffect(() => {
    props.heightLookup.current = loaded
      ? (coordinate) => heightAt(loaded, region, coordinate) ?? (overview && heightAt(overview, region, coordinate))
      : null;
    return () => { props.heightLookup.current = null; };
  }, [loaded, overview, region, props.heightLookup]);
  const detailBounds = useMemo(() => {
    if (!loaded?.tiles.length) return null;
    const resolution = region.resolutions[loaded.z];
    const span = region.tileSize * resolution;
    const minX = Math.min(...loaded.tiles.map((tile) => tile.x));
    const maxX = Math.max(...loaded.tiles.map((tile) => tile.x));
    const minY = Math.min(...loaded.tiles.map((tile) => tile.y));
    const maxY = Math.max(...loaded.tiles.map((tile) => tile.y));
    // The geometry begins at pixel centers, half a cell inside the tile grid.
    const left = region.extent[0] - (region.extent[0] + region.extent[2]) / 2 + minX * span + resolution / 2;
    const right = left + (maxX - minX + 1) * span;
    const north = (region.extent[1] + region.extent[3]) / 2 - region.extent[3] + minY * span + resolution / 2;
    const south = north + (maxY - minY + 1) * span;
    return { left, right, north, south };
  }, [loaded, region]);
  const clipPlanes = useMemo(() => {
    if (!loaded || !overview || loaded.z <= overview.z || !detailBounds) return undefined;
    const { left, right, north, south } = detailBounds;
    // With clipIntersection, the backdrop is discarded only inside all four
    // planes, leaving exactly one terrain surface at each location.
    return [new Plane(new Vector3(-1, 0, 0), left), new Plane(new Vector3(1, 0, 0), -right),
      new Plane(new Vector3(0, 0, -1), north), new Plane(new Vector3(0, 0, 1), -south)];
  }, [loaded, overview, detailBounds]);
  if (!loaded) return null;
  return <>
    {overview && clipPlanes && overview.tiles.map((tile) => <TileSurface key={`overview/${overview.z}/${tile.x}/${tile.y}`}
      region={region} loaded={overview} tile={tile} layers={props.layers} coverage={props.coverage} environmentalLayers={props.environmentalLayers} activeTool={props.activeTool} onPlace={props.onPlace}
      onError={onError} colorizeImagery={props.colorizeImagery} clipPlanes={clipPlanes} />)}
    {loaded.tiles.map((tile) => <TileSurface key={`${loaded.z}/${tile.x}/${tile.y}`} region={region} loaded={loaded}
      tile={tile} layers={props.layers} coverage={props.coverage} environmentalLayers={props.environmentalLayers} activeTool={props.activeTool} onPlace={props.onPlace} onError={onError}
      colorizeImagery={props.colorizeImagery} stitchOverview={clipPlanes ? overview : null} />)}
    {props.layers.coverage && props.coverage.missing && region.coverageLayers && <>
      {overview && clipPlanes && overview.tiles.map((tile) => <MissingSurface key={`missing-overview/${overview.z}/${tile.x}/${tile.y}`} region={region} loaded={overview} tile={tile} clipPlanes={clipPlanes} />)}
      {loaded.tiles.map((tile) => <MissingSurface key={`missing/${loaded.z}/${tile.x}/${tile.y}`} region={region} loaded={loaded} tile={tile} />)}
    </>}
    <Markers region={region} loaded={loaded} overview={overview} start={props.start} destination={props.destination} stop={props.stop} />
    <RouteLine region={region} loaded={loaded} overview={overview} route={props.route}
      detailBounds={detailBounds} overviewVisible={!!clipPlanes} />
  </>;
}

function Scene(props: Map3DProps & { reset: number; onStatus: (status: LodStatus) => void }) {
  const [selection, setSelection] = useState<Selection | null>(null);
  const heightLookup = useRef<((coordinate: MarsCoordinate) => number | null) | null>(null);
  const span = Math.max(props.region.extent[2] - props.region.extent[0], props.region.extent[3] - props.region.extent[1]);
  return <>
    <color attach="background" args={["#e3d8cf"]} />
    {/* Keep orbital imagery legible while adding enough directional light to
        reveal terrain relief; the brighter defaults clipped light CTX pixels. */}
    <ambientLight intensity={0.5} />
    <directionalLight position={[-span * 0.4, span, -span * 0.5]} intensity={0.65} />
    <CameraControls region={props.region} onSelection={setSelection} reset={props.reset} heightLookup={heightLookup} />
    <TerrainTiles {...props} selection={selection} heightLookup={heightLookup} />
  </>;
}

export default function Map3D(props: Map3DProps) {
  const [reset, setReset] = useState(0);
  const [status, setStatus] = useState<LodStatus>({ zoom: props.region.minZoom, loading: true, error: null });
  const resolution = props.region.resolutions[status.zoom];
  const overviewLevel = overviewZoom(props.region);
  return <div style={{ width: "100%", height: "100%", position: "relative" }}>
    <Canvas tabIndex={0} aria-label="Interactive 3D terrain map. Drag to orbit, right drag or arrow keys to pan, and scroll to zoom."
      camera={{ fov: 45 }} dpr={[1, 1.5]} onCreated={({ gl }) => { gl.localClippingEnabled = true; }}
      fallback={<p>This device cannot display the 3D terrain. Please use the 2D map.</p>}>
      <Scene {...props} reset={reset} onStatus={setStatus} />
    </Canvas>
    <div className="terrain-lod-status" role="status" style={{ padding: "7px 10px", borderRadius: 10,
      background: "rgba(255,255,255,.92)", color: "#453d39", fontSize: 12, maxWidth: "70%", boxShadow: "0 2px 12px #382a221a" }}>
      {status.error || (status.loading ? "Loading terrain tiles…" : `${status.zoom === props.region.maxZoom ? "Native" : "Coarse"} terrain grid · ${resolution.toFixed(0)} m/pixel${overviewLevel !== null && status.zoom > overviewLevel ? " near center · distant terrain coarser" : ""}${status.zoom < props.region.maxZoom ? " · zoom in for detail" : ""}`)}
    </div>
    <button className="terrain-reset" type="button" onClick={() => setReset((value) => value + 1)} style={{
      border: 0, borderRadius: 10, padding: "8px 12px", background: "white", color: "#453d39", cursor: "pointer",
      boxShadow: "0 2px 12px #382a221a", fontSize: 12 }}>Reset 3D view</button>
  </div>;
}
