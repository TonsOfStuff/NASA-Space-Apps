import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree, type ThreeEvent } from "@react-three/fiber";
import {
  BufferAttribute,
  BufferGeometry,
  CanvasTexture,
  DoubleSide,
  LineBasicMaterial,
  LineSegments,
  NearestFilter,
  Plane,
  Raycaster,
  SRGBColorSpace,
  Vector2,
  Vector3,
  type Camera,
  type PerspectiveCamera,
} from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { LayerId, MapViewProps, MarsCoordinate, RegionManifest } from "../types";

type Tile = { x: number; y: number };
type Selection = { z: number; tiles: Tile[]; key: string };
type Loaded = { z: number; tiles: Tile[]; data: Map<string, Float32Array | null> };
type LodStatus = { zoom: number; loading: boolean; error: string | null };
type HeightLookup = { current: ((coordinate: MarsCoordinate) => number | null) | null };

const layerOrder: LayerId[] = ["imagery", "elevation", "slope", "coverage"];
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

function loadImage(region: RegionManifest, layer: LayerId, z: number, x: number, y: number): Promise<HTMLImageElement | null> {
  if (!inRegion(region, z, x, y)) return Promise.resolve(null);
  const url = tileUrl(region.layers[layer], z, x, y);
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

function makeGeometry(region: RegionManifest, loaded: Loaded, tile: Tile): BufferGeometry {
  const size = region.tileSize;
  const side = size + 1;
  const resolution = region.resolutions[loaded.z];
  const positions = new Float32Array(side * side * 3);
  const uvs = new Float32Array(side * side * 2);
  const normals = new Float32Array(side * side * 3);
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
  const geometry = new BufferGeometry();
  geometry.setAttribute("position", new BufferAttribute(positions, 3));
  geometry.setAttribute("uv", new BufferAttribute(uvs, 2));
  geometry.setAttribute("normal", new BufferAttribute(normals, 3));
  geometry.setIndex(indices);
  return geometry;
}

function TileSurface({ region, loaded, tile, layers, activeTool, onPlace, onError }: Pick<MapViewProps, "region" | "layers" | "activeTool" | "onPlace"> & {
  loaded: Loaded; tile: Tile; onError: (message: string) => void;
}) {
  const geometry = useMemo(() => makeGeometry(region, loaded, tile), [region, loaded, tile]);
  const [map, setMap] = useState<CanvasTexture | null>(null);
  useEffect(() => () => geometry.dispose(), [geometry]);

  useEffect(() => {
    let cancelled = false;
    let created: CanvasTexture | null = null;
    const enabled = layerOrder.filter((layer) => layers[layer]);
    Promise.all(enabled.map(async (layer) => ({
      layer,
      images: await Promise.all(neighbours(tile).map(({ x, y }) => loadImage(region, layer, loaded.z, x, y))),
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
        context.imageSmoothingEnabled = layer !== "coverage";
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
      });
      created = new CanvasTexture(canvas);
      created.colorSpace = SRGBColorSpace;
      if (layers.coverage) {
        // Coverage is categorical; interpolating it can suggest coverage where
        // the source reports unknown terrain.
        created.magFilter = NearestFilter;
        created.minFilter = NearestFilter;
      }
      setMap(created);
    }).catch(() => { if (!cancelled) onError("Some visible map layer tiles could not be loaded."); });
    return () => { cancelled = true; created?.dispose(); };
  }, [region, loaded.z, tile, layers.imagery, layers.elevation, layers.slope, layers.coverage, onError]);

  const onClick = (event: ThreeEvent<MouseEvent>) => {
    if (!activeTool) return;
    event.stopPropagation();
    const [left, bottom, right, top] = region.extent;
    onPlace(activeTool, [event.point.x + (left + right) / 2, (top + bottom) / 2 - event.point.z]);
  };
  return <mesh geometry={geometry} onClick={onClick}>
    <meshStandardMaterial color={map ? "#ffffff" : "#a9998e"} map={map} roughness={1} metalness={0} side={DoubleSide} />
  </mesh>;
}

function Markers({ region, loaded, start, destination, stop }: Pick<MapViewProps, "region" | "start" | "destination" | "stop"> & { loaded: Loaded }) {
  const resolution = region.resolutions[loaded.z];
  const radius = resolution * 2;
  return <>{([
    ["start", start, "#147bdb"], ["destination", destination, "#f15e4a"], ["stop", stop, "#eaa24d"],
  ] as const).map(([name, coordinate, color]) => {
    if (!coordinate) return null;
    const height = heightAt(loaded, region, coordinate);
    if (height === null) return null;
    const [x, y, z] = toScene(region, coordinate, height);
    return <mesh key={name} position={[x, y + radius, z]}>
      <sphereGeometry args={[radius, 14, 10]} /><meshStandardMaterial color={color} roughness={0.65} />
    </mesh>;
  })}</>;
}

function RouteLine({ region, loaded, route }: Pick<MapViewProps, "region" | "route"> & { loaded: Loaded }) {
  const line = useMemo(() => {
    const positions: number[] = [];
    const step = region.resolutions[loaded.z] / 2;
    const diagonal = Math.hypot(region.extent[2] - region.extent[0], region.extent[3] - region.extent[1]);
    let last: [number, number, number] | null = null;
    for (let i = 0; i < route.length; i++) {
      const from = route[Math.max(0, i - 1)], to = route[i];
      const distance = Math.hypot(to[0] - from[0], to[1] - from[1]);
      if (!Number.isFinite(distance) || distance > diagonal * 2) { last = null; continue; }
      const count = Math.max(1, Math.ceil(distance / step));
      for (let n = i === 0 ? 0 : 1; n <= count; n++) {
        const t = n / count;
        const coordinate: MarsCoordinate = [from[0] + (to[0] - from[0]) * t, from[1] + (to[1] - from[1]) * t];
        const height = heightAt(loaded, region, coordinate);
        if (height === null) { last = null; continue; }
        const point = toScene(region, coordinate, height + step * 0.2);
        if (last) positions.push(...last, ...point);
        last = point;
      }
    }
    const geometry = new BufferGeometry();
    geometry.setAttribute("position", new BufferAttribute(new Float32Array(positions), 3));
    return new LineSegments(geometry, new LineBasicMaterial({ color: "#00d9ec" }));
  }, [region, loaded, route]);
  useEffect(() => () => { line.geometry.dispose(); (line.material as LineBasicMaterial).dispose(); }, [line]);
  return <primitive object={line} />;
}

function pickTiles(region: RegionManifest, camera: Camera, target: Vector3, heightPx: number): Selection {
  const perspective = camera as PerspectiveCamera;
  const distance = camera.position.distanceTo(target);
  const groundPerPixel = 2 * distance * Math.tan((perspective.fov || 45) * Math.PI / 360) / Math.max(heightPx, 1);
  let z = Math.max(region.minZoom, Math.min(region.maxZoom,
    Math.round(Math.log2(region.resolutions[region.minZoom] / Math.max(groundPerPixel, 0.001)))));
  const raycaster = new Raycaster();
  const plane = new Plane(new Vector3(0, 1, 0), -target.y);
  const points: Vector3[] = [];
  camera.updateMatrixWorld();
  for (const x of [-1, 1]) for (const y of [-1, 1]) {
    raycaster.setFromCamera(new Vector2(x, y), camera);
    const hit = raycaster.ray.intersectPlane(plane, new Vector3());
    if (hit) points.push(hit);
  }
  points.push(target.clone());
  if (points.length < 5) {
    points.push(new Vector3(target.x - distance * 2, target.y, target.z - distance * 2));
    points.push(new Vector3(target.x + distance * 2, target.y, target.z + distance * 2));
  }
  const centerX = (region.extent[0] + region.extent[2]) / 2;
  const centerY = (region.extent[1] + region.extent[3]) / 2;
  const minX = Math.min(...points.map((p) => p.x)) + centerX;
  const maxX = Math.max(...points.map((p) => p.x)) + centerX;
  const north = centerY - Math.min(...points.map((p) => p.z));
  const south = centerY - Math.max(...points.map((p) => p.z));
  const tilesAt = (level: number): Tile[] => {
    const span = region.tileSize * region.resolutions[level];
    const cols = Math.ceil((region.extent[2] - region.extent[0]) / span);
    const rows = Math.ceil((region.extent[3] - region.extent[1]) / span);
    const x0 = Math.max(0, Math.floor((minX - region.extent[0]) / span) - 1);
    const x1 = Math.min(cols - 1, Math.floor((maxX - region.extent[0]) / span) + 1);
    const y0 = Math.max(0, Math.floor((region.extent[3] - north) / span) - 1);
    const y1 = Math.min(rows - 1, Math.floor((region.extent[3] - south) / span) + 1);
    const result: Tile[] = [];
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) result.push({ x, y });
    return result;
  };
  let tiles = tilesAt(z);
  while (tiles.length > 16 && z > region.minZoom) tiles = tilesAt(--z);
  return { z, tiles, key: `${z}:${tiles.map(({ x, y }) => `${x},${y}`).join(";")}` };
}

function CameraControls({ region, onSelection, reset, heightLookup }: { region: RegionManifest; onSelection: (selection: Selection) => void; reset: number; heightLookup: HeightLookup }) {
  const { camera, gl, size } = useThree();
  const controlsRef = useRef<OrbitControls | null>(null);
  const lastKey = useRef("");
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
    const selection = pickTiles(region, camera, controls.target, size.height);
    if (selection.key !== lastKey.current) {
      lastKey.current = selection.key;
      onSelection(selection);
    }
  });
  return null;
}

function TerrainTiles(props: MapViewProps & { selection: Selection | null; onStatus: (status: LodStatus) => void; heightLookup: HeightLookup }) {
  const { region, selection, onStatus } = props;
  const [loaded, setLoaded] = useState<Loaded | null>(null);
  const [imageError, setImageError] = useState<string | null>(null);
  const onError = useCallback((message: string) => setImageError(message), []);
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
    props.heightLookup.current = loaded ? (coordinate) => heightAt(loaded, region, coordinate) : null;
    return () => { props.heightLookup.current = null; };
  }, [loaded, region, props.heightLookup]);
  if (!loaded) return null;
  return <>
    {loaded.tiles.map((tile) => <TileSurface key={`${loaded.z}/${tile.x}/${tile.y}`} region={region} loaded={loaded}
      tile={tile} layers={props.layers} activeTool={props.activeTool} onPlace={props.onPlace} onError={onError} />)}
    <Markers region={region} loaded={loaded} start={props.start} destination={props.destination} stop={props.stop} />
    <RouteLine region={region} loaded={loaded} route={props.route} />
  </>;
}

function Scene(props: MapViewProps & { reset: number; onStatus: (status: LodStatus) => void }) {
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

export default function Map3D(props: MapViewProps) {
  const [reset, setReset] = useState(0);
  const [status, setStatus] = useState<LodStatus>({ zoom: props.region.minZoom, loading: true, error: null });
  const resolution = props.region.resolutions[status.zoom];
  return <div style={{ width: "100%", height: "100%", position: "relative" }}>
    <Canvas tabIndex={0} aria-label="Interactive 3D terrain map. Drag to orbit, right drag or arrow keys to pan, and scroll to zoom."
      camera={{ fov: 45 }} dpr={[1, 1.5]} fallback={<p>This device cannot display the 3D terrain. Please use the 2D map.</p>}>
      <Scene {...props} reset={reset} onStatus={setStatus} />
    </Canvas>
    <div className="terrain-lod-status" role="status" style={{ padding: "7px 10px", borderRadius: 10,
      background: "rgba(255,255,255,.92)", color: "#453d39", fontSize: 12, maxWidth: "70%", boxShadow: "0 2px 12px #382a221a" }}>
      {status.error || (status.loading ? "Loading terrain tiles…" : `${status.zoom === props.region.maxZoom ? "Native" : "Coarse"} terrain grid · ${resolution.toFixed(0)} m/pixel${status.zoom < props.region.maxZoom ? " · zoom in for detail" : ""}`)}
    </div>
    <button className="terrain-reset" type="button" onClick={() => setReset((value) => value + 1)} style={{
      border: 0, borderRadius: 10, padding: "8px 12px", background: "white", color: "#453d39", cursor: "pointer",
      boxShadow: "0 2px 12px #382a221a", fontSize: 12 }}>Reset 3D view</button>
  </div>;
}
