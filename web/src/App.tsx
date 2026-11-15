import { Fragment, lazy, Suspense, useCallback, useEffect, useRef, useState } from "react";
import { ArrowRight, Box, ChevronDown, CircleHelp, Clock3, Compass, Layers3, LoaderCircle, Map as MapIcon, MapPin, Navigation, RotateCcw, Route, SlidersHorizontal, TriangleAlert, X } from "lucide-react";
import Map2D from "./components/Map2D";
import { isRouteServiceConfigured, requestRoute, type RouteResult } from "./routeService";
import type { CoverageVisibility, LayerId, LayerVisibility, MarkerKind, MarsCoordinate, RegionManifest } from "./types";

const Map3D = lazy(() => import("./components/Map3D"));
const manifestUrl = (import.meta.env.VITE_REGION_MANIFEST_URL as string | undefined)?.trim() || "/data/gale/manifest.json";
const initialLayers: LayerVisibility = { imagery: true, elevation: false, slope: false, coverage: true };
const layerDescriptions: { id: LayerId; label: string; description: string }[] = [
  { id: "imagery", label: "Surface imagery", description: "CTX orbital mosaic" },
  { id: "elevation", label: "Elevation", description: "Relative terrain height" },
  { id: "slope", label: "Slope", description: "Approximate · 0–30°+" },
  { id: "coverage", label: "Data coverage", description: "Missing & estimated elevation" },
];

type LoadedRegion = { manifest: RegionManifest };
const terrainTileCache = new Map<string, ArrayBuffer>();

function isFinitePoint(point: MarsCoordinate): boolean {
  return point.every(Number.isFinite);
}

async function hasTerrainAt(point: MarsCoordinate, { manifest }: LoadedRegion): Promise<boolean> {
  const [left, bottom, right, top] = manifest.extent;
  if (!isFinitePoint(point) || point[0] < left || point[0] > right || point[1] < bottom || point[1] > top) return false;
  const col = Math.min(manifest.width - 1, Math.floor(((point[0] - left) / (right - left)) * manifest.width));
  const row = Math.min(manifest.height - 1, Math.floor(((top - point[1]) / (top - bottom)) * manifest.height));
  const tileX = Math.floor(col / manifest.tileSize);
  const tileY = Math.floor(row / manifest.tileSize);
  const url = manifest.terrain.replace("{z}", String(manifest.maxZoom)).replace("{x}", String(tileX)).replace("{y}", String(tileY));
  let buffer = terrainTileCache.get(url);
  if (!buffer) {
    const response = await fetch(url);
    if (!response.ok) throw new Error("Could not load elevation coverage for this location.");
    buffer = await response.arrayBuffer();
    if (buffer.byteLength !== manifest.tileSize * manifest.tileSize * 4) throw new Error("The elevation tile has an unexpected size.");
    if (terrainTileCache.size >= 16) terrainTileCache.delete(terrainTileCache.keys().next().value!);
    terrainTileCache.set(url, buffer);
  }
  const offset = ((row % manifest.tileSize) * manifest.tileSize + col % manifest.tileSize) * 4;
  return Number.isFinite(new DataView(buffer).getFloat32(offset, true));
}

function displayCoordinate(point: MarsCoordinate | null): string {
  return point ? `E ${Math.round(point[0]).toLocaleString()}  ·  N ${Math.round(point[1]).toLocaleString()} m` : "Choose on map";
}

function formatDistance(distance: number): string {
  return distance < 1000 ? `${Math.round(distance)} m` : `${(distance / 1000).toFixed(2)} km`;
}

function movedDescription(label: string, requested: MarsCoordinate | null, evaluated: MarsCoordinate | null): string | null {
  if (!requested || !evaluated) return null;
  const distance = Math.hypot(evaluated[0] - requested[0], evaluated[1] - requested[1]);
  return distance > 0.1 ? `${label} evaluated ${distance.toFixed(1)} projected m from your selection.` : null;
}

export default function App() {
  const [data, setData] = useState<LoadedRegion | null>(null);
  const [dataError, setDataError] = useState<string | null>(null);
  const [layers, setLayers] = useState<LayerVisibility>(initialLayers);
  const [coverage, setCoverage] = useState<CoverageVisibility>({ estimated: true, missing: true });
  const [colorizeImagery, setColorizeImagery] = useState(true);
  const [showLayers, setShowLayers] = useState(false);
  const [view, setView] = useState<"2d" | "3d">("2d");
  const [start, setStart] = useState<MarsCoordinate | null>(null);
  const [destination, setDestination] = useState<MarsCoordinate | null>(null);
  const [activeTool, setActiveTool] = useState<MarkerKind | null>(null);
  const [slope, setSlope] = useState(15);
  const [marsSol, setMarsSol] = useState("");
  const [marsLocalTime, setMarsLocalTime] = useState("");
  const [routing, setRouting] = useState(false);
  const [result, setResult] = useState<RouteResult | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const requestRef = useRef<AbortController | null>(null);
  const placementRef = useRef(0);
  const mapStageRef = useRef<HTMLElement>(null);
  const helpRef = useRef<HTMLDialogElement>(null);
  const helpButtonRef = useRef<HTMLButtonElement>(null);
  const hasMarsDateTime = marsSol !== "" || marsLocalTime !== "";
  const validMarsDateTime = marsSol !== "" && Number.isSafeInteger(Number(marsSol)) && Number(marsSol) >= 0 &&
    /^(?:[01]\d|2[0-3]):[0-5]\d$/.test(marsLocalTime);

  useEffect(() => {
    const controller = new AbortController();
    async function load() {
      try {
        const response = await fetch(manifestUrl, { signal: controller.signal });
        if (!response.ok) throw new Error("The region manifest could not be loaded.");
        const manifest = await response.json() as RegionManifest;
        if (!manifest.id || !Array.isArray(manifest.extent) || manifest.extent.length !== 4 ||
            !Number.isInteger(manifest.width) || !Number.isInteger(manifest.height) ||
            manifest.width <= 0 || manifest.height <= 0 || !manifest.terrain || !manifest.layers ||
            !Number.isInteger(manifest.tileSize) || manifest.tileSize <= 0 ||
            !Number.isInteger(manifest.maxZoom) || !Array.isArray(manifest.resolutions) ||
            manifest.resolutions.length <= manifest.maxZoom) {
          throw new Error("The region manifest is incomplete.");
        }
        setData({ manifest });
      } catch (error) {
        if (!controller.signal.aborted) setDataError(error instanceof Error ? error.message : "The map data could not be loaded.");
      }
    }
    void load();
    return () => controller.abort();
  }, []);

  const clearRoute = useCallback(() => {
    requestRef.current?.abort();
    requestRef.current = null;
    setRouting(false);
    setResult(null);
    setMessage(null);
  }, []);

  useEffect(() => () => requestRef.current?.abort(), []);

  const onPlace = useCallback((kind: MarkerKind, point: MarsCoordinate) => {
    if (!data) return;
    const placement = ++placementRef.current;
    void (async () => {
      try {
        const covered = await hasTerrainAt(point, data);
        if (placement !== placementRef.current) return;
        if (!covered) {
          setMessage("There is no elevation data at that point. Choose a location within the mapped area.");
          return;
        }
        clearRoute();
        if (kind === "start") {
          setStart(point);
          setActiveTool(destination ? null : "destination");
        } else {
          setDestination(point);
          setActiveTool(start ? null : "start");
        }
      } catch (error) {
        if (placement === placementRef.current) setMessage(error instanceof Error ? error.message : "Could not check terrain coverage.");
      }
    })();
  }, [clearRoute, data, destination, start]);

  const calculate = async () => {
    if (!data || !start || !destination || !isRouteServiceConfigured || (hasMarsDateTime && !validMarsDateTime)) return;
    clearRoute();
    const controller = new AbortController();
    requestRef.current = controller;
    setRouting(true);
    try {
      const route = await requestRoute({
        regionId: data.manifest.id,
        start,
        destination,
        rover: { maxSlopeDeg: slope },
        ...(validMarsDateTime ? { marsDateTime: { sol: Number(marsSol), localSolarTime: marsLocalTime, solReference: "mission" as const } } : {}),
      }, data.manifest, controller.signal);
      if (!controller.signal.aborted) setResult(route);
    } catch (error) {
      if (!controller.signal.aborted) setMessage(error instanceof Error ? error.message : "The route could not be calculated.");
    } finally {
      if (requestRef.current === controller) {
        requestRef.current = null;
        setRouting(false);
      }
    }
  };

  const reset = () => {
    placementRef.current++;
    clearRoute();
    setStart(null);
    setDestination(null);
    setMarsSol("");
    setMarsLocalTime("");
    setActiveTool(null);
  };

  const selectTool = (kind: MarkerKind) => {
    placementRef.current++;
    setMessage(null);
    const next = activeTool === kind ? null : kind;
    setActiveTool(next);
    if (next && window.matchMedia("(max-width: 700px)").matches) {
      mapStageRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  };

  const cancelPlacement = () => {
    placementRef.current++;
    setActiveTool(null);
  };

  const updateSlope = (value: number) => {
    clearRoute();
    setSlope(value);
  };

  const mapProps = data ? {
    region: data.manifest,
    layers,
    coverage,
    start,
    destination,
    stop: result?.stop ?? null,
    route: result?.path ?? [],
    activeTool,
    onPlace,
  } : null;
  const shifts = result ? [
    movedDescription("Start", start, result.evaluatedStart),
    movedDescription("Destination", destination, result.evaluatedDestination),
  ].filter((entry): entry is string => entry !== null) : [];

  return (
    <div className="app-shell">
      <main ref={mapStageRef} className="map-stage" aria-label="Mars exploration map">
        {mapProps ? view === "2d" ? <Map2D {...mapProps} /> : (
          <Suspense fallback={<div className="map-loading"><LoaderCircle className="spin" /> Preparing terrain view…</div>}>
            <Map3D {...mapProps} colorizeImagery={colorizeImagery} />
          </Suspense>
        ) : (
          <div className="map-loading" role="status">
            {dataError ? <><TriangleAlert size={20} /><span>{dataError}<br />Check the region data and reload.</span></> : <><LoaderCircle className="spin" /> Loading Mars terrain…</>}
          </div>
        )}

        <div className="top-map-controls">
          <div className="mode-switch" role="group" aria-label="Map view">
            <button className={view === "2d" ? "selected" : ""} aria-pressed={view === "2d"} onClick={() => setView("2d")}><MapIcon size={15} /> 2D</button>
            <button className={view === "3d" ? "selected" : ""} aria-pressed={view === "3d"} onClick={() => setView("3d")}><Box size={15} /> 3D</button>
          </div>
          <div className="layer-anchor">
            <button className={`map-control ${showLayers ? "control-active" : ""}`} aria-label="Map layers" aria-expanded={showLayers} onClick={() => setShowLayers(!showLayers)}><Layers3 size={18} /><span>Layers</span><ChevronDown size={14} /></button>
            {showLayers && (
              <div className="layers-card">
                <div className="layers-heading"><div><span className="eyebrow">EXPLORE THE DATA</span><h2>Map layers</h2></div><button className="icon-button" aria-label="Close layers" onClick={() => setShowLayers(false)}><X size={16} /></button></div>
                {layerDescriptions.map((layer) => (
                  <Fragment key={layer.id}>
                    <label className="layer-row">
                      <span className={`layer-swatch swatch-${layer.id}`} aria-hidden="true" />
                      <span className="layer-name"><strong>{layer.label}</strong><small>{layer.description}</small></span>
                      <input type="checkbox" checked={layers[layer.id]} onChange={(event) => setLayers((previous) => ({ ...previous, [layer.id]: event.target.checked }))} />
                    </label>
                    {layer.id === "imagery" && view === "3d" && layers.imagery && (
                      <label className="imagery-tint-option">
                        <span><strong>Warm tint</strong><small>Display color, not natural color</small></span>
                        <input type="checkbox" checked={colorizeImagery} onChange={(event) => setColorizeImagery(event.target.checked)} />
                      </label>
                    )}
                    {layer.id === "coverage" && (
                      <div className="coverage-options" aria-label="Data coverage options">
                        {(["estimated", "missing"] as const).map((category) => (
                          <label className="coverage-option" key={category}>
                            <i className={`coverage-key coverage-${category}`} aria-hidden="true" />
                            <span>{category === "estimated" ? "Estimated" : "Missing"}</span>
                            <input type="checkbox" checked={coverage[category]} disabled={!layers.coverage || !data?.manifest.coverageLayers}
                              onChange={(event) => setCoverage((previous) => ({ ...previous, [category]: event.target.checked }))} />
                          </label>
                        ))}
                        {data && !data.manifest.coverageLayers && <small>Separate coverage masks are unavailable for this region.</small>}
                      </div>
                    )}
                  </Fragment>
                ))}
                {layers.slope && <div className="slope-legend"><span>Approximate slope</span><div className="slope-ramp" /><div className="legend-ticks"><span>0°</span><span>10°</span><span>20°</span><span>30°+</span></div></div>}
              </div>
            )}
          </div>
        </div>

        {activeTool && <div className="placement-hint"><MapPin size={15} /> Select {activeTool === "start" ? "your starting point" : "a science site"} on the map <button onClick={cancelPlacement}>Cancel</button></div>}
        <div className="map-attribution"><span className="status-dot" /> {data?.manifest.name.toUpperCase() ?? "MARS REGION"} <span className="attribution-divider">/</span> {data ? `${data.manifest.pixelSize[0].toFixed(1)} M SOURCE GRID` : "LOADING"}</div>
        {view === "2d" && <div className="map-compass" aria-label="North"><Navigation size={15} fill="currentColor" /><span>N</span></div>}
      </main>

      <aside className="planner-panel" aria-label="Trip planner">
        <div className="panel-topline">
          <div className="brand"><span className="brand-icon"><Compass size={17} strokeWidth={2.2} /></span><span>MarMap</span></div>
          <button ref={helpButtonRef} className="icon-button help-button" title="Map help" aria-label="Open map help" onClick={() => helpRef.current?.showModal()}><CircleHelp size={19} /></button>
        </div>

        <div className="panel-scroll">
          <div className="region-heading">
            <div><span className="region-overline">MARS</span><h1>{data?.manifest.name ?? "Mars region"}</h1></div>
            <button className="icon-button reset-button" onClick={reset} title="Reset trip" aria-label="Reset trip"><RotateCcw size={17} /></button>
          </div>
          <div className="waypoints">
            <span className="waypoint-line" aria-hidden="true" />
            <button className={`waypoint ${activeTool === "start" ? "waypoint-active" : ""}`} aria-pressed={activeTool === "start"} onClick={() => selectTool("start")}>
              <span className="point-marker start-marker">A</span><span className="waypoint-text"><strong>Start</strong><small>{displayCoordinate(start)}</small></span><MapPin size={16} />
            </button>
            <button className={`waypoint ${activeTool === "destination" ? "waypoint-active" : ""}`} aria-pressed={activeTool === "destination"} onClick={() => selectTool("destination")}>
              <span className="point-marker destination-marker">B</span><span className="waypoint-text"><strong>Science site</strong><small>{displayCoordinate(destination)}</small></span><MapPin size={16} />
            </button>
            <div className="waypoint waypoint-readonly">
              <span className="point-marker parking-marker">P</span><span className="waypoint-text"><strong>Parking{result?.stop ? " · candidate" : ""}</strong><small>{result?.stop ? displayCoordinate(result.stop) : "Determined by route"}</small></span><MapPin size={16} />
            </div>
          </div>

          <div className="settings-block">
            <div className="range-label"><label htmlFor="max-slope"><SlidersHorizontal size={15} /> Max slope</label><strong>{slope}°</strong></div>
            <input id="max-slope" type="range" min="3" max="30" step="1" value={slope} onChange={(event) => updateSlope(Number(event.target.value))} />
            <div className="range-ends"><span>3°</span><span>30°</span></div>
          </div>

          <div className="mars-time-block">
            <div className="mars-time-heading"><Clock3 size={15} /> Mars date &amp; time</div>
            <div className="mars-time-fields">
              <label htmlFor="mars-sol">Mission sol
                <input id="mars-sol" type="number" min="0" step="1" inputMode="numeric" placeholder="Sol" value={marsSol}
                  aria-invalid={hasMarsDateTime && !validMarsDateTime} aria-describedby={hasMarsDateTime && !validMarsDateTime ? "mars-time-error" : undefined}
                  onChange={(event) => { clearRoute(); setMarsSol(event.target.value); }} />
              </label>
              <label htmlFor="mars-time">Local solar time
                <input id="mars-time" type="time" value={marsLocalTime}
                  aria-invalid={hasMarsDateTime && !validMarsDateTime} aria-describedby={hasMarsDateTime && !validMarsDateTime ? "mars-time-error" : undefined}
                  onChange={(event) => { clearRoute(); setMarsLocalTime(event.target.value); }} />
              </label>
            </div>
            {hasMarsDateTime && !validMarsDateTime && <small id="mars-time-error" className="mars-time-error">Enter a valid sol and time.</small>}
          </div>

          <button className="route-button" disabled={!start || !destination || !isRouteServiceConfigured || routing || !data || (hasMarsDateTime && !validMarsDateTime)} onClick={() => void calculate()}>
            {routing ? <><LoaderCircle className="spin" size={17} /> Calculating…</> : <><Route size={17} /> Find route <ArrowRight size={16} /></>}
          </button>

          {!isRouteServiceConfigured && <div className="service-note" role="status"><span className="service-dot" /><span>Routing not connected</span></div>}
          {message && <div className="notice" role="status"><TriangleAlert size={16} /><span>{message}</span><button className="icon-button" aria-label="Dismiss message" onClick={() => setMessage(null)}><X size={14} /></button></div>}
          {result && <div className="result-card" role="status"><div className="result-head"><span className="eyebrow">ROUTE ESTIMATE</span><Route size={15} /></div><h3>{result.status === "reached" ? "Route ready" : result.status === "partial" ? "Partial route" : "No route found"}</h3>{result.distanceM !== null && <p className="route-distance">{formatDistance(result.distanceM)} <small>estimated distance</small></p>}{result.status === "partial" && <p>P marks a candidate stopping point.</p>}{result.status === "unreachable" && <p>The service did not find a route with these inputs.</p>}{shifts.map((shift) => <p key={shift}>{shift}</p>)}{result.warnings.map((warning, index) => <p key={`${index}-${warning}`}>{warning}</p>)}</div>}
        </div>
      </aside>

      <dialog ref={helpRef} className="help-dialog" aria-labelledby="help-title" onClose={() => helpButtonRef.current?.focus()}>
        <div className="help-header"><div><span className="eyebrow">MarMap</span><h2 id="help-title">Map guide</h2></div><button className="icon-button help-close" aria-label="Close map help" onClick={() => helpRef.current?.close()} autoFocus><X size={19} /></button></div>
        <div className="help-items">
          <div className="help-item"><span className="help-icon"><MapPin size={17} /></span><div><strong>Plan a trip</strong><p>Select A for the start and B for the science site. Optionally enter a mission-relative sol and local solar time at the start. The routing service supplies P as a parking candidate.</p></div></div>
          <div className="help-item"><span className="help-icon"><Layers3 size={17} /></span><div><strong>Layers &amp; coverage</strong><p>Amber marks missing elevation; blue marks estimates. Clear areas show source data, not confirmed safety.</p></div></div>
          <div className="help-item"><span className="help-icon"><Box size={17} /></span><div><strong>Terrain detail</strong><p>{data ? `The source grid is about ${data.manifest.pixelSize[0].toFixed(1)} m. ` : ""}The 3D view shows its current resolution; missing areas are flat markers, not heights. Imagery alignment and height units are unverified.</p></div></div>
          <div className="help-item"><span className="help-icon"><Route size={17} /></span><div><strong>Routes</strong><p>{isRouteServiceConfigured ? "Routes come from the connected service and are estimates." : "Connect a route service to calculate routes."} A partial route ends at a candidate stopping point.</p></div></div>
        </div>
      </dialog>
    </div>
  );
}
