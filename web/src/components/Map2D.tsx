import { useEffect, useRef } from "react";
import Map from "ol/Map.js";
import View from "ol/View.js";
import Feature from "ol/Feature.js";
import { containsCoordinate, getCenter } from "ol/extent.js";
import LineString from "ol/geom/LineString.js";
import Point from "ol/geom/Point.js";
import TileLayer from "ol/layer/Tile.js";
import VectorLayer from "ol/layer/Vector.js";
import Projection from "ol/proj/Projection.js";
import XYZ from "ol/source/XYZ.js";
import VectorSource from "ol/source/Vector.js";
import TileGrid from "ol/tilegrid/TileGrid.js";
import CircleStyle from "ol/style/Circle.js";
import Fill from "ol/style/Fill.js";
import Stroke from "ol/style/Stroke.js";
import Style from "ol/style/Style.js";
import Text from "ol/style/Text.js";
import "ol/ol.css";

import type { LayerId, MapViewProps, MarsCoordinate } from "../types";

const layerOrder: LayerId[] = ["imagery", "elevation", "slope", "coverage"];

function markerStyle(label: string, color: string) {
  return new Style({
    image: new CircleStyle({
      radius: 11,
      fill: new Fill({ color }),
      stroke: new Stroke({ color: "#fff", width: 2 }),
    }),
    text: new Text({
      text: label,
      font: "700 11px system-ui, sans-serif",
      fill: new Fill({ color: "#fff" }),
    }),
    zIndex: 2,
  });
}

const startStyle = markerStyle("A", "#1b805a");
const destinationStyle = markerStyle("B", "#ab493f");
// "P" marks a service-provided candidate, not a validated site.
const stopStyle = markerStyle("P", "#a26519");
const routeStyles = [
  new Style({ stroke: new Stroke({ color: "rgba(14, 25, 31, 0.75)", width: 7 }) }),
  new Style({ stroke: new Stroke({ color: "#59dbed", width: 4 }) }),
];

/**
 * Render a single region in its native projected Mars coordinates.
 *
 * The manifest's tiles and all route/marker coordinates must share this
 * projection and extent. Tiles use a top-left origin and positive x/y indices;
 * their resolutions are explicitly supplied rather than assuming an Earth XYZ
 * grid. Reprojection and registration are data-preparation steps. The parent
 * must provide a container with a nonzero height.
 */
export function Map2D({
  region,
  layers,
  start,
  destination,
  stop,
  route,
  activeTool,
  onPlace,
}: MapViewProps) {
  const targetRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<Map | null>(null);
  const overlayRef = useRef<VectorSource | null>(null);
  const layersRef = useRef<Partial<Record<LayerId, TileLayer<XYZ>>>>({});
  const interactionRef = useRef({ activeTool, onPlace });
  interactionRef.current = { activeTool, onPlace };

  useEffect(() => {
    const target = targetRef.current;
    if (!target) return;

    // A custom projection retains the DEM's local x/y metre coordinates. We do
    // not register or transform to OpenLayers' Earth-based default projections.
    const projection = new Projection({
      code: region.projection,
      units: "m",
      extent: region.extent,
    });

    // The common tile grid is aligned to the original DEM pixel grid. Its
    // extent also prevents requests for tile coordinates outside the region.
    const tileGrid = new TileGrid({
      extent: region.extent,
      origin: [region.extent[0], region.extent[3]],
      minZoom: region.minZoom,
      resolutions: region.resolutions,
      tileSize: region.tileSize,
    });
    const rasterLayers: Partial<Record<LayerId, TileLayer<XYZ>>> = {};
    for (const id of layerOrder) {
      rasterLayers[id] = new TileLayer({
        source: new XYZ({
          url: region.layers[id],
          projection,
          tileGrid,
          // Imagery may provide more pixels for the same geographic tile.
          tilePixelRatio: id === "imagery" ? (region.imageryPixelRatio ?? 1) : 1,
          wrapX: false,
          transition: 0,
          // Coverage is categorical; interpolation must not blend its classes.
          interpolate: id !== "coverage",
        }),
      });
    }

    const overlay = new VectorSource({ wrapX: false });
    const map = new Map({
      target,
      layers: [
        ...layerOrder.map((id) => rasterLayers[id]!),
        new VectorLayer({ source: overlay }),
      ],
      view: new View({
        projection,
        center: getCenter(region.extent),
        extent: region.extent,
        // Keep the map center in the region without forcing the entire
        // viewport to remain inside it, which would prevent zooming out.
        constrainOnlyCenter: true,
        // Allow a wider context beyond the dataset's bounds; only the available
        // Mars tiles render, against the map's outside-data background.
        maxResolution: region.resolutions[region.minZoom] * 4,
        // Beyond the native resolution, zooming only enlarges native pixels.
        minResolution: region.resolutions[region.maxZoom] / 4,
        enableRotation: false,
      }),
    });

    mapRef.current = map;
    overlayRef.current = overlay;
    layersRef.current = rasterLayers;

    const placeAt = (coordinate: number[]) => {
      const { activeTool: tool, onPlace: place } = interactionRef.current;
      if (tool && containsCoordinate(region.extent, coordinate)) {
        place(tool, [coordinate[0], coordinate[1]]);
      }
    };

    map.on("singleclick", (event) => placeAt(event.coordinate));

    // OpenLayers provides keyboard pan and zoom when its target has tabIndex.
    // Enter/Space select the map center for people who cannot click the map.
    const onKeyDown = (event: KeyboardEvent) => {
      // Do not steal keyboard activation from a focused zoom/control button.
      if (event.target !== target) return;
      if (event.key !== "Enter" && event.key !== " ") return;
      if (!interactionRef.current.activeTool) return;
      const center = map.getView().getCenter();
      if (center) {
        event.preventDefault();
        placeAt(center);
      }
    };
    target.addEventListener("keydown", onKeyDown);

    // The map may mount while its panel is hidden or before it has dimensions.
    // Fit once after it first has a size, without resetting subsequent user zoom.
    let fitted = false;
    const fitWhenSized = () => {
      const size = map.getSize();
      if (!fitted && size && size[0] > 0 && size[1] > 0) {
        // The desktop planner floats over the left side of the map. Keep the
        // initial region overview visible in the space beside it.
        const leftPadding = window.matchMedia("(max-width: 700px)").matches ? 24 : 405;
        map.getView().fit(region.extent, { size, padding: [24, 24, 24, leftPadding] });
        fitted = true;
      }
    };
    map.updateSize();
    fitWhenSized();
    const resizeObserver = new ResizeObserver(() => {
      map.updateSize();
      fitWhenSized();
    });
    resizeObserver.observe(target);

    return () => {
      resizeObserver.disconnect();
      target.removeEventListener("keydown", onKeyDown);
      map.setTarget(undefined);
      map.dispose();
      if (mapRef.current === map) {
        mapRef.current = null;
        overlayRef.current = null;
        layersRef.current = {};
      }
    };
  }, [region]);

  useEffect(() => {
    for (const id of layerOrder) layersRef.current[id]?.setVisible(layers[id]);
    // Elevation can be viewed alone or blended over the visual basemap.
    layersRef.current.elevation?.setOpacity(layers.imagery ? 0.65 : 1);
  }, [region, layers]);

  useEffect(() => {
    const overlay = overlayRef.current;
    if (!overlay) return;
    overlay.clear();

    if (route.length > 1) {
      const line = new Feature(new LineString(route));
      line.setStyle(routeStyles);
      overlay.addFeature(line);
    }
    const addMarker = (coordinate: MarsCoordinate | null, style: Style) => {
      if (!coordinate) return;
      const marker = new Feature(new Point(coordinate));
      marker.setStyle(style);
      overlay.addFeature(marker);
    };
    addMarker(start, startStyle);
    addMarker(destination, destinationStyle);
    addMarker(stop, stopStyle);
  }, [region, start, destination, stop, route]);

  return (
    <div style={{ position: "relative", width: "100%", height: "100%", minHeight: 0 }}>
      <div
        ref={targetRef}
        role="region"
        tabIndex={0}
        aria-label="Mars map. Use arrow keys to pan and plus or minus to zoom. When placing a point, press Enter to select the map center."
        style={{
          width: "100%",
          height: "100%",
          background: "#292a2a",
          cursor: activeTool ? "crosshair" : "grab",
        }}
      />
      {activeTool && (
        <span
          aria-hidden="true"
          style={{
            position: "absolute",
            left: "50%",
            top: "50%",
            transform: "translate(-50%, -50%)",
            color: "white",
            textShadow: "0 0 4px #000, 0 0 2px #000",
            font: "bold 24px system-ui, sans-serif",
            pointerEvents: "none",
          }}
        >
          +
        </span>
      )}
    </div>
  );
}

export default Map2D;
