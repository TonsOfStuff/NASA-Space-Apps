import type { MarsCoordinate, RegionManifest } from "./types";

/**
 * Frontend adapter for the route API. Keep backend-specific wire format and
 * coordinate conversions here if a different service is connected later.
 */
export interface RouteRequest {
  regionId: string;
  start: MarsCoordinate;
  destination: MarsCoordinate;
  rover: { maxSlopeDeg: number };
  /** Mission-relative sol and the local solar clock at the starting point. */
  marsDateTime?: {
    sol: number;
    localSolarTime: string;
    solReference: "mission";
  };
}

export interface RouteResult {
  status: "reached" | "partial" | "unreachable";
  path: MarsCoordinate[];
  stop: MarsCoordinate | null;
  distanceM: number | null;
  evaluatedStart: MarsCoordinate | null;
  evaluatedDestination: MarsCoordinate | null;
  resolutionM: MarsCoordinate | null;
  warnings: string[];
  thermalInertia: { coverageFraction: number; meanAlongRoute: number | null; units: string } | null;
}

const endpoint = (import.meta.env?.VITE_ROUTE_API_URL as string | undefined)?.trim() ||
  (import.meta.env?.DEV ? "http://127.0.0.1:8000/api/route" : "/api/route");
export const isRouteServiceConfigured = Boolean(endpoint);

/** On-demand layers live alongside the route endpoint; no tile archive is required. */
export function environmentalLayerUrls(localSolarTime: string): { shade: string; thermal: string } {
  const configuredBase = (import.meta.env?.VITE_MAP_API_BASE_URL as string | undefined)?.trim();
  const base = (configuredBase || endpoint.replace(/\/route(?:\?.*)?$/, "")).replace(/\/$/, "");
  return {
    shade: `${base}/tiles/shade/{z}/{x}/{y}.png?localSolarTime=${encodeURIComponent(localSolarTime)}`,
    thermal: `${base}/tiles/thermal/{z}/{x}/{y}.png`,
  };
}

const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

function coordinate(value: unknown): MarsCoordinate | null {
  return Array.isArray(value) && value.length === 2 &&
    typeof value[0] === "number" && Number.isFinite(value[0]) &&
    typeof value[1] === "number" && Number.isFinite(value[1])
    ? [value[0], value[1]] : null;
}

function inRegion(point: MarsCoordinate, region: RegionManifest): boolean {
  const [left, bottom, right, top] = region.extent;
  return point[0] >= left && point[0] <= right && point[1] >= bottom && point[1] <= top;
}

export async function requestRoute(
  request: RouteRequest,
  region: RegionManifest,
  signal: AbortSignal,
): Promise<RouteResult> {
  if (!endpoint) throw new Error("Route service is not configured.");
  let response: Response;
  try {
    response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(request),
      signal,
    });
  } catch (error) {
    if (signal.aborted) throw error;
    throw new Error("Could not reach the route service. Check its URL and connection.");
  }

  const payload: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const error = isObject(payload) && isObject(payload.error) ? payload.error : null;
    throw new Error(error && typeof error.message === "string" ? error.message : `Route service returned HTTP ${response.status}.`);
  }
  return parseRouteResponse(payload, region);
}

/** Validate the proposed wire format before any backend geometry reaches the map. */
export function parseRouteResponse(payload: unknown, region: RegionManifest): RouteResult {
  if (!isObject(payload) || payload.regionId !== region.id ||
      (payload.status !== "reached" && payload.status !== "partial" && payload.status !== "unreachable") ||
      !Array.isArray(payload.path)) {
    throw new Error("The route service returned an unrecognized response. Check the API adapter contract.");
  }
  const points = payload.path.map(coordinate);
  if (points.some((point) => point === null || !inRegion(point, region))) {
    throw new Error("The route service returned an invalid or out-of-region path.");
  }
  const path = points as MarsCoordinate[];
  const stop = payload.stop == null ? null : coordinate(payload.stop);
  if ((payload.stop != null && (!stop || !inRegion(stop, region))) ||
      (payload.status === "partial" && (!stop || path.length === 0)) ||
      (payload.status === "reached" && path.length === 0) ||
      (payload.status === "unreachable" && path.length !== 0)) {
    throw new Error("The route service returned inconsistent route data.");
  }
  const distanceM = payload.distanceM == null ? null : payload.distanceM;
  if (distanceM !== null && (typeof distanceM !== "number" || !Number.isFinite(distanceM) || distanceM < 0)) {
    throw new Error("The route service returned an invalid distance.");
  }
  const evaluatedStart = payload.evaluatedStart == null ? null : coordinate(payload.evaluatedStart);
  const evaluatedDestination = payload.evaluatedDestination == null ? null : coordinate(payload.evaluatedDestination);
  if ((payload.evaluatedStart != null && (!evaluatedStart || !inRegion(evaluatedStart, region))) ||
      (payload.evaluatedDestination != null && (!evaluatedDestination || !inRegion(evaluatedDestination, region)))) {
    throw new Error("The route service returned an invalid evaluated location.");
  }
  const thermal = isObject(payload.thermalInertia) ? payload.thermalInertia : null;
  const thermalInertia = thermal && typeof thermal.coverageFraction === "number" && Number.isFinite(thermal.coverageFraction) &&
    thermal.coverageFraction >= 0 && thermal.coverageFraction <= 1 &&
    (thermal.meanAlongRoute === null || (typeof thermal.meanAlongRoute === "number" && Number.isFinite(thermal.meanAlongRoute))) &&
    typeof thermal.units === "string"
    ? { coverageFraction: thermal.coverageFraction, meanAlongRoute: thermal.meanAlongRoute, units: thermal.units }
    : null;
  return {
    status: payload.status,
    path,
    stop,
    distanceM,
    evaluatedStart,
    evaluatedDestination,
    resolutionM: coordinate(payload.resolutionM),
    warnings: Array.isArray(payload.warnings) ? payload.warnings.filter((warning): warning is string => typeof warning === "string") : [],
    thermalInertia,
  };
}
