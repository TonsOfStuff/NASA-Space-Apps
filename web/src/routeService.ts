import type { MarsCoordinate, RegionManifest } from "./types";

/**
 * Proposed adapter contract, pending agreement with the backend team.
 * Replace this module to map a different API or coordinate system; the UI
 * should not need to know about the backend's wire format.
 */
export interface RouteRequest {
  regionId: string;
  start: MarsCoordinate;
  destination: MarsCoordinate;
  rover: { maxSlopeDeg: number };
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
}

const endpoint = (import.meta.env?.VITE_ROUTE_API_URL as string | undefined)?.trim() ?? "";
export const isRouteServiceConfigured = Boolean(endpoint);

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
  return {
    status: payload.status,
    path,
    stop,
    distanceM,
    evaluatedStart,
    evaluatedDestination,
    resolutionM: coordinate(payload.resolutionM),
    warnings: Array.isArray(payload.warnings) ? payload.warnings.filter((warning): warning is string => typeof warning === "string") : [],
  };
}
