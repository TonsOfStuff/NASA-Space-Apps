import assert from "node:assert/strict";
import { test } from "node:test";
import { parseRouteResponse } from "../src/routeService.ts";
import type { RegionManifest } from "../src/types.ts";

const region = { id: "gale", extent: [0, 0, 100, 100] } as RegionManifest;
const response = {
  regionId: "gale",
  status: "reached",
  path: [[10, 10], [80, 80]],
  stop: null,
  distanceM: 100,
  evaluatedStart: [10, 10],
  evaluatedDestination: [80, 80],
  resolutionM: [20, 20],
  warnings: ["Experimental route"],
};

test("accepts a valid route without changing the service's coordinates or warnings", () => {
  const result = parseRouteResponse(response, region);
  assert.deepEqual(result.path, response.path);
  assert.deepEqual(result.warnings, response.warnings);
});

test("preserves a partial route and its candidate endpoint", () => {
  const result = parseRouteResponse({ ...response, status: "partial", stop: [50, 50] }, region);
  assert.equal(result.status, "partial");
  assert.deepEqual(result.stop, [50, 50]);
});

test("preserves a parking point selected by the route service", () => {
  const result = parseRouteResponse({ ...response, stop: [70, 65] }, region);
  assert.deepEqual(result.stop, [70, 65]);
});

test("rejects mismatched regions, malformed geometry and inconsistent statuses", () => {
  assert.throws(() => parseRouteResponse({ ...response, regionId: "earth" }, region));
  assert.throws(() => parseRouteResponse({ ...response, path: [[10, 10], [101, 10]] }, region));
  assert.throws(() => parseRouteResponse({ ...response, path: [] }, region));
  assert.throws(() => parseRouteResponse({ ...response, status: "partial", stop: null }, region));
  assert.throws(() => parseRouteResponse({ ...response, status: "unreachable" }, region));
  assert.throws(() => parseRouteResponse({ ...response, evaluatedStart: [NaN, 10] }, region));
});

test("accepts an unreachable result without creating a route", () => {
  const result = parseRouteResponse({ ...response, status: "unreachable", path: [], distanceM: null }, region);
  assert.deepEqual(result.path, []);
  assert.equal(result.distanceM, null);
});
