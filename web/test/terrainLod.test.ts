import assert from "node:assert/strict";
import { test } from "node:test";
import { PerspectiveCamera, Vector3 } from "three";
import { MAX_TERRAIN_TILES, overviewZoom, pickTiles } from "../src/terrainLod.ts";
import type { RegionManifest } from "../src/types.ts";

const region = {
  extent: [-32768, -32768, 32768, 32768],
  tileSize: 256,
  minZoom: 0,
  maxZoom: 6,
  resolutions: [128, 64, 32, 16, 8, 4, 2],
} as RegionManifest;
const height = 900;
const target = new Vector3(0, 0, 0);

function cameraAt(groundPerPixel: number, elevationAngle: number) {
  const camera = new PerspectiveCamera(45, 16 / 9, 1, 1000000);
  const distance = groundPerPixel * height / (2 * Math.tan(Math.PI / 8));
  camera.position.set(0, distance * Math.sin(elevationAngle), distance * Math.cos(elevationAngle));
  camera.lookAt(target);
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld();
  return camera;
}

test("terrain detail stays at the same level when the camera rotates", () => {
  const overhead = pickTiles(region, cameraAt(15, Math.PI / 2 - 0.01), target, height);
  const acrossSurface = pickTiles(region, cameraAt(15, 0.1), target, height);
  assert.equal(overhead.z, 3);
  assert.equal(acrossSurface.z, overhead.z);
  assert.equal(acrossSurface.key, overhead.key);
  assert.ok(acrossSurface.tiles.length <= MAX_TERRAIN_TILES);
  assert.ok(acrossSurface.tiles.length > 0);
});

test("the next detail level loads before the previous grid becomes coarse", () => {
  assert.equal(pickTiles(region, cameraAt(110, 1), target, height).z, 0);
  assert.equal(pickTiles(region, cameraAt(100, 1), target, height).z, 1);
  // The old unbiased rounding still selected level 0 at this distance.
  assert.equal(Math.round(Math.log2(region.resolutions[0] / 100)), 0);
  assert.equal(pickTiles(region, cameraAt(98, 1), target, height, 0).z, 1);
});

test("orbiting exactly at a detail threshold does not switch levels", () => {
  const threshold = region.resolutions[0] / 2 ** 0.3;
  const first = pickTiles(region, cameraAt(threshold, 0.05), target, height);
  for (const angle of [0.1, 0.3, 0.7, 1, 1.3, 1.55]) {
    const rotated = pickTiles(region, cameraAt(threshold, angle), target, height, first.z);
    assert.equal(rotated.key, first.key);
  }
});

test("terrain detail stays within the available zoom levels", () => {
  assert.equal(pickTiles(region, cameraAt(1000, 1), target, height).z, region.minZoom);
  assert.equal(pickTiles(region, cameraAt(0.1, 1), target, height).z, region.maxZoom);
});

test("the detailed tile window moves as the orbit target is panned", () => {
  const camera = cameraAt(15, 1);
  const first = pickTiles(region, camera, target, height);
  const movedTarget = new Vector3(region.tileSize * region.resolutions[first.z] * 2, 0, 0);
  camera.position.add(movedTarget);
  const moved = pickTiles(region, camera, movedTarget, height);
  assert.equal(moved.z, first.z);
  assert.notEqual(moved.key, first.key);
  assert.ok(moved.tiles.length <= MAX_TERRAIN_TILES);
});

test("the whole-region backdrop uses at most four tiles when possible", () => {
  assert.equal(overviewZoom(region), 0);
  const smaller = { ...region, extent: [-16384, -16384, 16384, 16384] } as RegionManifest;
  assert.equal(overviewZoom(smaller), 1);
  const larger = { ...region, extent: [-65536, -65536, 65536, 65536] } as RegionManifest;
  assert.equal(overviewZoom(larger), null);
});
