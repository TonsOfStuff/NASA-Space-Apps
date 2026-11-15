# Wayfinder frontend

A React, TypeScript, OpenLayers and Three.js interface for the available Gale Crater imagery and elevation data. The map uses the DEM's own Mars-projected coordinates, supports imagery, elevation, slope and missing-elevation layers, and loads increasingly detailed tiles as you zoom. The 3D view uses the same terrain and imagery tiles.

## Run locally

Use Node.js 22.12 or newer. The source GeoTIFFs are not checked into Git. Place `gale_20m_mosaic.tif` in `src/spaceapps/` and `curiosity_ctx_mosaic.tif` in `src/spaceapps/data/`. The existing `merge_dtms.py` can build the elevation mosaic from the source DTM tiles. The frontend uses a separately interpolated mosaic and its provenance raster by default. See `scripts/prepare_gale.py --help` for alternate input and output paths.

```sh
# From the repository root, after installing the Python dependencies.
# Run once to generate the derived DEM and provenance; this never overwrites them:
.venv/bin/python scripts/interpolate_dem_gaps.py src/spaceapps/gale_20m_mosaic.tif \
  --output src/spaceapps/data/gale_20m_interpolated.tif \
  --provenance src/spaceapps/data/gale_20m_interpolated_provenance.tif

# Then prepare the frontend tiles:
.venv/bin/python scripts/prepare_gale.py

cd web
npm install
npm run dev
```

The full tile pyramid generated from the current Gale files occupies about 1 GiB; preparation used about 1.3 GiB of peak memory on the development machine. Generated tiles are local data artifacts and are ignored by Git. The production build currently copies them into `dist`. Run `npm run build` to type-check and build the frontend and `npm test` to check the proposed route response adapter.

For Google Cloud hosting, the frontend can be served from Firebase Hosting and the generated tiles from Cloud Storage, with CORS enabled for the frontend's origin. Generate the region manifest with `--asset-base` set to the tile hosting URL and publish the manifest with the frontend. Configure the team's route-service URL separately; the frontend's source does not deploy or implement that backend.

To swap in another compatible region, prepare a separate manifest and tile set with `--dem`, `--imagery`, `--id`, `--name`, `--output`, and `--asset-base`, then point `VITE_REGION_MANIFEST_URL` at that manifest. The current preparation script supports the same spherical, equirectangular Mars projection used by Gale; other projections need an appropriate slope-distance model before they are accepted. Each new dataset still needs its coordinate conventions and alignment checked.

## Connecting the team's route service

The backend and pathfinding are intentionally maintained separately. Until an API contract is agreed, [`src/routeService.ts`](src/routeService.ts) is the only frontend module that needs to be adapted. There is currently **no configured endpoint**. Set `VITE_ROUTE_API_URL` in a local `.env.local` file and restart Vite, or set it in the frontend build environment. For example, after the team supplies its actual URL:

```sh
VITE_ROUTE_API_URL=https://example-team-service.invalid/api/route
```

If the backend is on another origin, configure its CORS policy for the frontend origin. The route button stays unavailable when no URL is configured; the application does not invent route results.

The **proposed**, not yet agreed, JSON contract is:

```json
POST <VITE_ROUTE_API_URL>
{
  "regionId": "gale",
  "start": [-2494500, -316000],
  "destination": [-2493000, -315000],
  "rover": { "maxSlopeDeg": 15 }
}
```

```json
{
  "regionId": "gale",
  "status": "reached",
  "path": [[-2494490, -315990], [-2493100, -315100]],
  "stop": [-2493100, -315100],
  "distanceM": 1800,
  "evaluatedStart": [-2494490, -315990],
  "evaluatedDestination": [-2493010, -315010],
  "resolutionM": [20.234, 20.234],
  "warnings": ["Example only: terrain and route must be independently verified."]
}
```

Coordinates are `[easting, northing]` in the selected region's projected **Mars** CRS and are interpreted using the region manifest; they are not Earth coordinates. `status` is `reached`, `partial`, or `unreachable`. The backend can return `stop` as its selected parking point; the frontend does not choose it. `unreachable` uses an empty path. `distanceM`, evaluated points and resolution may be null when unavailable. If the backend evaluates cell centers rather than the exact selections, return those positions so the UI can disclose the displacement. Errors can use `{ "error": { "code": "...", "message": "..." } }` with a non-success HTTP status.

This format is an integration proposal; if your team chooses a different endpoint, wire format, or coordinate system, update the adapter to transform the service's data into the frontend types. It validates responses and cancels obsolete requests when the selection or rover settings change. Do not silently convert missing coverage into traversable terrain.

## Data limitations

The available elevation mosaic has approximately 20.23 m projected source pixel spacing, and the available imagery has its own coverage limits. The default derived DEM fills 4,242 narrow-gap cells using surrounding heights; the provenance raster records which cells are estimated. Amber coverage marks remaining missing elevation and blue marks interpolated cells. Tiles shown at broader zoom levels are coarser; zooming beyond the native source resolution cannot reveal additional terrain detail. The source imagery and DEM have different Mars reference-system metadata and their feature alignment has not been verified. The source imagery lacks a declared NoData mask, and elevation vertical units and datum need source-documentation verification. Slope and 3D height are illustrative estimates, not safety assessments. A clear coverage cell does not mean the area is safe. The rectangular source footprint has not been verified to include every part of the crater.
