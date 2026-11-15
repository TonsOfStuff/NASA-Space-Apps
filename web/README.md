# MarMap frontend

A React, TypeScript, OpenLayers and Three.js interface for the available Gale Crater imagery and elevation data. The map uses the DEM's own Mars-projected coordinates, supports imagery, elevation, slope and independently selectable estimated and missing data coverage, and loads increasingly detailed tiles as you zoom. The 3D view drapes the same imagery tiles over the terrain; missing areas appear on a flat marker plane because their heights are unknown. Its optional warm tint adds display color to the grayscale CTX imagery; turn it off under Layers to see the original tones. The tint does not represent measured surface color.

## Run locally

Use Node.js 22.12 or newer. The source GeoTIFFs are not checked into Git. Place `gale_20m_mosaic.tif` in `src/spaceapps/`, the original DTM tiles in `src/spaceapps/DTMs/`, and `curiosity_ctx_mosaic.tif` in `src/spaceapps/data/`. The existing `merge_dtms.py` can build the elevation mosaic from the source DTM tiles. The frontend uses a separately interpolated and overlap-blended mosaic and its provenance raster by default. See `scripts/prepare_gale.py --help` for alternate input and output paths.

```sh
# From the repository root, after installing the Python dependencies.
# Run once to generate the derived DEM and provenance; this never overwrites them:
.venv/bin/python scripts/interpolate_dem_gaps.py src/spaceapps/gale_20m_mosaic.tif \
  --output src/spaceapps/data/gale_20m_interpolated.tif \
  --provenance src/spaceapps/data/gale_20m_interpolated_provenance.tif

# Blend the tile 3/4 overlap into another separate DEM:
.venv/bin/python scripts/blend_dem_overlap.py src/spaceapps/data/gale_20m_interpolated.tif \
  --provenance src/spaceapps/data/gale_20m_interpolated_provenance.tif \
  --earlier src/spaceapps/DTMs/PG_Gale_CTX_3_3_AATE_20m.tif \
  --later src/spaceapps/DTMs/PG_Gale_CTX_4_AATE_20m.tif \
  --exclude src/spaceapps/DTMs/PG_Gale_CTX_1_AATE_20m.tif \
  --exclude src/spaceapps/DTMs/PG_Gale_CTX_2_2_AATE_20m.tif \
  --output src/spaceapps/data/gale_20m_interpolated_feathered.tif \
  --output-provenance src/spaceapps/data/gale_20m_interpolated_feathered_provenance.tif

# Then prepare the frontend tiles:
.venv/bin/python scripts/prepare_gale.py

cd web
npm install
npm run dev
```

The full tile pyramid generated from the current Gale files occupies about 1 GiB; preparation used about 1.3 GiB of peak memory on the development machine. Generated tiles are local data artifacts and are ignored by Git. The production build currently copies them into `dist`. Run `npm run build` to type-check and build the frontend and `npm test` to check the proposed route response adapter.

For Google Cloud hosting, the frontend can be served from Firebase Hosting and the generated tiles from Cloud Storage, with CORS enabled for the frontend's origin. Generate the region manifest with `--asset-base` set to the tile hosting URL and publish the manifest with the frontend. Configure the team's route-service URL separately; the frontend's source does not deploy or implement that backend.

To swap in another compatible region, prepare a separate manifest and tile set with `--dem`, `--imagery`, `--id`, `--name`, `--output`, and `--asset-base`, then point `VITE_REGION_MANIFEST_URL` at that manifest. The current preparation script supports the same spherical, equirectangular Mars projection used by Gale; other projections need an appropriate slope-distance model before they are accepted. Each new dataset still needs its coordinate conventions and alignment checked.

## Running the route service

From the repository root, start the Python backend in a second terminal:

```sh
.venv/bin/python -m spaceapps.api
```

It listens on `http://127.0.0.1:8000/api/route` and uses the existing `new_algo.py` router, the Gale terrain tiles, and `src/spaceapps/data/thermal.cub`. The frontend connects to this address automatically during development. The terrain tiles and thermal cube must be present on the machine running the backend. Use `python -m spaceapps.api --help` for alternate file paths, host, port, and additional allowed frontend origins.

In production, the frontend defaults to `/api/route` on its own origin. Set `VITE_ROUTE_API_URL` in a local `.env.local` file or in the frontend build environment to use a different service. For example:

```sh
VITE_ROUTE_API_URL=https://example-team-service.invalid/api/route
```

If the backend is on another origin, allow that frontend origin with the backend's `--allow-origin` option. The default CORS policy permits the local Vite origins on port 5173.

The JSON contract is:

```json
POST <VITE_ROUTE_API_URL>
{
  "regionId": "gale",
  "start": [-2494500, -316000],
  "destination": [-2493000, -315000],
  "rover": { "maxSlopeDeg": 15 },
  "marsDateTime": { "sol": 42, "localSolarTime": "14:30", "solReference": "mission" }
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
  "thermalInertia": { "coverageFraction": 1, "meanAlongRoute": 280, "units": "TIU" },
  "warnings": ["Example only: terrain and route must be independently verified."]
}
```

Coordinates are `[easting, northing]` in the selected region's projected **Mars** CRS and are interpreted using the region manifest; they are not Earth coordinates. `marsDateTime` is optional: `sol` is a non-negative mission-relative sol and `localSolarTime` is a 24-hour `HH:MM` local solar clock at the start point. With a time, the backend estimates the Sun's direction assuming an equinox; the sol cannot determine season until the mission epoch is known. Without a time, sunlight is not used as a route preference. The backend reports these assumptions in its warnings. `status` is `reached`, `partial`, or `unreachable`. The backend returns its route endpoint as `stop`; this is a parking candidate, not a separate parking-suitability calculation. `unreachable` uses an empty path. `distanceM`, evaluated points and resolution may be null when unavailable. If the backend evaluates cell centers rather than the exact selections, return those positions so the UI can disclose the displacement. Errors use `{ "error": { "code": "...", "message": "..." } }` with a non-success HTTP status.

The service uses a bounded search area around the chosen points and selects a coarser terrain level for large requests. It warns about missing thermal data and reduced resolution. The thermal thresholds use the defaults in `new_algo.py`; its input cube has approximately 100 m source pixels. The frontend adapter validates responses and cancels obsolete requests when the selection or rover settings change. A future service can replace the adapter without changing the map controls.

The **Shade** and **Thermal inertia** map layers are rendered by the backend on demand; no additional tile archive is stored. Thermal values use a fixed 100–600 TIU color scale, with missing values transparent. Shade uses the local solar time entered in the planner, or 12:00 when empty, with an equinox assumption and a 500 m shadow search. The mission sol does not determine a season until an epoch is provided. The API keeps a bounded in-memory cache of generated PNG tiles. If map layers are hosted at a different API base from routing, set `VITE_MAP_API_BASE_URL` in the frontend environment.

## Data limitations

The available elevation mosaic has approximately 20.23 m projected source pixel spacing, and the available imagery has its own coverage limits. The default derived DEM fills 4,242 narrow-gap cells and blends 1,084,886 cells near the tile 3/4 overlap; the provenance raster records both as estimates. Amber coverage marks remaining missing elevation and blue marks estimated cells. Tiles shown at broader zoom levels are coarser; zooming beyond the native source resolution cannot reveal additional terrain detail. The source imagery and DEM have different Mars reference-system metadata and their feature alignment has not been verified. The source imagery lacks a declared NoData mask, and elevation vertical units and datum need source-documentation verification. Slope and 3D height are illustrative estimates, not safety assessments. A clear coverage cell does not mean the area is safe. The rectangular source footprint has not been verified to include every part of the crater.
