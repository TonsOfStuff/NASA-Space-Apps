# NASA-Space-Apps

An exploratory Mars map and terrain-routing application.

- [`frontend/`](frontend/) contains the React map and its Node.js dependencies.
- [`backend/`](backend/) contains the Python route and map-layer service, its
  Python environment configuration, and local raster data in `backend/data/`.
- [`scripts/`](scripts/) contains the one-time terrain preparation utilities
  and their tests.

From the repository root, install dependencies and start the services:

```sh
(cd backend && uv sync)
backend/.venv/bin/python -m spaceapps.api
```

In a second terminal:

```sh
cd frontend
npm install
npm run dev
```

The map and routing service require local raster datasets and generated tiles;
see the [frontend setup](frontend/README.md) and [data preparation guide](scripts/README.md).
