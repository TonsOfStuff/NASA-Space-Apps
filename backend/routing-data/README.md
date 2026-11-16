# Routing data

This folder contains the terrain manifest and tiles, elevation raster, and
thermal raster needed by the backend. The large data files are ignored by Git;
build the image locally with `backend/` as the Docker build context so that
`routing-data/` is included. The original `data/` and `.venv/` are excluded
from that context by `.dockerignore`.

With the image working directory set to the backend directory and `uv` and the
project dependencies installed, start the service with:

```sh
uv run spaceapps --host 0.0.0.0 --port 8080 \
  --manifest routing-data/manifest.json --terrain-dir routing-data/terrain \
  --dem routing-data/gale_20m_interpolated_feathered.tif \
  --thermal routing-data/thermal.cub
```

Use the configured Cloud Run port if it differs from `8080`. For a frontend on
a different origin, append `--allow-origin https://your-frontend.example` using
the frontend's actual origin.
