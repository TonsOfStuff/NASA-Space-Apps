# Routing data

This folder contains the terrain manifest and tiles, elevation raster, and
thermal raster needed by the backend. The large data files are ignored by Git;
build the image locally with `backend/` as the Docker build context so that
`routing-data/` is included. The original `data/` and `.venv/` are excluded
from that context by `.dockerignore`.

From the backend directory with `uv` installed, start the service locally with:

```sh
uv run spaceapps --host 0.0.0.0 --port 8080 \
  --manifest routing-data/manifest.json --terrain-dir routing-data/terrain \
  --dem routing-data/gale_20m_interpolated_feathered.tif \
  --thermal routing-data/thermal.cub
```

Use the configured Cloud Run port if it differs from `8080`. For a frontend on
a different origin, append `--allow-origin https://your-frontend.example` using
the frontend's actual origin.

## Container image

From the repository root, build and test the image:

```sh
docker build --platform linux/amd64 -t mars-routing:latest backend/
docker run --rm -p 8080:8080 mars-routing:latest
```

Check `http://localhost:8080/api/health` while the container is running. On
Cloud Run, the image listens on `PORT`. For a separately hosted frontend, set
the `FRONTEND_ORIGIN` environment variable to its exact origin, such as
`https://example.com` (without a trailing slash).
