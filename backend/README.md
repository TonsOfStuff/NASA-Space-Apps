# Mars routing backend

The Python service provides routes and on-demand map layers. It depends on the
terrain router in `src/spaceapps/new_algo.py` and uses datasets in `data/` plus
the terrain tiles generated in `../frontend/public/data/gale/`.

From this directory, install the Python 3.14 environment with `uv sync`, then
run `uv run python -m spaceapps.api` (or `uv run spaceapps`). Use `--help` for
alternate data paths, host, port, and allowed frontend origins.

The local raster datasets are not committed. See the [frontend setup](../frontend/README.md)
and [data preparation guide](../scripts/README.md) for the required files and
commands to generate the terrain products.

For a self-contained local container build, see [routing data](routing-data/README.md).

Run the backend tests from the repository root with
`backend/.venv/bin/python -m unittest discover -s backend/tests -v`.
