FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.8 /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
RUN uv sync --locked --no-dev --no-cache

RUN apt-get update \
    && apt-get install -y --no-install-recommends libexpat1 \
    && rm -rf /var/lib/apt/lists/* \
    && MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python -c 'import spaceapps.api' \
    && rm -rf /tmp/matplotlib

COPY routing-data/ ./routing-data/
RUN test -s routing-data/manifest.json \
    && test -s routing-data/terrain/0/0/0.f32 \
    && test -s routing-data/gale_20m_interpolated_feathered.tif \
    && test -s routing-data/thermal.cub

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    NUMBA_CACHE_DIR=/tmp/numba \
    MPLCONFIGDIR=/tmp/matplotlib

USER nobody

CMD ["sh", "-c", "if [ -n \"$FRONTEND_ORIGIN\" ]; then set -- --allow-origin \"$FRONTEND_ORIGIN\"; fi; exec spaceapps --host 0.0.0.0 --port \"${PORT:-8080}\" --manifest routing-data/manifest.json --terrain-dir routing-data/terrain --dem routing-data/gale_20m_interpolated_feathered.tif --thermal routing-data/thermal.cub \"$@\""]
