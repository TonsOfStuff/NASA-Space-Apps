from contextlib import ExitStack
from pathlib import Path

import rasterio
from rasterio.enums import MaskFlags
from rasterio.merge import merge

tile_paths = sorted(Path("DTMs").glob("*_20m.tif"))

if not tile_paths:
    raise FileNotFoundError(
        "No tiles found. Put the extracted DTMs folder next to this script."
    )

with ExitStack() as stack:
    tiles = [
        stack.enter_context(rasterio.open(path))
        for path in tile_paths
    ]

    print(f"Found {len(tiles)} elevation tiles:\n")

    for path, tile in zip(tile_paths, tiles):
        print(
            f"{path.name}: "
            f"CRS={tile.crs}, "
            f"resolution={tile.res}, "
            f"NoData={tile.nodata}, "
            f"mask={tile.mask_flag_enums[0]}"
        )

    # Do not silently combine tiles whose positioning or missing-data
    # handling cannot be established.
    if any(tile.crs is None for tile in tiles):
        raise ValueError("A tile has no CRS. Inspect its metadata first.")

    if any(tile.crs != tiles[0].crs for tile in tiles):
        raise ValueError("The tiles have different CRSs; reproject them first.")

    if any(tile.count != 1 for tile in tiles):
        raise ValueError("Expected one elevation band per tile.")

    if any(
        tile.scales[0] != 1 or tile.offsets[0] != 0
        for tile in tiles
    ):
        raise ValueError("A tile has a scale or offset; inspect it first.")

    if any(
        MaskFlags.all_valid in tile.mask_flag_enums[0]
        for tile in tiles
    ):
        raise ValueError(
            "At least one tile has no NoData value or mask. "
            "Check the dataset README and tile metadata before merging."
        )

    merge(
        tiles,
        dst_path="gale_20m_mosaic.tif",
        dtype="float32",
        nodata=float("nan"),
        method="first",
        dst_kwds={"compress": "deflate"},
    )

print("\nSaved gale_20m_mosaic.tif")