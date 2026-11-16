"""Merge the original Gale elevation tiles into a single DEM."""

import argparse
from contextlib import ExitStack
from pathlib import Path

import rasterio
from rasterio.enums import MaskFlags
from rasterio.merge import merge

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TILES = ROOT / "backend/data/DTMs"
DEFAULT_OUTPUT = ROOT / "backend/data/gale_20m_mosaic.tif"


def merge_dem_tiles(tile_dir: Path, output: Path) -> None:
    tile_paths = sorted(tile_dir.glob("*_20m.tif"))
    if not tile_paths:
        raise FileNotFoundError(f"No elevation tiles found in {tile_dir}.")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    with ExitStack() as stack:
        tiles = [stack.enter_context(rasterio.open(path)) for path in tile_paths]
        print(f"Found {len(tiles)} elevation tiles:\n")

        for path, tile in zip(tile_paths, tiles):
            print(
                f"{path.name}: "
                f"CRS={tile.crs}, "
                f"resolution={tile.res}, "
                f"NoData={tile.nodata}, "
                f"mask={tile.mask_flag_enums[0]}"
            )

        if any(tile.crs is None for tile in tiles):
            raise ValueError("A tile has no CRS. Inspect its metadata first.")

        if any(tile.crs != tiles[0].crs for tile in tiles):
            raise ValueError("The tiles have different CRSs; reproject them first.")

        if any(tile.count != 1 for tile in tiles):
            raise ValueError("Expected one elevation band per tile.")

        if any(tile.scales[0] != 1 or tile.offsets[0] != 0 for tile in tiles):
            raise ValueError("A tile has a scale or offset; inspect it first.")

        if any(MaskFlags.all_valid in tile.mask_flag_enums[0] for tile in tiles):
            raise ValueError(
                "At least one tile has no NoData value or mask. "
                "Check the dataset README and tile metadata before merging."
            )

        output.parent.mkdir(parents=True, exist_ok=True)
        merge(
            tiles,
            dst_path=output,
            dtype="float32",
            nodata=float("nan"),
            method="first",
            dst_kwds={"compress": "deflate"},
        )

    print(f"\nSaved {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiles", type=Path, default=DEFAULT_TILES, help="Directory containing the source DTM tiles")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output DEM; must not already exist")
    args = parser.parse_args()
    try:
        merge_dem_tiles(args.tiles, args.output)
    except (OSError, ValueError, rasterio.errors.RasterioError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
