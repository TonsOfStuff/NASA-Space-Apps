import numpy as np
import rasterio

# This must be a crop of the DEM, not the color hillshade.
with rasterio.open(input("give me a file: ")) as dataset:
    print("Bands:", dataset.count)
    print("Data type:", dataset.dtypes[0])
    print("CRS:", dataset.crs)
    print("Pixel size:", dataset.res)
    print("NoData:", dataset.nodata)
    print("Scale and offset:", dataset.scales[0], dataset.offsets[0])

    # Preserve the dataset's NoData mask and apply any scale/offset metadata.
    raw = dataset.read(1, masked=True).astype(np.float64)
    elevation = raw * dataset.scales[0] + dataset.offsets[0]

    if elevation.count() == 0:
        raise ValueError("The crop contains no valid elevation pixels.")

    print("Min elevation (m):", elevation.min())
    print("Max elevation (m):", elevation.max())

    np.savetxt(
        "mars_elevations.csv",
        elevation.filled(np.nan),
        delimiter=",",
    )