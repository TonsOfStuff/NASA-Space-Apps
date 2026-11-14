import numpy as np
import pandas as pd
import rasterio

tif_path = "gale_20m_mosaic.tif"

with rasterio.open(tif_path) as src:
  # 1. Crop a 500x500 pixel patch near the center (500m x 500m area)
  center_row, center_col = src.height // 2, src.width // 2
  window = rasterio.windows.Window(center_col, center_row, 500, 500)

  heights = src.read(1, window=window)
  transform = src.window_transform(window)

  # 2. Generate grid indices and convert them to real (X, Y) coordinates in meters
  rows, cols = np.indices(heights.shape)
  xs, ys = rasterio.transform.xy(transform, rows, cols)

  # 3. Flatten into (X, Y, Height) columns and filter out NoData voids
  df = pd.DataFrame({"x_meters": np.array(xs).ravel(), "y_meters": np.array(ys).ravel(), "height_meters": heights.ravel()})
  if src.nodata is not None:
    df = df[df["height_meters"] != src.nodata]

df.to_csv("gale_crater_xyz.csv", index=False)
print(df.head())