# Preparing the Mars elevation data

`interpolate_dem_gaps.py` creates a separate, estimated DEM and a provenance raster. It never modifies the source DEM. The original Gale mosaic contains very long narrow NoData seams connected to its outer missing region, so looking only for isolated, enclosed holes would miss them.

A missing pixel is eligible only when **original valid source cells are present on both sides** along the same horizontal, vertical or diagonal line, with a gap no wider than the configured pixel limit. The solver includes all eight immediate neighboring positions, weighted by inverse squared projected distance. Neighbors that are also eligible missing pixels are solved together; other unknown neighbors provide no assumed elevation. Original valid pixels are unchanged in the interpolated output. Boundary pixels, unsupported gaps and components larger than the configured limit remain NoData.

To recreate the derived Gale products from the repository root:

```sh
.venv/bin/python scripts/interpolate_dem_gaps.py \
  src/spaceapps/gale_20m_mosaic.tif \
  --output src/spaceapps/data/gale_20m_interpolated.tif \
  --provenance src/spaceapps/data/gale_20m_interpolated_provenance.tif \
  --max-gap-width 2 \
  --max-component-cells 4096 \
  --smooth-output src/spaceapps/data/gale_20m_interpolated_smoothed_sigma1.tif \
  --smooth-sigma 1
```

Choose fresh output names if those files already exist; the utility will not overwrite files. Omit both `--smooth-output` and `--smooth-sigma` when a separate smoothed raster is not needed. The optional Gaussian sigma is specified in source pixels: with the Gale grid, one pixel is approximately 20.23 projected meters. Gaussian smoothing is NaN-aware and leaves unsupported cells as NoData, but it alters original valid elevations as well as interpolated cells and can blur real terrain features.

The provenance raster uses **0 = original valid source, 1 = interpolated estimate, 2 = still missing**. For the smoothed DEM it describes the origin of each cell before smoothing; class 0 does not mean the smoothed value is identical to the original source value. Keep the provenance raster with either derived DEM.

The local DEM generated with these settings interpolated 4,242 cells; 22,061,853 remained missing. A full comparison confirmed that all original valid source cells were unchanged in the unsmoothed output. The optional smoothed output changed about 74.47 million original valid cells; in a regular sample the median absolute change was about 0.37 and the 99th percentile about 3.87 source height units, while the largest change anywhere was about 915 units. These are estimates, and elevation vertical units and image registration need verification. Existing DTM tiles also disagree where both contain data: filling NoData and applying a general blur do not register the tiles or correct their systematic offsets. The data remain a prototype for visualization and route experiments, not verified navigation terrain.

## Feather the central overlap

`blend_dem_overlap.py` creates a separate DEM by gradually blending source tile 3 into tile 4 near their shared boundary. It retains existing gap fills, excludes areas owned by the higher-priority tiles 1 and 2, leaves unsupported NoData unchanged, and marks every changed cell as class 1 in a new provenance raster. The default feather width is 128 source pixels (about 2.6 km). This reduces an abrupt step but does not establish which tile is more accurate or correct their registration.

The exact command to recreate the active blended DEM is in the [frontend setup](../web/README.md). On the current data, 1,084,886 heights were blended; over 2,816 adjacent-cell comparisons along the central boundary, the 90th-percentile height jump fell from about 198 to 8 source height units. These measurements are specific to this seam and dataset.

The frontend and team backend must explicitly select a derived DEM before they will use it. To regenerate map tiles with a different DEM, see `prepare_gale.py --help`; producing the GeoTIFF does not automatically change the running application. The source rasters and derived data products are ignored by Git.

Run the interpolation tests with:

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_interpolate_dem_gaps.py' -v
```
