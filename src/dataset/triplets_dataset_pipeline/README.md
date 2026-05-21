# PhiSat-2 & Sentinel-2 Multi-Modal Dataset (v1)

## Overview
This dataset provides a multi-modal, co-registered collection of satellite imagery patches. Its primary purpose is to provide spatially aligned triplets of high-resolution optical data from PhiSat-2 (real acquisitions), PhiSat-2 simulated images, and multi-spectral data from Sentinel-2.

## Band Ordering & Spectral Specifications
The dataset aligns PhiSat-2 multispectral capabilities with a specific 7-band subset of Sentinel-2. It is important to note the channel ordering for the image tensors.

### PhiSat-2 (Real & Simulated)
Tensors in `real/images` and `sim/images` contain 8 bands:
- **Band 0:** Panchromatic (PAN)
- **Band 1:** Blue (490 nm)
- **Band 2:** Green (560 nm)
- **Band 3:** Red (665 nm)
- **Band 4:** Red Edge 1 (705 nm)
- **Band 5:** Red Edge 2 (740 nm)
- **Band 6:** Red Edge 3 (783 nm)
- **Band 7:** Near Infrared - NIR (842 nm)

*Note: The Spectral Response Functions (SRF) are not identical between PhiSat-2 and Sentinel-2B.*

### Sentinel-2B
Tensors in `s2b/images` contain 7 bands, extracted from the standard Sentinel-2 products:
- **Band 0:** B02 (Blue)
- **Band 1:** B03 (Green)
- **Band 2:** B04 (Red)
- **Band 3:** B05 (Red Edge 1)
- **Band 4:** B06 (Red Edge 2)
- **Band 5:** B07 (Red Edge 3)
- **Band 6:** B08 (NIR Broad)

## File Specifications
- **Format:** HDF5 (.h5)
- **Number of Patchs** `N=259150`
- **Patch Size:** 256 x 256 pixels
- **Data Types:** `int16` for images, `uint8` for masks
- **Optimization:** Data is chunked per patch `(1, channels, 256, 256)` and compressed using the `lzf` filter

## Dataset Structure

### 1. Image Modalities
The HDF5 file contains three primary image sources and two mask sources:

- `real/images` : PhiSat-2 real acquisitions. 8 bands. Shape `(N, 8, 256, 256)`. GRD : 4.75m
- `real/masks` : PhiSat-2 cloud masks (0=clear, 1=thick, 2=thin, 3=shadow). Shape `(N, 256, 256)`. GRD : 4.75m
- `sim/images` : PhiSat-2 simulated images derived from Sentinel-2. 8 bands aligned to real PhiSat-2 characteristics. Shape `(N, 8, 256, 256)`. GRD : 4.75m (from "Sentinel-2 -> Phisat-2" simulator)
- `s2b/images` : Sentinel-2 aligned data. 7 selected bands. Shape `(N, 7, 256, 256)`. GRD : 4.75m (Bicubic interpolation from 10m native resolution)
- `s2b/masks` : Sentinel-2 SCL masks (0 = No data, 1 = Saturated / Defective, 2 = Dark Area Pixels, 3 = Cloud Shadows, 4 = Vegetation, 5 = Bare Soils, 6 = Water, 7 = Clouds low probability / Unclassified, 8 = Clouds medium probability, 9 = Clouds high probability, 10 = Cirrus, 11 = Snow / Ice). Shape `(N, 256, 256)`. GRD : 4.75m (Nearest interpolation from 20m native resolution)

### 2. Metadata
Each patch is associated with detailed metadata located in the `metadata/` group. All metadata arrays have a length of `N`, where index `i` corresponds directly to the patch at index `i` in the image datasets.

- `product_id` (int32): Unique identifier for the parent image.
- `date_phi` (bytes/utf-8): Acquisition date of the PhiSat-2 image.
- `date_s2b` (bytes/utf-8): Acquisition date of the Sentinel-2 image.
- `center_lat`, `center_lon` (float32): Geographic center of the patch.
- `ul_lat`, `ul_lon` ... `ll_lat`, `ll_lon` (float32): Bounding box coordinates of the patch.
- `koppen_zone` (int8): Koppen-Geiger climate classification zone (values 1 to 30) for the patch center, based on the Beck et al. v3 (1991-2020) 1km resolution map.

## Normalization Statistics

The statistics below were calculated on the entire dataset after applying a non-linear transformation: `np.sqrt(np.maximum(img, 0))`. 

### 1. `real` (PhiSat-2)

The raw PhiSat-2 data shows saturated values up to 4095. After the square root transformation, a clipping at **38.729** ($\sqrt{1500}$, corresponding to the global 98th percentile) is recommended before applying the Z-score.

| Band | Min (raw) | Max (raw) | Mean | Std |
| :---: | :---: | :---: | :---: | :---: |
| **0** | 7.0 | 1083.0 | 15.0381 | 8.2196 |
| **1** | 5.0 | 4095.0 | 14.5305 | 10.6197 |
| **2** | 2.0 | 3887.0 | 14.4030 | 9.4811 |
| **3** | 1.0 | 4095.0 | 15.4191 | 9.0923 |
| **4** | 0.0 | 4095.0 | 13.6231 | 10.5712 |
| **5** | 0.0 | 4095.0 | 14.2143 | 10.4277 |
| **6** | 0.0 | 4095.0 | 14.7041 | 10.3784 |
| **7** | 0.0 | 4095.0 | 13.1745 | 9.7216 |

### 2. `sim` (Simulated Sentinel-2)

The simulated data contains processing artifacts generating negative values. After the `np.sqrt(np.maximum(img, 0))` transformation, a clipping at **100.0** is recommended before applying the Z-score.

| Band | Min (raw) | Max (raw) | Mean | Std |
| :---: | :---: | :---: | :---: | :---: |
| **0** | -395.0 | 28059.0 | 49.7866 | 7.2800 |
| **1** | -345.0 | 28499.0 | 49.0253 | 6.5203 |
| **2** | -380.0 | 28384.0 | 48.4297 | 6.9570 |
| **3** | -439.0 | 28662.0 | 49.2364 | 9.0981 |
| **4** | -433.0 | 28122.0 | 51.1648 | 8.3858 |
| **5** | -447.0 | 28146.0 | 55.4065 | 7.9555 |
| **6** | -465.0 | 28128.0 | 57.3572 | 8.3155 |
| **7** | -483.0 | 28473.0 | 56.7808 | 8.3664 |

### 3. `s2b` (Sentinel-2)

| Band | Min (raw) | Max (raw) | Mean | Std |
| :---: | :---: | :---: | :---: | :---: |
| **0 (B1)** | 0.0 | 31463.0 | 49.0215 | 6.5464 |
| **1 (B2)** | -32701.0 | 32724.0 | 48.4241 | 6.9918 |
| **2 (B3)** | 0.0 | 32218.0 | 49.2270 | 9.1444 |
| **3 (B4)** | 0.0 | 29106.0 | 51.1619 | 8.3999 |
| **4 (B5)** | 0.0 | 29044.0 | 55.4031 | 7.9740 |
| **5 (B6)** | 0.0 | 29031.0 | 57.3537 | 8.3373 |
| **6 (B7)** | -32690.0 | 32424.0 | 56.7685 | 8.4429 |



> (/!\ BAD_PRODUCT_IDS = [1296,1342,1385,1397,1420,1460,1497,1647,1854,2223,2246,2259,2373,2631,2640,2743,2834,2853,3374,3619,4071,4693,4813,4942,2352,2882,3322,3914,4702,1333,1466,1615,2460,2729,2763] : you have to filter patchs from this product_ids)


## TODO
- filter more products (inliers < K)
- divide images into smaller images before registration (avoid optical deformation)
- add worldcover labels
- add all the S2 bands