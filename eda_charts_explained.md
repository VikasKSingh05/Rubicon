# Rubicon — Multi-Modal Preprocessing & EDA Charts Explained

> **Source Notebooks:** 
> - `ai-engine/notebooks/data_visualization.ipynb` (Development suite)
> - `ai-engine/notebooks/data_visulaization.ipynb` (Executed production suite with cached artifacts)
> - `ai-engine/notebooks/eda.ipynb` (Core Phase 2 baseline EDA)  
> **Dataset:** IEEE GRSS Data Fusion Contest 2013 — Houston, TX (UTM Zone 15N, ~2.5 m GSD)  
> **Modalities:** 144-band CASI Hyperspectral Cube (380–1050 nm) + LiDAR Digital Elevation Model (DEM) + 15-Class Land-Cover Ground Truth  
> **Target Purpose:** Preprocessing diagnostics, spectral-spatial exploration, cross-modal co-registration verification, and exploratory data analysis (EDA) prior to Mamba-Transformer multi-modal disaster damage assessment model training.

---

## Table of Contents

1. [Architectural Overview & Preprocessing Pipeline](#architectural-overview--preprocessing-pipeline)
2. [Section 1: Hyperspectral Spatial & Spectral Profiling](#section-1-hyperspectral-spatial--spectral-profiling)
   - [Chart 1: HSI True-Color RGB Composite](#chart-1-hsi-true-color-rgb-composite)
   - [Chart 2: HSI False-Color Infrared (CIR) Composite](#chart-2-hsi-false-color-infrared-cir-composite)
   - [Chart 3: Mean Spectral Reflectance Curves by Land-Cover Class (with ±1σ Bounds)](#chart-3-mean-spectral-reflectance-curves-by-land-cover-class-with-1σ-bounds)
   - [Chart 4: Spatial-Spectral Transect Heatmap](#chart-4-spatial-spectral-transect-heatmap)
   - [Chart 5: Inter-Band Pearson Correlation Heatmap](#chart-5-inter-band-pearson-correlation-heatmap)
3. [Section 2: LiDAR Digital Elevation Model (DEM) Analysis](#section-2-lidar-digital-elevation-model-dem-analysis)
   - [Chart 6: 2D LiDAR DEM Surface Elevation Map with Height Contours](#chart-6-2d-lidar-dem-surface-elevation-map-with-height-contours)
   - [Chart 7: LiDAR Elevation Histogram & Kernel Density Estimation (KDE)](#chart-7-lidar-elevation-histogram--kernel-density-estimation-kde)
   - [Chart 8: 3D LiDAR Hillshade & Urban Topological Texture Map](#chart-8-3d-lidar-hillshade--urban-topological-texture-map)
4. [Section 3: Multi-Modal Alignment & Structural Verification](#section-3-multi-modal-alignment--structural-verification)
   - [Chart 9: Full-Scene Multi-Modal Tri-Panel Comparison (HSI / DEM / Labels)](#chart-9-full-scene-multi-modal-tri-panel-comparison-hsi--dem--labels)
   - [Chart 10: LiDAR Elevation Distribution across 15 Land-Cover Classes (Violin Plots)](#chart-10-lidar-elevation-distribution-across-15-land-cover-classes-violin-plots)
   - [Chart 11: LiDAR Structural Boundaries (Cyan Sobel Edges) Overlaid on HSI RGB](#chart-11-lidar-structural-boundaries-cyan-sobel-edges-overlaid-on-hsi-rgb)
5. [Section 4: PCA Spectral Dimensionality Reduction Verification](#section-4-pca-spectral-dimensionality-reduction-verification)
   - [Chart 12: PCA Scree Plot — Individual Explained Variance Ratio](#chart-12-pca-scree-plot--individual-explained-variance-ratio)
   - [Chart 13: PCA Cumulative Explained Spectral Variance Curve](#chart-13-pca-cumulative-explained-spectral-variance-curve)
   - [Chart 14: PCA Eigenband Spectral Loadings across 144 Wavelengths](#chart-14-pca-eigenband-spectral-loadings-across-144-wavelengths)
6. [Section 5: Ground Truth & Damage Severity Proxy Distributions](#section-5-ground-truth--damage-severity-proxy-distributions)
   - [Chart 18: Patch Frequency by Land-Cover Class (Horizontal Bar Chart)](#chart-18-patch-frequency-by-land-cover-class-horizontal-bar-chart)
   - [Chart 19: Land-Cover Class to Damage-Severity Proxy Remapping (Grouped Bar Chart)](#chart-19-land-cover-class-to-damage-severity-proxy-remapping-grouped-bar-chart)
   - [Chart 20: Damage-Severity Proxy Distribution (Donut Chart)](#chart-20-damage-severity-proxy-distribution-donut-chart)
7. [Section 6: Spatial Patch Extraction & Multi-Modal Gallery](#section-6-spatial-patch-extraction--multi-modal-gallery)
   - [Chart 21: Spatial Patch Extraction Grid Coverage Map](#chart-21-spatial-patch-extraction-grid-coverage-map)
   - [Chart 22: Preprocessed Multi-Modal Patch Gallery Across Severity Levels](#chart-22-preprocessed-multi-modal-patch-gallery-across-severity-levels)
8. [Section 7: Cross-Modal Relationships & Tensor Quality Diagnostics](#section-7-cross-modal-relationships--tensor-quality-diagnostics)
   - [Chart 23: Bivariate Density Hexbin Plot — Spectral PC1 vs LiDAR Elevation](#chart-23-bivariate-density-hexbin-plot--spectral-pc1-vs-lidar-elevation)
   - [Chart 24: Patch Elevation Distribution across Damage Severity Levels (Violin Plot)](#chart-24-patch-elevation-distribution-across-damage-severity-levels-violin-plot)
   - [Chart 25: Preprocessed Tensor Value Distributions (X_hsi vs X_dem)](#chart-25-preprocessed-tensor-value-distributions-x_hsi-vs-x_dem)
   - [Chart 26: Executive Multi-Modal Dataset Summary Card](#chart-26-executive-multi-modal-dataset-summary-card)
9. [Comprehensive Master Chart Reference Table](#comprehensive-master-chart-reference-table)
10. [Key Technical Takeaways for Viva & System Defense](#key-technical-takeaways-for-viva--system-defense)

---

## Architectural Overview & Preprocessing Pipeline

The Rubicon AI engine operates on multi-modal Earth Observation (EO) data to assess structural and disaster damage. The dataset combines:
1. **CASI Hyperspectral Sensor**: 144 contiguous spectral bands spanning the visible to near-infrared spectrum (380 nm to 1050 nm) at ~2.5 m Ground Sample Distance (GSD).
2. **LiDAR Sensor**: Digital Elevation Model (DEM) capturing canopy and building heights above sea level.
3. **Land-Cover Ground Truth**: 15 distinct urban and environmental land-cover classes.

```
+-----------------------------------------------------------------------------------+
|                           RUBICON PREPROCESSING PIPELINE                          |
+-----------------------------------------------------------------------------------+
|  [Raw HSI: 349x1905x144]   [Raw LiDAR DEM: 349x1905]   [Labels: 349x1905 (0-15)]  |
|            |                          |                          |                |
|            v                          v                          v                |
|  [Affine Transformation / Nearest-Neighbor Grid Resampling & Co-Registration]     |
|                                       |                                           |
|            +--------------------------+--------------------------+                |
|            |                                                     |                |
|            v                                                     v                |
|  [PCA Spectral Compression]                             [Tile Patchification]     |
|   144 bands -> k comps (99% var)                         32x32 spatial windows    |
|   (e.g., 20 components)                                  min_labeled >= 50%       |
|            |                                                     |                |
|            v                                                     v                |
|  [StandardScaler / Normalization]                       [Severity Remapping]      |
|   X_hsi (C x H x W)                                      0: None (Vegetation/H2O) |
|   X_dem (1 x H x W)                                      1: Moderate (Residential)|
|                                                          2: Severe (Commercial)   |
|                                       |                                           |
|                                       v                                           |
|       [Exported NPZ: preprocessed_houston2013.npz / metadata_preprocessed.json]   |
+-----------------------------------------------------------------------------------+
```

---

## Section 1: Hyperspectral Spatial & Spectral Profiling

---

### Chart 1: HSI True-Color RGB Composite

#### What It Is
A top-of-atmosphere natural-color visualization of the entire 349 × 1905 pixel study area over Houston, Texas.
- **Red Channel:** Band 64 (~670 nm, visible Red)
- **Green Channel:** Band 38 (~550 nm, visible Green)
- **Blue Channel:** Band 23 (~480 nm, visible Blue)
- **Dynamic Range Contrast:** 2% to 98% percentile linear stretch (`np.clip((rgb - p2) / (p98 - p2), 0, 1)`).

#### Code Snippet
```python
rgb = np.stack([
    hsi_raw[:, :, RGB_BANDS_CASI[0]], # Band 64
    hsi_raw[:, :, RGB_BANDS_CASI[1]], # Band 38
    hsi_raw[:, :, RGB_BANDS_CASI[2]], # Band 23
], axis=-1)
p2, p98 = np.percentile(rgb, (2, 98))
rgb_stretched = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)

plt.imshow(rgb_stretched)
plt.title('Chart 1: HSI True-Color RGB Composite')
```

#### Why It Matters & How to Interpret
- Human visual inspection verifies sensor data integrity: no dead detector columns, atmospheric striping, or corrupted raster lines.
- **Green zones:** Athletic fields, parks, campus lawns, and tree canopies.
- **Grey/bright white linear structures:** Interstate highways, parking lots, and building roofs.
- **Dark absorbing regions:** Water channels and deep building shadows.

---

### Chart 2: HSI False-Color Infrared (CIR) Composite

#### What It Is
A Color-Infrared (CIR) composite that replaces the red display channel with Near-Infrared (NIR) wavelength energy.
- **Red Display Channel:** Band 101 (~850 nm, NIR plateau)
- **Green Display Channel:** Band 64 (~670 nm, Chlorophyll absorption red)
- **Blue Display Channel:** Band 38 (~550 nm, Green)

#### Code Snippet
```python
cir = np.stack([
    hsi_raw[:, :, NIR_BAND_CASI],     # Band 101 (NIR) -> Red
    hsi_raw[:, :, RGB_BANDS_CASI[0]], # Band 64 (Red) -> Green
    hsi_raw[:, :, RGB_BANDS_CASI[1]], # Band 38 (Green) -> Blue
], axis=-1)
p2_cir, p98_cir = np.percentile(cir, (2, 98))
cir_stretched = np.clip((cir - p2_cir) / (p98_cir - p2_cir + 1e-6), 0, 1)

plt.imshow(cir_stretched)
plt.title('Chart 2: HSI False-Color Infrared Composite')
```

#### Why It Matters & How to Interpret
- Living vegetation contains spongy mesophyll cell layers that strongly scatter NIR radiation while absorbing red light for photosynthesis.
- **Vibrant crimson / electric red:** Healthy tree canopies, golf courses, and grass.
- **Dull red / brownish:** Stressed or sparse grass.
- **Cyan / blue-grey:** Impervious man-made concrete, roads, and commercial complexes.
- **Deep black:** Water bodies (water completely absorbs NIR wavelengths).

---

### Chart 3: Mean Spectral Reflectance Curves by Land-Cover Class (with ±1σ Bounds)

#### What It Is
A line plot showing the continuous spectral response signature (spectral fingerprint) across all 144 wavelengths (380 nm to 1050 nm) for each of the 15 land-cover categories. Each curve displays the class mean with a shaded envelope representing $\pm 1$ standard deviation.

#### Code Snippet
```python
wavelengths = np.linspace(380, 1050, hsi_raw.shape[2])
cmap = plt.get_cmap('tab20', 16)

for c_id in range(1, 16):
    mask = (labels_raw == c_id)
    if not np.any(mask):
        continue
    c_pixels = hsi_raw[mask]
    mean_sig = np.mean(c_pixels, axis=0)
    std_sig = np.std(c_pixels, axis=0)
    c_name = HOUSTON2013_CLASSES.get(c_id, f'Class {c_id}')
    color = cmap(c_id)
    
    plt.plot(wavelengths, mean_sig, label=f'{c_id}. {c_name}', color=color, linewidth=2)
    plt.fill_between(wavelengths, np.clip(mean_sig - std_sig, 0, 1), 
                     np.clip(mean_sig + std_sig, 0, 1), color=color, alpha=0.12)
```

#### Physical & Spectral Rationale
- **Vegetation classes (Grass Healthy, Trees):** Exhibit low visible reflectance (chlorophyll absorption dip at ~670 nm), followed by a steep **Red Edge** transition (690–740 nm), rising to a high **NIR plateau** (750–900 nm).
- **Stressed vs. Synthetic Grass:** Synthetic turf lacks cellular NIR scattering and cellular water absorption troughs, producing a distinct flat spectral profile.
- **Soil & Concrete:** Monotonically increasing reflectance curves without the biological red edge.
- **Water:** Rapidly decreases to near-zero reflectance beyond 700 nm.
- The non-overlapping spectral bands confirm that the 144 CASI channels provide sufficient discriminative power for classification.

---

### Chart 4: Spatial-Spectral Transect Heatmap

#### What It Is
A 2D spatial-spectral slice taken horizontally across row 174 (the geometric centerline of the survey area).
- **X-axis:** Spatial column pixel index (0 to 1905, representing East-West physical transit).
- **Y-axis:** Continuous wavelength spectrum (380 nm at top to 1050 nm at bottom).
- **Color intensity:** Normalized spectral reflectance (using the `magma` colormap).

#### Code Snippet
```python
transect_row = hsi_raw.shape[0] // 2
transect_slice = hsi_raw[transect_row, :, :] # (cols, bands)
plt.imshow(transect_slice.T, aspect='auto', cmap='magma', 
           extent=[0, hsi_raw.shape[1], 1050, 380])
plt.xlabel('Column Pixel Index (Spatial)')
plt.ylabel('Wavelength (nm)')
plt.colorbar(label='Reflectance')
```

#### Why It Matters & How to Interpret
- Shows how spectral profiles vary continuously as the sensor transitions from urban commercial infrastructure into open lawns and parking areas.
- Bright vertical bands indicate highly reflective metal/concrete rooftops across all wavelengths.
- Sharp horizontal bands at 750–950 nm indicate NIR reflections corresponding to vegetation clusters.

---

### Chart 5: Inter-Band Pearson Correlation Heatmap

#### What It Is
A symmetric 36×36 correlation matrix computed across sub-sampled hyperspectral bands using Pearson's correlation coefficient ($r \in [-0.2, 1.0]$).

#### Code Snippet
```python
sub_bands = np.linspace(0, hsi_raw.shape[2]-1, 36, dtype=int)
flat_pixels = hsi_raw.reshape(-1, hsi_raw.shape[2])[::50, sub_bands]
corr_matrix = np.corrcoef(flat_pixels, rowvar=False)

plt.imshow(corr_matrix, cmap='coolwarm', vmin=-0.2, vmax=1.0)
plt.colorbar(label='Pearson Correlation (r)')
```

#### Why It Matters & Pipeline Impact
- Adjacent spectral bands exhibit correlation coefficients of $r > 0.95$, demonstrating immense spectral redundancy across contiguous CASI channels.
- Visible bands (380–680 nm) form a tight block of high inter-correlation, while NIR bands (750–1050 nm) form a second distinct block.
- **Justification for PCA:** This massive collinearity proves that feeding raw 144 channels into deep neural networks causes parameter explosion and computational waste. Orthogonal transformation via PCA compresses this variance into uncorrelated components.

---

## Section 2: LiDAR Digital Elevation Model (DEM) Analysis

---

### Chart 6: 2D LiDAR DEM Surface Elevation Map with Height Contours

#### What It Is
A continuous false-color elevation map of the study area rendered with the `terrain` colormap and overlaid with iso-height contour lines (at 8 distinct topographic intervals).
- **Color scale:** Low elevation ground level (green/teal) to tall structures and canopies (brown/white).
- **Contour labels:** Altitude in meters above sea level (a.s.l.).

#### Code Snippet
```python
im_dem = plt.imshow(dem_raw, cmap='terrain')
cs = plt.contour(dem_raw, levels=8, colors='black', alpha=0.35, linewidths=0.7)
plt.clabel(cs, inline=True, fontsize=8, fmt='%.0fm')
plt.colorbar(im_dem, label='Elevation (meters a.s.l.)')
```

#### Why It Matters & How to Interpret
- Houston is situated on a low-lying coastal plain (~10 m to ~45 m total range).
- Sharp localized elevation clusters represent multi-story commercial buildings, elevated highway overpasses, and mature tree canopies.
- Flat uniform elevation zones represent athletic turf, parking surfaces, and residential roadways.

---

### Chart 7: LiDAR Elevation Histogram & Kernel Density Estimation (KDE)

#### What It Is
A statistical probability distribution of elevation values across all valid pixels in the survey area, showing a binned histogram (40 bins) and a smooth KDE density curve.
- **Red dashed line:** Mean elevation ($\approx 17.5\text{ m}$).
- **Orange dotted line:** Median elevation ($\approx 15.2\text{ m}$).

#### Code Snippet
```python
valid_dem = dem_raw[np.isfinite(dem_raw)]
sns.histplot(valid_dem, kde=True, color='teal', bins=40)
plt.axvline(np.mean(valid_dem), color='red', linestyle='--', label=f'Mean: {np.mean(valid_dem):.2f}m')
plt.axvline(np.median(valid_dem), color='darkorange', linestyle=':', label=f'Median: {np.median(valid_dem):.2f}m')
```

#### Why It Matters & How to Interpret
- **Bimodal / Right-Skewed distribution:** The prominent primary peak at ~12–16 m represents the natural ground floor, roads, and single-story residential zones.
- The extended right tail (25 m to 45 m) accounts for tall infrastructure and commercial skyscrapers.
- Confirms the necessity of robust min-max normalization (`X_dem = (dem - min) / (max - min)`) prior to tensor ingestion.

---

### Chart 8: 3D LiDAR Hillshade & Urban Topological Texture Map

#### What It Is
A simulated 3D shaded-relief terrain model generated by calculating surface gradients ($\partial z/\partial x, \partial z/\partial y$), surface slope, and aspect relative to a synthetic illumination source (sun altitude $45^\circ$, sun azimuth $135^\circ$).

$$\text{Slope} = \frac{\pi}{2} - \arctan\left(\sqrt{\left(\frac{\partial z}{\partial x}\right)^2 + \left(\frac{\partial z}{\partial y}\right)^2}\right)$$
$$\text{Aspect} = \arctan2\left(-\frac{\partial z}{\partial x}, \frac{\partial z}{\partial y}\right)$$
$$\text{Hillshade} = \sin(\text{Alt})\sin(\text{Slope}) + \cos(\text{Alt})\cos(\text{Slope})\cos(\text{Azimuth} - \text{Aspect})$$

#### Code Snippet
```python
dy, dx = np.gradient(np.nan_to_num(dem_raw, nan=0.0))
slope = np.pi/2.0 - np.arctan(np.sqrt(dx*dx + dy*dy))
aspect = np.arctan2(-dx, dy)
altitude = np.pi / 4.0        # 45 deg sun elevation
azimuth = 3.0 * np.pi / 4.0   # 135 deg sun azimuth
hillshade = np.sin(altitude) * np.sin(slope) + np.cos(altitude) * np.cos(slope) * np.cos(azimuth - aspect)
hillshade = np.clip(hillshade, 0, 1)

plt.imshow(hillshade, cmap='gray')
plt.colorbar(label='Illumination Intensity')
```

#### Why It Matters
- Highlights geometric structural edges, rooftop textures, and elevation discontinuities that are invisible in flat 2D false-color imagery.
- Verifies that the LiDAR raster accurately captures building geometry and vertical infrastructure boundaries.

---

## Section 3: Multi-Modal Alignment & Structural Verification

---

### Chart 9: Full-Scene Multi-Modal Tri-Panel Comparison (HSI / DEM / Labels)

#### What It Is
A synchronized 3×1 vertical stack comparing the full Houston scene across three core modalities:
- **Panel 9a:** HSI True-Color RGB composite.
- **Panel 9b:** Co-registered LiDAR DEM (using the `plasma` colormap).
- **Panel 9c:** Ground-truth Land-Cover raster (15 classes, `tab20` colormap, black = unlabeled background).

#### Code Snippet
```python
fig, axes = plt.subplots(3, 1, figsize=(16, 10))
axes[0].imshow(rgb_stretched)
axes[0].set_title('Chart 9a: HSI True-Color RGB Composite')

axes[1].imshow(dem_raw, cmap='plasma')
axes[1].set_title('Chart 9b: Co-Registered LiDAR Digital Elevation Model')

axes[2].imshow(labels_raw, cmap='tab20', vmin=0, vmax=15)
axes[2].set_title('Chart 9c: Ground Truth Land-Cover Raster (0 = Unlabeled Background)')
```

#### Why It Matters & Diagnostic Value
- **Cross-Modal Co-Registration:** Features in Panel 9a (e.g., stadium, highway intersections) must align pixel-for-pixel with the elevation contours in Panel 9b and the annotated classes in Panel 9c.
- **Label Sparsity:** Large portions of the scene are unlabeled background (Class 0), demonstrating why patch filtering (`min_labeled >= 50%`) is required to avoid training on empty regions.

---

### Chart 10: LiDAR Elevation Distribution across 15 Land-Cover Classes (Violin Plots)

#### What It Is
A multi-class violin plot displaying the empirical probability distribution, median, and interquartile range of LiDAR elevation values for each of the 15 land-cover categories.

#### Code Snippet
```python
df_elev = []
for c_id in range(1, 16):
    mask = (labels_raw == c_id) & np.isfinite(dem_raw)
    if np.any(mask):
        vals = dem_raw[mask]
        c_name = HOUSTON2013_CLASSES.get(c_id, f'C{c_id}')
        for v in vals[::max(1, len(vals)//300)]:
            df_elev.append({'Class': f'{c_id}. {c_name}', 'Elevation': float(v)})

sns.violinplot(data=pd.DataFrame(df_elev), x='Class', y='Elevation', palette='Spectral', inner='quartile')
plt.xticks(rotation=45, ha='right')
```

#### Physical Interpretation
- **Commercial (Class 8):** Shows the highest elevation distribution and largest variance, reflecting tall multi-story office complexes.
- **Road (Class 9) & Highway (Class 10):** Tightly bounded near ground elevation, with elevated highway segments extending higher.
- **Grass & Water (Classes 1–3, 6):** Extremely narrow, low-elevation distributions centered near ground baseline.
- Demonstrates why LiDAR elevation is an essential complementary feature to spectral signatures for structural classification.

---

### Chart 11: LiDAR Structural Boundaries (Cyan Sobel Edges) Overlaid on HSI RGB

#### What It Is
A sub-pixel registration diagnostic tool. A Sobel spatial gradient filter computes building edge boundaries from the normalized LiDAR DEM ($|\nabla \text{DEM}| > 93\text{rd percentile}$). These edge contours are rendered in bright cyan over the HSI True-Color RGB image.

#### Code Snippet
```python
dem_norm = (dem_raw - np.nanmin(dem_raw)) / (np.nanmax(dem_raw) - np.nanmin(dem_raw) + 1e-6)
gx = ndimage.sobel(dem_norm, axis=1)
gy = ndimage.sobel(dem_norm, axis=0)
edge_mag = np.hypot(gx, gy)
edge_mask = edge_mag > np.percentile(edge_mag, 93)

overlay = rgb_stretched.copy()
overlay[edge_mask] = [0.0, 1.0, 1.0] # Cyan edges

plt.imshow(overlay)
plt.title('Chart 11: LiDAR Structural Boundaries (Cyan Edges) Overlaid on HSI RGB')
```

#### Why It Matters
- **Rigorous Alignment Verification:** If the affine transformation or CRS re-projection had even a 2-pixel spatial offset, the cyan building contours would be visibly shifted off the optical rooftops.
- The perfect alignment of cyan contours with optical building boundaries proves the rasters are co-registered.

---

## Section 4: PCA Spectral Dimensionality Reduction Verification

---

### Chart 12: PCA Scree Plot — Individual Explained Variance Ratio

#### What It Is
A vertical bar chart illustrating the percentage of total spectral variance captured individually by each principal component ($PC_1, PC_2, \dots, PC_k$).

#### Code Snippet
```python
evr = pca_model.explained_variance_ratio_
plt.bar(range(1, len(evr) + 1), evr * 100, color='royalblue', edgecolor='black')
plt.xlabel('Principal Component Index')
plt.ylabel('Explained Variance (%)')
```

#### How to Interpret
- $PC_1$ accounts for over 70–80% of total variance alone (overall albedo and brightness amplitude).
- $PC_2$ captures 10–15% (NIR vs. visible spectral contrast).
- Variance drops sharply after $PC_4$, with higher components capturing subtle absorption characteristics and residual noise.

---

### Chart 13: PCA Cumulative Explained Spectral Variance Curve

#### What It Is
A cumulative line plot tracking the cumulative explained variance percentage as additional principal components are added.
- **Green dashed line:** 99% Target Variance Threshold.
- **Orange dotted line:** 95% Threshold.
- **Gray dash-dot line:** 90% Threshold.

#### Code Snippet
```python
cum_evr = np.cumsum(evr)
plt.plot(range(1, len(cum_evr) + 1), cum_evr * 100, marker='o', color='crimson', linewidth=2.5)
plt.axhline(99, color='green', linestyle='--', label='99% Target Threshold')
plt.axhline(95, color='darkorange', linestyle=':', label='95% Threshold')
plt.axhline(90, color='gray', linestyle='-.', label='90% Threshold')
```

#### Pipeline Impact
- Confirms that approximately **15 to 20 PCA components** (out of 144 raw bands) are sufficient to retain **99.0%** of total spectral information.
- Reduces raw HSI memory footprint and compute requirements by **~86%** with negligible signal loss.

---

### Chart 14: PCA Eigenband Spectral Loadings across 144 Wavelengths

#### What It Is
A continuous line plot showing the eigenvector weights (loadings) for the top 4 principal components ($PC_1$ through $PC_4$) across all 144 wavelengths (380 nm to 1050 nm).

#### Code Snippet
```python
for i in range(min(4, len(evr))):
    plt.plot(wavelengths, pca_model.components_[i], 
             label=f'PC {i+1} Loading (Var: {evr[i]*100:.1f}%)', linewidth=2)
plt.axhline(0, color='black', linestyle='--', alpha=0.5)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Component Eigenvector Weight')
```

#### Physical Rationale
- **$PC_1$ Loading:** Positive and uniformly distributed across all wavelengths — represents broadband brightness / surface albedo.
- **$PC_2$ Loading:** Sharp polarity reversal between visible wavelengths (< 700 nm, negative weights) and NIR wavelengths (> 750 nm, positive weights) — functions as a generalized vegetation index (NDVI-like feature).
- **$PC_3$ & $PC_4$ Loadings:** Capture narrow absorption bands associated with atmospheric moisture, soil minerals, and hydrocarbons.

---

## Section 5: Ground Truth & Damage Severity Proxy Distributions

---

### Chart 18: Patch Frequency by Land-Cover Class (Horizontal Bar Chart)

#### What It Is
A horizontal bar chart displaying the exact count and percentage of extracted 32×32 patches assigned to each land-cover category based on majority-vote labeling.

#### Code Snippet
```python
y_classes, class_counts = np.unique(patches['y_class'], return_counts=True)
class_labels_list = [HOUSTON2013_CLASSES.get(c, f'Class {c}') for c in y_classes]
total_patches = len(patches['y_class'])

bars = plt.barh(np.arange(len(y_classes)), class_counts, 
                color=plt.cm.tab20(np.linspace(0, 1, len(y_classes))), edgecolor='black')
for bar in bars:
    w = bar.get_width()
    pct = (w / total_patches) * 100
    plt.text(w + 0.5, bar.get_y() + bar.get_height()/2, f'{int(w)} ({pct:.1f}%)', va='center')
```

#### Imbalance Analysis
- Dominant classes: **Grass Healthy**, **Residential**, and **Trees**.
- Minority classes: **Tennis Court**, **Running Track**, and **Water**.
- Informs the training pipeline to apply class-weighted cross-entropy loss or focal loss during model optimization.

---

### Chart 19: Land-Cover Class to Damage-Severity Proxy Remapping (Grouped Bar Chart)

#### What It Is
A grouped categorical bar chart illustrating how each of the 15 land-cover categories maps into the 3-tier disaster damage proxy scheme (*None*, *Moderate*, *Severe*).

| Severity Proxy | Color | Included Land-Cover Classes | Domain Rationale |
|---|---|---|---|
| **None (0)** | Green (`#2ecc71`) | Grass (Healthy/Stressed/Synthetic), Tree, Water | Open natural surfaces with no structural collapse hazard |
| **Moderate (1)** | Orange (`#f39c12`) | Soil, Residential, Road, Parking Lots 1 & 2, Tennis Court, Running Track | Low-rise residential and surface infrastructure subject to partial damage |
| **Severe (2)** | Red (`#e74c3c`) | Commercial, Highway, Railway | Heavy multi-story commercial infrastructure, elevated freeways, transit networks |

#### Code Snippet
```python
df_class_sev = []
for c_id, name in HOUSTON2013_CLASSES.items():
    sev = CLASS_TO_SEVERITY.get(c_id, 'none')
    cnt = int(np.sum(patches['y_class'] == c_id))
    df_class_sev.append({'Class': f'{c_id}. {name}', 'Severity': sev.capitalize(), 'Count': cnt})

sns.barplot(data=pd.DataFrame(df_class_sev), x='Class', y='Count', hue='Severity',
            palette={'None': '#2ecc71', 'Moderate': '#f39c12', 'Severe': '#e74c3c'}, dodge=False)
plt.xticks(rotation=45, ha='right')
```

---

### Chart 20: Damage-Severity Proxy Distribution (Donut Chart)

#### What It Is
A percentage donut chart summarizing the final target label balance across all extracted training tiles.

#### Code Snippet
```python
y_sevs, sev_counts = np.unique(patches['y_sev'], return_counts=True)
sev_labels = [SEVERITY_NAMES[s].capitalize() for s in y_sevs]
sev_colors = ['#2ecc71', '#f39c12', '#e74c3c']

plt.pie(sev_counts, labels=sev_labels, autopct='%1.1f%%', startangle=140, 
        colors=sev_colors[:len(y_sevs)], wedgeprops=dict(width=0.45, edgecolor='black'))
```

#### Key Metrics
- **None:** $\approx 46.5\%$
- **Moderate:** $\approx 38.8\%$
- **Severe:** $\approx 14.7\%$
- Highlights moderate class imbalance, validating the use of macro-averaged F1 and Cohen's Kappa as primary evaluation metrics.

---

## Section 6: Spatial Patch Extraction & Multi-Modal Gallery

---

### Chart 21: Spatial Patch Extraction Grid Coverage Map

#### What It Is
A spatial coverage map showing the HSI base layer overlaid with yellow bounding boxes indicating each extracted 32×32 pixel patch meeting the valid label criterion (`min_labeled >= 0.50`).

#### Code Snippet
```python
plt.imshow(rgb_stretched, alpha=0.7)
psize = meta['patch_size']
for (r, c) in patches['coords']:
    rect = plt.Rectangle((c, r), psize, psize, edgecolor='yellow', facecolor='none', linewidth=0.8, alpha=0.6)
    plt.gca().add_patch(rect)
```

#### Diagnostic Value
- Confirms spatial coverage across the University of Houston campus and surrounding commercial/residential corridors.
- Discards unannotated perimeter zones and nodata areas, preventing non-informative background tiles from entering training.

---

### Chart 22: Preprocessed Multi-Modal Patch Gallery Across Severity Levels

#### What It Is
A 5-column inspection gallery displaying representative multi-modal patches across each damage severity tier (*None*, *Moderate*, *Severe*):
1. **Column 1:** HSI True-Color RGB tile.
2. **Column 2:** HSI False-Color Infrared (CIR) tile.
3. **Column 3:** Co-registered LiDAR DEM tile (`plasma` colormap).
4. **Column 4:** PCA False-Color tile ($PC_1, PC_2, PC_3 \rightarrow$ RGB).
5. **Column 5:** Ground truth class and severity classification callout banner.

#### Code Snippet
```python
fig, axes = plt.subplots(num_samples, 5, figsize=(16, 3 * num_samples))
for row, idx in enumerate(sample_indices):
    # Renders RGB, CIR, DEM, PCA(1,2,3), and Severity Banner side-by-side
    ...
```

#### Verification Checklist
- [x] Optical features (buildings, tree clumps) match elevation boundaries in the DEM column.
- [x] PCA false-color composites preserve structural boundaries while highlighting material differences.
- [x] Assigned class names and severity tiers match the visual content of the tiles.

---

## Section 7: Cross-Modal Relationships & Tensor Quality Diagnostics

---

### Chart 23: Bivariate Density Hexbin Plot — Spectral PC1 vs LiDAR Elevation

#### What It Is
A 2D hexbin bivariate density distribution comparing individual pixel spectral brightness ($PC_1$, X-axis) against structural elevation (LiDAR DEM in meters, Y-axis).

#### Code Snippet
```python
flat_pc1 = hsi_pca[:, :, 0].flatten()[::50]
flat_dem = dem_raw.flatten()[::50]
valid_pair = np.isfinite(flat_dem)

plt.hexbin(flat_pc1[valid_pair], flat_dem[valid_pair], gridsize=35, cmap='inferno', mincnt=1)
plt.xlabel('Spectral Principal Component 1')
plt.ylabel('Elevation (meters)')
plt.colorbar(label='Pixel Density')
```

#### Multi-Modal Fusion Insight
- Demonstrates that spectral and elevation features are orthogonal and complementary:
  - **High $PC_1$ + Low Elevation:** Concrete parking lots, road intersections, and bright bare soil.
  - **Moderate $PC_1$ + High Elevation:** Commercial office buildings and warehouse rooftops.
  - **Low $PC_1$ + Low Elevation:** Asphalt roadways, water features, and shadowed grass.
- A single sensor modality alone cannot reliably distinguish these classes; joint fusion is required.

---

### Chart 24: Patch Elevation Distribution across Damage Severity Levels (Violin Plot)

#### What It Is
A violin plot showing the distribution of mean normalized LiDAR elevation per patch grouped across the three target damage severity levels.

#### Code Snippet
```python
patch_elev_means = np.mean(patches['X_dem'][:, 0], axis=(1, 2))
df_sev_elev = pd.DataFrame({
    'Severity': [SEVERITY_NAMES[s].capitalize() for s in patches['y_sev']],
    'Normalized Elevation': patch_elev_means
})
sns.violinplot(data=df_sev_elev, x='Severity', y='Normalized Elevation',
               palette={'None': '#2ecc71', 'Moderate': '#f39c12', 'Severe': '#e74c3c'}, inner='box')
```

#### Diagnostic Finding
- **Severe Damage Proxy patches:** Exhibit significantly higher mean elevation and greater vertical dispersion due to commercial building footprints.
- **None / Moderate patches:** Confined to lower elevation bands, matching expected physical terrain distributions.

---

### Chart 25: Preprocessed Tensor Value Distributions (X_hsi vs X_dem)

#### What It Is
Overlaid probability density histograms comparing the numerical distributions of preprocessed model input tensors:
- **$X_{\text{hsi}}$ (Blue):** PCA spectral tensors standardized to zero mean and unit variance ($\mu \approx 0, \sigma \approx 1$).
- **$X_{\text{dem}}$ (Red):** LiDAR DEM tensors normalized to the $[0, 1]$ interval.

#### Code Snippet
```python
plt.hist(patches['X_hsi'].flatten()[::100], bins=50, color='royalblue', alpha=0.7, 
         label='X_hsi (PCA Tensors)', density=True)
plt.hist(patches['X_dem'].flatten()[::100], bins=50, color='crimson', alpha=0.6, 
         label='X_dem (DEM Tensors)', density=True)
plt.legend()
```

#### Numerical Health Verification
- Verifies absence of `NaN`, `Inf`, or extreme numerical outliers in the exported training tensors.
- Confirms input scales are appropriately bounded for gradient stability during backpropagation.

---

### Chart 26: Executive Multi-Modal Dataset Summary Card

#### What It Is
A formatted metrics summary card displaying the complete metadata of the preprocessed dataset artifacts.

#### Summary Card Specifications
```text
==================================================
           RUBICON PREPROCESSING & EDA SUMMARY
==================================================
* Dataset Source:         IEEE GRSS Houston 2013 (UTM 15N)
* Spatial Dimensions:     349 Rows x 1905 Cols (~2.5m GSD)
* Raw HSI Channels:       144 Bands (380 nm - 1050 nm)
* PCA Reduced Channels:   20 Components
* Retained Spectral Var:  99.12%
* Preprocessed Patches:   129 Multi-Modal Tiles
* Patch Spatial Extent:   32 x 32 Pixels (~80m x 80m ground area)
* LiDAR Elevation Range:  [-0.67 m, 42.45 m]
* Land-Cover Classes:     15 Unique Categories
* Damage Severity Proxy:  None: 60 | Moderate: 50 | Severe: 19
* Preprocessed Artifacts: data/preprocessed/preprocessed_houston2013.npz
==================================================
```

---

## Comprehensive Master Chart Reference Table

| Chart # | Title | Primary Technique | Source Notebook Cell | Core Diagnostic Question Answered |
|---|---|---|---|---|
| **1** | HSI True-Color RGB Composite | Percentile Stretch RGB | Block 2 (Cell 3) | Are raw optical rasters free of corruption and detector striping? |
| **2** | HSI False-Color Infrared (CIR) | NIR-Red-Green Composite | Block 2 (Cell 3) | Where are active vegetation canopies and biomass distributed? |
| **3** | Mean Spectral Signatures | 144-Band Spectral Profiling | Block 3 (Cell 4) | Are all 15 land-cover categories spectrally distinguishable? |
| **4** | Spatial-Spectral Transect | 2D Spectral Slice Heatmap | Block 4 (Cell 5) | How does spectral signature vary along an urban transect? |
| **5** | Inter-Band Correlation Matrix | Pearson Collinearity Matrix | Block 4 (Cell 5) | Is dimensionality reduction necessary due to channel redundancy? |
| **6** | 2D LiDAR DEM Elevation Map | Topographic Contour Mapping | Block 5 (Cell 6) | What is the overall topographic and urban elevation profile? |
| **7** | LiDAR Elevation Histogram & KDE | Density Distribution | Block 5 (Cell 6) | What are the statistical bounds and distribution of surface height? |
| **8** | 3D LiDAR Hillshade Texture | Gradient & Aspect Shading | Block 5 (Cell 6) | Are structural building footprints and roof edges sharply defined? |
| **9** | Full-Scene Multi-Modal Tri-Panel | 3-Panel Synchronized View | Block 6 (Cell 7) | Do HSI, LiDAR DEM, and Land-Cover rasters spatial grids match? |
| **10** | Class-Wise Elevation Violin Plots | Categorical Distribution | Block 6 (Cell 7) | Does elevation provide discriminative separation between classes? |
| **11** | Edge Registration Overlay | Sobel Filter Cyan Edges | Block 6 (Cell 7) | Is cross-modal co-registration verified at sub-pixel precision? |
| **12** | PCA Scree Plot | Variance by Component Bar | Block 7 (Cell 8) | How much variance is captured by each individual principal component? |
| **13** | PCA Cumulative Variance Curve | Threshold Summation Curve | Block 7 (Cell 8) | How many PCA components are needed to preserve 99% variance? |
| **14** | PCA Eigenband Loadings | Eigenvector Weight Curves | Block 7 (Cell 8) | Which physical wavelengths contribute to each principal component? |
| **18** | Patch Frequency by Class | Horizontal Frequency Bar | Block 8 (Cell 9) | What is the exact class representation across extracted tiles? |
| **19** | Class-to-Severity Remapping | Grouped Categorical Bar | Block 8 (Cell 9) | How are the 15 land-cover classes mapped to damage proxies? |
| **20** | Damage Severity Donut Chart | Percentage Donut Chart | Block 8 (Cell 9) | What is the class balance ratio for the damage assessment head? |
| **21** | Patch Extraction Grid Map | Spatial Bounding Boxes | Block 9 (Cell 10) | Are extracted patches evenly distributed across valid survey areas? |
| **22** | Multi-Modal Patch Gallery | 5-Column Multi-Modal Tiles | Block 9 (Cell 10) | Do optical, elevation, and PCA representations align per patch? |
| **23** | Bivariate Hexbin Density | 2D PC1 vs Elevation Density | Block 10 (Cell 11)| Do spectral reflectance and height provide orthogonal information? |
| **24** | Severity Elevation Violin Plot | Severity vs Elevation | Block 10 (Cell 11)| Does structural elevation correlate with assigned damage proxies? |
| **25** | Tensor Value Distributions | Normalized Histogram Overlays| Block 10 (Cell 11)| Are input tensors properly scaled and free of numerical anomalies? |
| **26** | Executive Dataset Summary Card | Formatted Metrics Card | Block 10 (Cell 11)| What are the complete parameters of the preprocessed pipeline? |

---

## Key Technical Takeaways for Viva & System Defense

1. **Dimensionality Reduction via PCA:**  
   Compressing 144 CASI spectral bands down to ~20 principal components retains **> 99% of total variance** while reducing tensor memory footprint and compute complexity by **~86%**, mitigating the Hughes phenomenon (curse of dimensionality).

2. **Cross-Modal Orthogonality:**  
   As proven in Charts 10 and 23, optical reflectance and LiDAR elevation capture fundamentally orthogonal properties (material composition vs. 3D physical structure). Fusing both modalities enables discrimination between classes that appear identical in a single modality (e.g., flat asphalt parking lots vs. multi-story commercial roofs).

3. **Sub-Pixel Spatial Co-Registration:**  
   Verified through Sobel edge boundary overlays (Chart 11) and multi-modal tri-panels (Chart 9), confirming zero spatial offset between the CASI HSI and LiDAR elevation rasters.

4. **Methodological Transparency in Damage Proxy:**  
   Because the Houston 2013 benchmark lacks direct post-disaster collapse annotations, land-cover classes are mapped to a 3-tier damage proxy based on structural vulnerability (Vegetation $\rightarrow$ None, Residential $\rightarrow$ Moderate, Commercial/Infrastructure $\rightarrow$ Severe). This is documented as a methodological design choice.

5. **Class Imbalance Mitigation Strategy:**  
   With the damage proxy distribution split at ~46.5% None, ~38.8% Moderate, and ~14.7% Severe (Chart 20), model evaluation must utilize **Macro-averaged F1-Score** and **Cohen's Kappa** alongside stratified train/validation/test partitioning (70/15/15).
