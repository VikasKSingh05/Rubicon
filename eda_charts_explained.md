# EDA Visualization Charts — Detailed Explanation

> **Source Notebook:** `ai-engine/notebooks/eda.ipynb`  
> **Dataset:** IEEE GRSS Data Fusion Contest 2013 — Houston, TX  
> **Purpose:** Exploratory Data Analysis (EDA) to understand the hyperspectral + LiDAR dataset before training the Mamba-Transformer model.

---

## Table of Contents

1. [Chart 1: PCA Cumulative Explained Variance Curve](#chart-1-pca-cumulative-explained-variance-curve)
2. [Chart 2: Land-Cover Class Pixel Distribution (Bar Chart)](#chart-2-land-cover-class-pixel-distribution-bar-chart)
3. [Chart 3: Severity-Proxy Pixel Distribution (Bar Chart)](#chart-3-severity-proxy-pixel-distribution-bar-chart)
4. [Chart 4: Sample Patch Grid (HSI False-Color + DEM Overlay)](#chart-4-sample-patch-grid-hsi-false-color--dem-overlay)
5. [Chart 5: Full-Scene Triptych (HSI / DEM / Labels)](#chart-5-full-scene-triptych-hsi--dem--labels)

---

## Chart 1: PCA Cumulative Explained Variance Curve

### What It Is
A **line plot** showing the cumulative explained variance ratio as a function of the number of PCA (Principal Component Analysis) components. It has:
- **X-axis:** Number of PCA components (1 to 144, since the HSI has 144 spectral bands)
- **Y-axis:** Cumulative explained variance (0.0 to 1.0)
- **Black dashed horizontal line** at `y = 0.99` (the 99% variance threshold)
- **Red dotted vertical line** at the component count `k99` where 99% variance is first reached

### Code That Produces It (Cell 3 / Block 3)
```python
from sklearn.decomposition import PCA

pix = hsi.reshape(-1, hsi.shape[2])          # flatten spatial dims → (N_pixels, 144)
pca_all = PCA().fit(pix)                      # fit PCA on all 144 components
cum = np.cumsum(pca_all.explained_variance_ratio_)  # cumulative sum
k99 = int(np.searchsorted(cum, 0.99) + 1)    # find where 99% is reached

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(np.arange(1, len(cum) + 1), cum)      # the curve
ax.axhline(0.99, ls="--", c="k", lw=0.8)      # 99% threshold line
ax.axvline(k99, ls=":", c="r", lw=0.8)        # component cutoff line
ax.set_xlabel("PCA components")
ax.set_ylabel("cumulative explained variance")
print(f"components needed for 99% variance: {k99}")
```

### Why It Matters
The hyperspectral image has **144 spectral bands**, which is extremely high-dimensional. Many of these bands are **highly correlated** (neighboring wavelengths capture similar information). PCA identifies the **principal directions of variance** in the data and ranks them by importance.

This chart answers a critical question: **How many components do we actually need to retain 99% of the spectral information?**

### How to Interpret It
- The curve typically rises **very steeply** at first (the first few components capture the most variance) and then **flattens out** (remaining components add negligible information).
- Where the **red vertical line** intersects the x-axis tells you the number of components needed. For the Houston 2013 dataset, this is typically around **15–30 components** (out of 144).
- This means we can **reduce dimensionality by ~80-90%** with almost no information loss.

### Significance in the Pipeline
- The value `k99` is used in the preprocessing pipeline to compress the 144-band HSI cube down to `k99` PCA components.
- This dramatically **reduces memory usage, training time, and overfitting risk** for the Mamba-Transformer model.
- The `pca_variance: 0.99` setting in `configs/phase2.yaml` controls this threshold.

---

## Chart 2: Land-Cover Class Pixel Distribution (Bar Chart)

### What It Is
A **vertical bar chart** (left panel of a 1×2 subplot) showing the **number of pixels** belonging to each of the 15 land-cover classes in the ground truth. Each bar is labeled with the class name on the x-axis, and the bar height represents pixel count.

### Code That Produces It (Cell 4 / Block 4)
```python
from collections import Counter
from src.preprocessing import HOUSTON2013_CLASSES

cnt = Counter(labels.ravel())    # count pixels per class
cnt.pop(0, None)                 # remove background (class 0)

class_ids = sorted(cnt)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(
    [HOUSTON2013_CLASSES[c] for c in class_ids],  # class names as x-labels
    [cnt[c] for c in class_ids]                    # pixel counts as bar heights
)
axes[0].tick_params(axis="x", rotation=45, labelsize=8)
axes[0].set_title("land-cover class pixels")
```

### The 15 Land-Cover Classes
| Class ID | Name | Severity Proxy |
|----------|------|----------------|
| 1 | Grass healthy | none |
| 2 | Grass stressed | none |
| 3 | Grass synthetic | none |
| 4 | Tree | none |
| 5 | Soil | moderate |
| 6 | Water | none |
| 7 | Residential | moderate |
| 8 | Commercial | severe |
| 9 | Road | moderate |
| 10 | Highway | severe |
| 11 | Railway | severe |
| 12 | Parking Lot 1 | moderate |
| 13 | Parking Lot 2 | moderate |
| 14 | Tennis Court | moderate |
| 15 | Running Track | moderate |

### Why It Matters
This chart reveals **class imbalance** — a critical concern in any classification task. If certain classes have far fewer pixels than others, the model may:
- **Underperform** on minority classes (low recall)
- **Overfit** to majority classes
- Produce **biased** severity predictions

### How to Interpret It
- **Tall bars** = dominant classes (many pixels available for training)
- **Short bars** = rare classes (risk of poor model performance)
- Classes like "Grass healthy" and "Tree" typically dominate, while "Tennis Court" and "Running Track" are underrepresented.
- This imbalance directly affects the severity proxy distribution (Chart 3).

### What to Look For
- Is the distribution **skewed**? → May need stratified sampling or class weighting during training
- Are any classes **extremely rare** (< 1% of total)? → May need data augmentation
- Does the distribution make **geographic sense**? → Houston is a city, so residential/road classes should be well-represented

---

## Chart 3: Severity-Proxy Pixel Distribution (Bar Chart)

### What It Is
A **color-coded vertical bar chart** (right panel of the same 1×2 subplot) showing the total number of pixels in each severity category after mapping land-cover classes to severity proxies. It has exactly 3 bars:
- **Green bar** (`#22C55E`): `none` severity
- **Amber bar** (`#F59E0B`): `moderate` severity  
- **Red bar** (`#DC2626`): `severe` severity

### Code That Produces It (Cell 4 / Block 4 — same cell as Chart 2)
```python
sev_counts = {s: 0 for s in SEVERITY_NAMES}   # {"none": 0, "moderate": 0, "severe": 0}
for c, n in cnt.items():
    sev_counts[CLASS_TO_SEVERITY[c]] += n       # aggregate by severity

axes[1].bar(
    sev_counts.keys(),
    sev_counts.values(),
    color=["#22C55E", "#F59E0B", "#DC2626"]     # green, amber, red
)
axes[1].set_title("severity-proxy pixels")
print("severity pixel counts:", sev_counts)
```

### The Severity Mapping Logic
Since Houston 2013 has **no actual damage labels**, the project uses a domain-knowledge-based proxy:

| Severity | Source Classes | Rationale |
|----------|---------------|-----------|
| **none** | Grass (healthy/stressed/synthetic), Tree, Water | Vegetated and water areas have minimal structural exposure — unlikely to show building damage |
| **moderate** | Soil, Residential, Road, Parking Lots, Tennis Court, Running Track | Habitable or partially-built structures — partial damage potential |
| **severe** | Commercial, Highway, Railway | Large hard-built infrastructure — highest collapse/destruction exposure |

### Why It Matters
This is the **target distribution** for the Mamba-Transformer model's output head. The model predicts one of these 3 severity levels for each image patch. If this distribution is heavily imbalanced:
- The model might predict "none" for everything (if "none" dominates)
- Loss functions may need **class weights** to compensate
- Evaluation metrics should use **macro-averaged F1** rather than accuracy

### How to Interpret It
- Ideally, bars should be **roughly balanced** for good model training
- In reality, "none" (vegetation) often dominates in the Houston dataset
- The print statement `severity pixel counts: {...}` gives the exact numbers
- A 3:1 or higher imbalance ratio suggests stratified sampling is critical

### Important Note (Acknowledged Simplification)
This proxy is an **explicit simplification** — real disaster damage assessment would use actual damage labels (e.g., from FEMA or post-disaster surveys). This mapping is documented as a methodological limitation.

---

## Chart 4: Sample Patch Grid (HSI False-Color + DEM Overlay)

### What It Is
A **2×3 grid of image patches** (6 randomly selected patches), where each patch shows:
- **Base layer:** HSI false-color RGB composite (bands 63, 37, 22 ≈ Red, Green, Blue)
- **Overlay:** LiDAR DEM (Digital Elevation Model) height map with `terrain` colormap at 45% transparency
- **Title:** Patch index, ground-truth class ID, and mapped severity label

### Code That Produces It (Cell 5 / Block 5)
```python
patches = patch_dataset(
    hsi, dem_a, labels,
    patch_size=CFG["pipeline"]["patch_size"],     # 32×32 pixels
    overlap=CFG["pipeline"]["overlap"],            # 0.0 (no overlap)
    min_labeled=CFG["pipeline"]["min_labeled"],    # 0.5 (50% labeled minimum)
)
print("patches:", patches["hsi"].shape)

rng = np.random.default_rng(0)                    # reproducible random seed
picks = rng.choice(len(patches["hsi"]), 6, replace=False)  # pick 6 random patches

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for ax, i in zip(axes.ravel(), picks):
    rgb = np.stack([patches["hsi"][i, :, :, b] for b in RGB_BANDS], axis=-1)
    ax.imshow(np.clip(rgb, 0, 1))                 # HSI false-color base
    ax.imshow(patches["dem"][i], cmap="terrain", alpha=0.45)  # DEM overlay
    yc = patches["y_class"][i]
    ax.set_title(f"patch {i} | class {yc} -> {CLASS_TO_SEVERITY[yc]}", fontsize=8)
plt.tight_layout()
```

### Understanding the False-Color Composite
The 144-band CASI sensor captures wavelengths far beyond human vision. To visualize:
- **Band 63 → Red channel** (~670 nm — visible red)
- **Band 37 → Green channel** (~550 nm — visible green)
- **Band 22 → Blue channel** (~480 nm — visible blue)

These approximate a natural-color view, but with enhanced spectral contrast from the hyperspectral sensor.

### Understanding the DEM Overlay
- The **terrain colormap** shows height: low areas (blue-green) → high areas (brown-white)
- At **45% opacity**, it blends with the HSI, showing both spectral and elevation information simultaneously
- This is exactly how the model "sees" the data: it fuses HSI spectral features with LiDAR height features

### Why It Matters
This is a **visual sanity check** that verifies:
1. **Patching is correct** — patches look like coherent image regions, not garbage
2. **Alignment worked** — the DEM overlay matches the HSI (buildings/roads in HSI align with height features in DEM)
3. **Labels make sense** — the class label shown in each title should match what you see (e.g., "Tree" patches should look green/vegetated)
4. **Data quality** — no missing data, no corrupted regions, reasonable dynamic range

### How to Interpret It
- **Green/vegetated patches** labeled "none" → ✅ Correct
- **Building patches** labeled "severe" or "moderate" → ✅ Correct
- **Water patches** should look dark/uniform → ✅ Correct
- If labels don't match the visual appearance → 🚨 Data alignment or labeling issue

---

## Chart 5: Full-Scene Triptych (HSI / DEM / Labels)

### What It Is
A **1×3 panel** showing the **entire scene** (full Houston 2013 study area) in three different representations side by side:
1. **Left panel:** HSI false-colour composite (the entire hyperspectral image rendered as approximate RGB)
2. **Middle panel:** LiDAR DEM (the full height map with terrain colormap)
3. **Right panel:** Land-cover ground truth labels (with `tab20` discrete colormap, 0–15)

### Code That Produces It (Cell 6 / Block 6)
```python
comp = np.stack([hsi[:, :, b] for b in RGB_BANDS], axis=-1)
comp = (comp - comp.min()) / max(comp.max() - comp.min(), 1e-9)  # normalize to [0, 1]

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
axes[0].imshow(comp)
axes[0].set_title("HSI false-colour")
axes[1].imshow(dem_a, cmap="terrain")
axes[1].set_title("LiDAR DEM")
axes[2].imshow(labels, cmap="tab20", vmin=0, vmax=15)
axes[2].set_title("land-cover labels")
plt.tight_layout()
```

### Panel-by-Panel Breakdown

#### Panel 1: HSI False-Colour
- Shows the **349 × 1905 pixel** study area as a natural-looking image
- Normalization: `(pixel - min) / (max - min)` ensures full [0, 1] dynamic range
- **Green areas** = vegetation (grass, trees)
- **Gray/white areas** = buildings, concrete, roads
- **Dark blue areas** = water bodies
- The image covers the **University of Houston campus and surrounding neighborhoods**

#### Panel 2: LiDAR DEM
- Shows **elevation/height** at every pixel using the `terrain` colormap
- **Low areas** (ground level): blue-green tones
- **Tall structures** (buildings, trees): brown/white tones
- Useful for identifying: building footprints, tree canopy, flat roads vs. elevated highways
- The DEM has been **aligned/resampled** to match the HSI's spatial grid exactly (done by the `align_rasters` function)

#### Panel 3: Land-Cover Labels
- Shows the **ground truth** — a discrete classification of every pixel into one of 15 classes (or 0 = background/unlabeled)
- Uses `tab20` colormap (20 distinct colors) with range `vmin=0, vmax=15`
- Each color corresponds to a class ID (1–15) from the `HOUSTON2013_CLASSES` dictionary
- **Black/dark areas** = class 0 = background (unlabeled pixels, not used in training)
- This is the "answer key" that the model learns from

### Why It Matters
This triptych is the **most important visualization** in the entire EDA because it shows:

1. **Spatial context:** You can see the geographic layout — where the university campus is, where the roads are, where commercial districts are
2. **Co-registration quality:** If the DEM and HSI are properly aligned, the spatial features should match across panels (buildings in the same locations)
3. **Label coverage:** How much of the scene is labeled vs. background — only labeled areas contribute to training
4. **Data modalities together:** Understanding how the two sensor inputs (HSI spectral, LiDAR height) complement each other — HSI captures material/spectral properties, LiDAR captures 3D structure

### How to Interpret It
- **Cross-reference all three panels**: A building should appear as a gray/white structure in HSI (Panel 1), a tall object in DEM (Panel 2), and colored as "Commercial" or "Residential" in labels (Panel 3)
- **Look for misalignment**: If the DEM features are shifted relative to the HSI features, the `align_rasters` function has a bug
- **Check label density**: If most of the image is class 0 (background), the usable training data is limited
- **Compare scale**: Both HSI and DEM should have the same pixel dimensions (349 × 1905 after alignment)

---

## Summary Table

| Chart # | Type | Cell | What It Shows | Key Question Answered |
|---------|------|------|---------------|----------------------|
| 1 | Line plot | 3 | PCA cumulative variance | How many spectral components are needed? |
| 2 | Bar chart | 4 (left) | Per-class pixel counts | Is the dataset class-balanced? |
| 3 | Bar chart | 4 (right) | Per-severity pixel counts | Is the severity distribution balanced? |
| 4 | Image grid | 5 | 6 random HSI+DEM patches | Are patches, alignment & labels sane? |
| 5 | Triptych | 6 | Full scene: HSI, DEM, labels | Does the complete data look correct? |

---

## Key Takeaways for Viva / Interview

1. **PCA reduces 144 bands to ~15-30 components** with only 1% information loss — this is critical for making the model trainable.

2. **Class imbalance exists** — vegetation dominates, severe-class pixels (Commercial, Highway, Railway) are fewer. This is addressed via **stratified train/val/test splits** (70/15/15) in the pipeline.

3. **The severity mapping is an acknowledged simplification** — Houston 2013 has no real damage labels, so land-cover is used as a proxy. This is a methodological choice, not a dataset feature.

4. **Dual-sensor fusion is validated visually** — the triptych and patch overlays confirm that HSI (spectral) and LiDAR (elevation) data are properly co-registered and complementary.

5. **Patches are 32×32 pixels** with no overlap and 50% minimum labeled-pixel threshold — this discards mostly-background patches that would add noise to training.
