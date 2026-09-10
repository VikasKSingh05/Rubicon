"""Phase 2 preprocessing: HSI/LiDAR loading, alignment, PCA, and patchification.

Houston 2013 (IEEE GRSS DFC) has no damage labels, so land-cover classes are
repurposed as a **damage-severity proxy**. This is an explicit, acknowledged
simplification documented for the report:
  - none     -> vegetated / water (minimal structural damage exposure)
  - moderate -> soil, residential, roads, courts (partial damage)
  - severe   -> commercial, highway, railway (high collapse exposure)
"""

HOUSTON2013_CLASSES = {
    1: "Grass healthy",
    2: "Grass stressed",
    3: "Grass synthetic",
    4: "Tree",
    5: "Soil",
    6: "Water",
    7: "Residential",
    8: "Commercial",
    9: "Road",
    10: "Highway",
    11: "Railway",
    12: "Parking Lot 1",
    13: "Parking Lot 2",
    14: "Tennis Court",
    15: "Running Track",
}

# Land-cover -> severity proxy (acknowledged simplification).
CLASS_TO_SEVERITY = {
    1: "none",       # Grass healthy
    2: "none",       # Grass stressed
    3: "none",       # Grass synthetic
    4: "none",       # Tree
    5: "moderate",   # Soil
    6: "none",       # Water
    7: "moderate",   # Residential
    8: "severe",     # Commercial
    9: "moderate",   # Road
    10: "severe",    # Highway
    11: "severe",    # Railway
    12: "moderate",  # Parking Lot 1
    13: "moderate",  # Parking Lot 2
    14: "moderate",  # Tennis Court
    15: "moderate",  # Running Track
}

SEVERITY_NAMES = ["none", "moderate", "severe"]
SEVERITY_TO_INDEX = {name: i for i, name in enumerate(SEVERITY_NAMES)}

__all__ = [
    "HOUSTON2013_CLASSES",
    "CLASS_TO_SEVERITY",
    "SEVERITY_NAMES",
    "SEVERITY_TO_INDEX",
]