# FlexViz Mathematical Documentation

Visual and mathematical explanation of the full 3D flex PCB bending pipeline, using the **H-shape test board** (`tests/test_data/h_shape.kicad_pcb`) as a running example: 8 fold markers with mixed angles (45, -90, 100, 50 degrees), 7 traces, 14 pads.

## Pipeline Overview

```
 .kicad_pcb file
       |
       v
 [1. Board Import]          Parse S-expressions, extract outline,
       |                     traces, pads, fold markers from layers
       v
 [2. Region Segmentation]   Cut the flat board into regions using
       |                     fold marker lines (planar subdivision)
       v
 [3. Fold Recipes]          BFS from anchor region assigns each
       |                     region a recipe: which folds affect it,
       |                     in what order, from which side
       v
 [4. 3D Transformation]     Map each region's 2D vertices to 3D:
       |                     cylindrical bending in fold zones,
       |                     rotated plane continuation after
       v
  Mesh / STEP export
```

Each stage is documented in detail with annotated plots:

| Stage | Document | Figures | Key Concepts |
|-------|----------|---------|--------------|
| 1 | [Board Import](01_board_import.md) | 6 | Outline, layers, traces, pads, fold marker anatomy, R = w/\|&theta;\| |
| 2 | [Region Segmentation](02_region_segmentation.md) | 5 | Cutting lines, planar graph, face tracing, winding order |
| 3 | [Fold Recipes](03_fold_recipes.md) | 5 | BFS traversal, recipe buildup, classification, back-entry |
| 4 | [3D Transformation](04_3d_transformation.md) | 6 | Cylindrical mapping, AFTER plane, continuity, multi-fold 3D |

## Generating the Figures

```bash
cd docs/math
source ../../venv/bin/activate
python 01_board_import.py
python 02_region_segmentation.py
python 03_fold_recipes.py
python 04_3d_transformation.py
```

Figures are saved to `figures/` (22 PNG files at 300 DPI).

## Dependencies

- Python 3.10+
- matplotlib >= 3.5
- numpy
- Plugin modules (auto-loaded from `plugins/com_github_aightech_flexviz/`)
