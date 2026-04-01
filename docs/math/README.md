# Mathematical Documentation

Visual explanations of the FlexViz 3D transformation pipeline, using the **h_shape** test board as a reference.

## Scripts

| Script | Topic | Figures |
|--------|-------|---------|
| `01_board_import.py` | Board data extraction: outline, traces, pads, fold markers | 6 |
| `02_region_segmentation.py` | Planar subdivision: cutting lines, regions, winding | 5 |
| `03_fold_recipes.py` | BFS recipe computation, classification, back-entry | 5 |
| `04_3d_transformation.py` | Cylindrical mapping, AFTER plane, multi-fold, normals | 6 |

## Running

```bash
cd docs/math
source ../../venv/bin/activate
python 01_board_import.py
python 02_region_segmentation.py
python 03_fold_recipes.py
python 04_3d_transformation.py
```

Figures are saved to `figures/`.

## Dependencies

- matplotlib >= 3.5
- numpy
- Plugin modules (loaded from `plugins/com_github_aightech_flexviz/`)
