# FlexViz Improvement Plan

## Status Summary (2026-03-07)

### Completed
- [x] Phase 0: Adaptive arc tessellation (`kicad_parser.py`) — 2mm max segment length
- [x] Phase 0: Adaptive circle tessellation (`geometry.py`) — ~60% vertex reduction
- [x] Variable/formula support for fold angles (`markers.py`)
- [x] Variable display labels in viewer UI (`viewer.py`)
- [x] Pure Python STEP writer (`step_writer.py`) — no build123d dependency
- [x] STEP export with exact B-Rep geometry (`step_export.py`)
- [x] 3D component model export (STEP embed + WRL tessellation)
- [x] Stiffener support in STEP export
- [x] Traces/pads unchecked by default

### Completed (Phase 1)
- [x] Trace placement bug fixes (boundary fallback, adaptive subdivision, normal consistency)
- [x] Pad rendering fixes (per-vertex region lookup for pads crossing folds)
- [x] Z-fighting fix (increased z_offset: traces 0.05mm, pads 0.08mm)
- [x] 20 unit tests + 8-scene visual validation

### Not Started
- [ ] Phase 2: Separate mesh layers (viewer performance)
- [ ] Phase 3: Region lookup acceleration (spatial index)
- [ ] Phase 4: STEP export electronics (traces/pads as copper geometry)
- [ ] Phase 5: Export button UI consolidation

---

## Current Issues

### Issue 1: Slow GUI — Full mesh rebuild on every toggle
**Severity: HIGH** | **Impact: Every user interaction**

`on_display_option_changed()` → `update_mesh()` → `create_board_geometry_mesh()` rebuilds
ALL geometry (board + traces + pads + stiffeners + components) even when toggling a single
checkbox. For a complex board (44 folds, 115 arcs), each toggle takes seconds.

**Root cause:** Single monolithic `Mesh` object compiled into one OpenGL display list.
No concept of layers or cached sub-meshes.

### Issue 2: Trace/pad drawing bugs
**Severity: HIGH** | **Impact: Visual correctness**

Three sub-issues:
1. **Boundary region lookup failure** — When a trace subdivision point falls exactly on a
   fold zone boundary, `find_containing_region()` returns `None`. The point gets an empty
   recipe (stays flat) while neighbors fold → visible discontinuity.
2. **Traces crossing narrow fold zones** — Fixed 20 subdivisions may skip a narrow fold
   zone entirely, causing the trace to "jump" from one side to the other.
3. **Pads use single region for all vertices** — A pad straddling a fold boundary uses
   one region's recipe for all vertices → the pad doesn't follow the bend.

### Issue 3: STEP export missing electronics
**Severity: MEDIUM** | **Impact: Export quality**

The STEP exporter only writes the board shape (outline + bends + stiffeners). Copper
traces and pads are not included. Component 3D models work but may fail for certain
model formats or placements.

---

## Implementation Phases

### Phase 1: Fix trace/pad rendering (correctness first)
**Goal:** Traces and pads render at correct 3D positions across all fold configurations.

#### 1A: Region lookup fallback for traces
**File:** `mesh.py` — `create_trace_mesh()` (line 970-978)

When `find_containing_region()` returns `None`, fall back to the previous valid region:
```python
last_valid_region_1 = None
last_valid_region_2 = None
for i in range(subdivisions + 1):
    ...
    if regions:
        containing_region_1 = find_containing_region(p1, regions)
        if containing_region_1:
            last_valid_region_1 = containing_region_1
        elif last_valid_region_1:
            containing_region_1 = last_valid_region_1
        # same for p2
```

#### 1B: Adaptive trace subdivision at fold boundaries
**File:** `mesh.py` — `create_trace_mesh()` (line 964)

Replace fixed `range(subdivisions + 1)` with t-values that include fold zone crossings:
```python
t_values = [i / subdivisions for i in range(subdivisions + 1)]
if regions and markers:
    for marker in markers:
        t_cross = _compute_fold_crossing_t(v0, v1, v3, v2, marker)
        t_values.extend(t_cross)
    t_values = sorted(set(t_values))
```

#### 1C: Per-vertex region lookup for pads
**File:** `mesh.py` — `create_pad_mesh()` (line 1091-1102)

Instead of using one region for the entire pad, look up per-vertex:
```python
for v in outer_2d:
    v_region = find_containing_region(v, regions) or containing_region
    v_recipe = get_region_recipe(v_region) if v_region else region_recipe
    v3d, normal = transform_point_and_normal(v, v_recipe)
    ...
```

#### 1D: Normal consistency check
**File:** `mesh.py` — after computing normals in trace/pad mesh

```python
if last_normal and dot(n, last_normal) < 0:
    n = (-n[0], -n[1], -n[2])
```

**Validation checkpoint:**
- [x] Test: single fold + trace crossing it → trace follows the bend smoothly
- [x] Test: two folds + trace crossing both → no jumps or discontinuities
- [x] Test: pad straddling a fold boundary → pad wraps around the bend
- [x] Visual: 8 scenes saved to `tests/visual/phase1_traces/` (diagonal, multi-fold, pads, front/back layers, dense, accordion, transparent, real PCB)
- [x] Z-fighting fix: traces 0.05mm, pads 0.08mm above surface
- [x] 240 tests pass (20 new + 220 existing)

---

### Phase 2: Separate mesh layers (performance)
**Goal:** Toggling checkboxes is instant. Only fold angle changes trigger rebuilds.

#### 2A: Mesh layer architecture
**File:** `viewer.py` — `GLCanvas` class

Replace single `self.mesh` + `self.display_list` with layer dict:
```python
self._layers = {
    'board': {'mesh': None, 'display_list': None, 'visible': True},
    'traces': {'mesh': None, 'display_list': None, 'visible': False},
    'pads': {'mesh': None, 'display_list': None, 'visible': False},
    'stiffeners': {'mesh': None, 'display_list': None, 'visible': True},
    'components': {'mesh': None, 'display_list': None, 'visible': False},
}
```

Each layer gets its own display list. `on_paint()` iterates visible layers:
```python
for name, layer in self._layers.items():
    if layer['visible'] and layer['display_list']:
        glCallList(layer['display_list'])
```

#### 2B: Split mesh generation
**File:** `mesh.py` — `create_board_geometry_mesh()`

Split into independent functions that return separate meshes:
- `create_board_mesh_with_regions()` — already exists, returns board mesh
- `create_all_traces_mesh(traces, regions, thickness)` — new, returns trace mesh
- `create_all_pads_mesh(pads, regions, thickness)` — new, returns pad mesh
- `create_all_stiffeners_mesh(...)` — new
- `create_all_components_mesh(...)` — new

#### 2C: Display toggle = visibility only
**File:** `viewer.py` — `on_display_option_changed()`

```python
def on_display_option_changed(self, event):
    self.canvas.set_layer_visible('traces', self.cb_traces.GetValue())
    self.canvas.set_layer_visible('pads', self.cb_pads.GetValue())
    self.canvas.set_layer_visible('stiffeners', self.cb_stiffeners.GetValue())
    self.canvas.set_layer_visible('components', self.cb_components.GetValue())
    self.canvas.Refresh()  # no rebuild!
```

#### 2D: Lazy first-build for trace/pad layers
Since traces/pads are unchecked by default, don't build their meshes until first enabled:
```python
def set_layer_visible(self, name, visible):
    layer = self._layers[name]
    layer['visible'] = visible
    if visible and layer['mesh'] is None:
        layer['mesh'] = self._build_layer_mesh(name)
        self._compile_layer_display_list(name)
```

**Validation checkpoint:**
- [ ] Toggle traces on/off → instant (< 50ms), no mesh rebuild
- [ ] Toggle pads on/off → instant
- [ ] Change fold angle → all layers rebuild correctly
- [ ] Measure: time `update_mesh()` before and after with neubondV4
- [ ] Visual: save toggle comparison to `tests/visual/phase2_layers/`

---

### Phase 3: Region lookup acceleration
**Goal:** Trace mesh generation 10-100x faster for complex boards.

#### 3A: Bounding box pre-filter
**File:** `planar_subdivision.py` — `find_containing_region()`

```python
def find_containing_region(point, regions):
    for region in regions:
        if not _point_in_bbox(point, region._bbox):
            continue  # fast reject
        if point_in_polygon(point, region.outline):
            return region
    return None
```

Precompute `region._bbox = (min_x, min_y, max_x, max_y)` once after subdivision.

#### 3B: Grid spatial index (if bbox is insufficient)
Build a cell grid (5mm cells). Each cell stores references to overlapping regions.

```python
class RegionGrid:
    def __init__(self, regions, cell_size=5.0):
        # compute grid, assign regions to cells by bbox overlap
    def find(self, point):
        cell = self._cell_for(point)
        for region in self._grid[cell]:
            if point_in_polygon(point, region.outline):
                return region
```

**Validation checkpoint:**
- [ ] Benchmark: neubondV4 trace mesh time before/after
- [ ] Same visual output (diff traces screenshot before vs after)
- [ ] Unit test: `find_containing_region()` returns same results with/without grid

---

### Phase 4: STEP export electronics
**Goal:** STEP files include copper traces and pads as thin extruded solids.

#### 4A: Trace geometry in STEP
**File:** `step_export.py`

For each trace, create a thin extruded solid (copper thickness ~0.035mm):
- Convert trace to ribbon polygon (line_segment_to_ribbon)
- Find containing region, get fold recipe
- Transform to 3D
- Build flat solid via `step_writer.build_flat_solid()`

Group all copper on one layer into a single STEP body: `COPPER_F` / `COPPER_B`.

#### 4B: Pad geometry in STEP
Similar to traces but with pad polygons. Through-hole pads need annular ring geometry.

#### 4C: Component model placement fixes
- Verify STEP model embedding with rotated/mirrored components
- Test WRL fallback tessellation quality

**Validation checkpoint:**
- [ ] Export simple board with traces → open in FreeCAD → copper visible
- [ ] Export board with pads → annular rings visible
- [ ] Export board with 3D models → models positioned correctly
- [ ] Visual: save FreeCAD screenshots to `tests/visual/phase4_step/`

---

### Phase 5: UI polish and export consolidation
**Goal:** Professional UI matching Altium-level polish.

#### 5A: Export buttons in single "Export" StaticBox
```python
export_box = wx.StaticBox(control_panel, label="Export")
export_sizer = wx.StaticBoxSizer(export_box, wx.VERTICAL)
# OBJ, STL, STEP buttons in one row
```

#### 5B: Progress feedback during mesh generation
Show busy cursor or progress bar during long operations (trace mesh, STEP export).

#### 5C: Keyboard shortcuts
- Space: toggle bend on/off
- T: toggle traces
- P: toggle pads
- W: toggle wireframe
- R: reset camera

---

## Visual Validation Framework

All phases produce visual artifacts saved to `tests/visual/<phase>/` for regression testing.

### Directory structure
```
tests/visual/
  phase1_traces/
    trace_single_fold_before.png
    trace_single_fold_after.png
    pad_on_fold_boundary.png
  phase2_layers/
    toggle_traces_on.png
    toggle_traces_off.png
    timing_report.txt
  phase3_spatial/
    neubondv4_traces_before.png
    neubondv4_traces_after.png
    benchmark_results.txt
  phase4_step/
    freecad_traces.png
    freecad_pads.png
    freecad_components.png
  reference/
    altium_comparison.png
```

### Headless visual testing approach
Since the viewer uses wxPython + OpenGL (hard to automate), use PyVista for offline
rendering of the same mesh data:

```python
# tests/visual_test_helper.py
import pyvista as pv

def render_mesh_to_image(mesh, filename, camera_position='iso'):
    """Render a Mesh object to a PNG file using PyVista."""
    plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
    # Convert our Mesh to PyVista PolyData
    faces = []
    for face in mesh.faces:
        faces.extend([len(face)] + list(face))
    pd = pv.PolyData(mesh.vertices, faces)
    plotter.add_mesh(pd, color='green', show_edges=True)
    plotter.camera_position = camera_position
    plotter.screenshot(filename)
    plotter.close()
```

This allows automated visual regression without needing a display server.

---

## Priority Order

| Priority | Phase | Effort | Impact |
|----------|-------|--------|--------|
| 1 | Phase 1: Fix trace/pad rendering | 1-2 days | Correctness |
| 2 | Phase 2: Separate mesh layers | 2-3 days | Performance (10x toggle speed) |
| 3 | Phase 3: Spatial index | 1 day | Performance (10-100x trace gen) |
| 4 | Phase 4: STEP electronics | 2-3 days | Export quality |
| 5 | Phase 5: UI polish | 1 day | UX |

**Total estimated: 7-10 days for all phases.**

---

## Long-term: Altium Viewer Parity

To match Altium's 3D viewer experience, the following additional features would be needed
beyond the phases above:

| Feature | Altium | FlexViz Current | Gap |
|---------|--------|-----------------|-----|
| Instant layer toggle | Yes | Full rebuild (slow) | Phase 2 |
| Smooth trace rendering | Yes | Buggy near folds | Phase 1 |
| STEP with copper | Yes | Board only | Phase 4 |
| GPU-accelerated rendering | Yes (DirectX) | OpenGL immediate mode | Future: switch to VBOs or modernGL |
| Texture-mapped silkscreen | Yes | Not implemented | Future |
| Board stackup visualization | Yes | Single-layer PCB | Future |
| Real-time fold animation | Partial | Slider-based (rebuilds) | Phase 2 enables this |
| Anti-aliased edges | Yes | No AA | Future: MSAA in GLCanvas |
| Component hover/selection | Yes | Not implemented | Future |
| Cross-section view | Yes | Not implemented | Future |
