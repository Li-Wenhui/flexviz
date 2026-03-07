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
- [x] Bend subdivision alignment: trace t-values match board mesh facets via `num_bend_subdivisions`
- [x] Back-entry AFTER rotation fix: separated tangent angle (π−angle) from cumulative angle (−angle)
  in `bend_transform.py`, fixing -90° fold discontinuities in multi-fold configurations
- [x] 33 unit tests + 10-scene visual validation (h_shape PCB with 8 mixed-angle folds)
- [x] All 253 tests passing

### Completed (Phase 1.2)
- [x] Oval drill parsing: `(drill oval X Y)` now correctly parsed (was returning 0)
- [x] `*.Cu` wildcard layer expansion for through-hole pads
- [x] Improved region fallback for through-hole components (drill hole avoidance via
  offset probes, pad position fallback for 3D model and box mesh placement)
- [x] Per-vertex region lookup for component boxes and 3D models
- [x] Component bounding box rotation: bbox now applies footprint angle when computing
  pad positions, fixing displaced component boxes for rotated through-hole footprints
- [x] 7 new through-hole-specific tests
- [x] All 260 tests passing

### Completed (Phase 2)
- [x] Separate mesh layers: each display layer (board, traces, pads, components, 3D models, stiffeners) gets its own mesh and OpenGL display list
- [x] `create_board_layer_meshes()` in mesh.py generates all layers at once, sharing region computation
- [x] Toggling checkboxes (Show Traces, Show Pads, etc.) is now instant visibility toggle — no mesh rebuild
- [x] Lazy display list compilation: OpenGL display lists built on first visibility enable
- [x] OBJ/STL export merges visible layers on demand via `get_visible_mesh()`
- [x] 6 new tests, all 266 tests passing

### Not Started
- [ ] Phase 3: Region lookup acceleration (spatial index)
- [ ] Phase 4: STEP export electronics (traces/pads as copper geometry)
- [ ] Phase 5: Export button UI consolidation

---

## Current Issues

### ~~Issue 1: Slow GUI — Full mesh rebuild on every toggle~~ RESOLVED
**Status: FIXED** in Phase 2

Toggling display checkboxes now only changes OpenGL layer visibility — no mesh rebuild.
All layers are pre-built during `update_mesh()` (fold angle changes, refresh, settings).
Display lists are lazily compiled on first visibility enable.

### ~~Issue 2: Trace/pad drawing bugs~~ RESOLVED
**Status: FIXED** in Phase 1

All three sub-issues resolved:
1. ~~Boundary region lookup failure~~ → fallback to last valid region
2. ~~Traces crossing narrow fold zones~~ → adaptive subdivision at fold boundaries
3. ~~Pads use single region~~ → per-vertex region lookup
4. **Additional fix:** Back-entry AFTER rotation used wrong cumulative angle (π−angle
   instead of −angle), causing 12mm+ discontinuities for -90° folds in multi-fold boards.
   Fixed by separating tangent angle from cumulative rotation angle in `bend_transform.py`.

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
- [x] Visual: 10 scenes saved to `tests/visual/phase1_traces/` (includes h_shape PCB with 8 mixed-angle folds)
- [x] Z-fighting fix: traces 0.05mm, pads 0.08mm above surface
- [x] Bend subdivision alignment: trace quads match board mesh facets
- [x] Back-entry -90° fold continuity: all 8 folds pass max_jump < 0.15mm
- [x] 253 tests pass (33 new + 220 existing)

---

### Phase 1.2: Through-hole part display fixes
**Goal:** Through-hole pads and 3D models render at correct positions.

#### Bugs fixed

**1A: Oval drill parsing** (`kicad_parser.py` line 1134)
`(drill oval 0.8 1.7)` was parsed as `drill=0` because `get_float(0)` tried to parse
the string `"oval"`. Fixed to detect `"oval"` keyword and use `max(dim1, dim2)`.

**1B: `*.Cu` wildcard layer expansion** (`geometry.py` line 322)
Through-hole pads have `layers=["*.Cu", "*.Mask"]`. The layer check only matched exact
`"F.Cu"` / `"B.Cu"`. Added `*.Cu` wildcard expansion so through-hole pads are correctly
assigned to F.Cu or B.Cu.

**1C: Component center in drill hole** (`mesh.py`)
For through-hole components where a pad is at (0,0) relative to footprint, the component
center falls in a drill hole cutout → region lookup returns None. Improved fallback:
1. Try small offsets (0.5mm) around center to escape drill hole
2. Try bounding box corners
3. Try pad positions and pad polygon vertices
Applied to both `create_component_mesh` and `create_component_3d_model_mesh`.

**1D: Pad center in drill hole** (`mesh.py` `create_pad_mesh`)
Same drill hole escape: try offsets around pad center based on drill radius before
falling back to polygon vertex iteration.

**1E: Component per-vertex region lookup** (`mesh.py`)
`create_component_mesh` and `create_component_3d_model_mesh` used a SINGLE region
recipe for ALL vertices, causing components spanning fold boundaries to be misplaced.
Fixed by adding per-vertex region lookup: each vertex finds its own region and uses
that region's fold recipe. Falls back to the component center's region recipe when
no region found. This matches the approach already used in `create_pad_mesh`.

**1F: Component bounding box not rotated** (`geometry.py` line 293)
`pad_xs = [fp.at_x + p.at_x ...]` computed pad positions WITHOUT applying footprint
rotation angle, causing component bounding boxes for rotated footprints (90°, -90°)
to be at completely wrong positions. All pads fell outside the bbox for the audio jack
(-90°) and 2/6 pads were outside for the USB connector (90°). Fixed by applying the
same rotation transform used for pad center computation.

**Validation checkpoint:**
- [x] Oval drill pads now have correct drill diameter (h_shape: 1.7mm, neubondV4: 1.7mm)
- [x] Through-hole pads on F.Cu footprints correctly assigned F.Cu layer
- [x] Pad center in drill hole → fallback finds correct region
- [x] 3D pad-to-trace distance preserved after bend transform
- [x] Component boxes follow fold surface via per-vertex region lookup
- [x] Rotated component bboxes contain all pad positions
- [x] 260 tests pass (7 new)

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
