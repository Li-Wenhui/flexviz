# STEP Export Rework Plan

## Goal

Replace the current build123d/OpenCASCADE-based STEP export with a **zero-dependency
pure-Python STEP writer** that generates exact parametric geometry (PLANE +
CYLINDRICAL_SURFACE) directly from the board regions and fold definitions.

This eliminates the need for `step_venv`, the CLI workaround, and the ~500MB
build123d dependency. STEP export will work directly inside the KiCad plugin.

---

## Current State

`step_export.py` (~1400 lines) uses build123d + OCP to:
1. Split board into regions via `planar_subdivision`
2. Create OCC solids per region (extrude flat, sweep bends)
3. Fuse with `BRepAlgoAPI_Fuse`
4. Export via `STEPCAFControl_Writer`

Problems:
- Requires separate venv (`step_venv`) due to KiCad Python conflicts
- CLI-only workflow (`step_export_cli.py`)
- Complex fallback chains (3 export paths)
- ~500MB dependency for a text file writer

---

## Key Insight

A flex PCB STEP file only needs two surface types:

| Surface              | STEP Entity             | Used For                     |
|----------------------|-------------------------|------------------------------|
| Flat plane           | `PLANE`                 | Top/bottom/sides of flat regions |
| Cylindrical arc      | `CYLINDRICAL_SURFACE`   | Inner/outer faces of bends   |

Everything else (edges, vertices, topology) is straight lines and circular arcs.
We already compute all of this analytically in `bend_transform.py`.

---

## Architecture

### Data Flow

```
BoardGeometry + FoldMarkers
    │
    ▼
planar_subdivision.split_board_into_regions(num_bend_subdivisions=1)
    │
    ▼
List[Region]  ←── each has: outline (2D polygon), holes, fold_recipe
    │
    ▼
For each Region, build 3D solid:
    ├── FLAT region  → extruded prism (6+ planar faces)
    └── BEND region  → cylindrical shell section (curved + planar faces)
    │
    ▼
StepWriter assembles B-Rep entities with shared edges
    │
    ▼
Write ASCII STEP AP214 file
```

### Planar Subdivision (existing — no changes needed)

With `num_bend_subdivisions=1`, `split_board_into_regions()` already produces
exactly the right decomposition:

- **One region per flat section** (not subdivided further)
- **One region per bend zone** (single region between the two fold boundary lines)
- **fold_recipe** correctly classifies each: empty or all-AFTER = flat, contains IN_ZONE = bend

Example with 2 folds on a rectangular board:
```
Region 0: FLAT    fold_recipe = []
Region 1: BEND    fold_recipe = [(fold1, "IN_ZONE")]
Region 2: FLAT    fold_recipe = [(fold1, "AFTER")]
Region 3: BEND    fold_recipe = [(fold1, "AFTER"), (fold2, "IN_ZONE")]
Region 4: FLAT    fold_recipe = [(fold1, "AFTER"), (fold2, "AFTER")]
```

### StepWriter Class

A pure-Python builder that manages STEP entity IDs and cross-references.

```python
class StepWriter:
    """Builds and writes STEP AP214 files with B-Rep geometry."""

    def __init__(self):
        self._id = 0
        self._entities = []
        self._point_cache = {}    # (x,y,z) → entity_id  (dedup shared vertices)
        self._dir_cache = {}      # (dx,dy,dz) → entity_id
        self._edge_cache = {}     # (v1_id, v2_id) → edge_curve_id (shared edges)

    # --- Geometric primitives ---
    def point(self, x, y, z) -> int
    def direction(self, dx, dy, dz) -> int
    def vector(self, dir_id, magnitude) -> int
    def axis2_placement(self, origin_id, z_dir_id, x_dir_id) -> int
    def line(self, point_id, vector_id) -> int
    def circle(self, placement_id, radius) -> int
    def plane(self, placement_id) -> int
    def cylindrical_surface(self, placement_id, radius) -> int

    # --- Topology ---
    def vertex(self, point_id) -> int
    def edge_curve(self, v1_id, v2_id, curve_id, same_sense=True) -> int
    def oriented_edge(self, edge_id, orientation=True) -> int
    def edge_loop(self, oriented_edge_ids) -> int
    def face_outer_bound(self, loop_id, orientation=True) -> int
    def face_bound(self, loop_id, orientation=True) -> int
    def advanced_face(self, bound_ids, surface_id, same_sense=True) -> int
    def closed_shell(self, face_ids) -> int
    def manifold_solid_brep(self, shell_id, name="") -> int

    # --- High-level helpers ---
    def planar_face(self, vertices_3d, normal) -> int
    def cylindrical_face(self, axis_origin, axis_dir, radius,
                          inner_arc_pts, outer_arc_pts) -> int

    # --- Output ---
    def write(self, filename, product_name="FLEX_PCB")
```

Key design decisions:
- **Point/direction deduplication**: Cache by rounded coordinates to reuse shared
  vertices and avoid floating-point mismatches at region boundaries
- **Edge sharing**: When two faces share an edge (same two vertex IDs), reuse the
  same `EDGE_CURVE` entity. The second face references it via `ORIENTED_EDGE` with
  opposite orientation (`.F.`).
- **Coordinate rounding**: Round to 6 decimal places for cache keys to handle
  floating-point noise from transform_point calculations.

---

## Solid Construction Per Region

### Flat Region → Extruded Prism

A flat region is a 2D polygon that has been transformed to 3D (possibly rotated
by previous folds). We extrude it by `thickness` along the negative normal.

```
        top face (PLANE)
       ┌──────────────┐
      ╱│             ╱│
     ╱ │            ╱ │  ← side faces (PLANE each)
    ┌──────────────┐  │
    │  └───────────│──┘
    │ ╱            │ ╱   thickness
    │╱             │╱
    └──────────────┘
       bottom face (PLANE)
```

Faces:
1. **Top face**: PLANE, edges from transformed outline polygon
2. **Bottom face**: PLANE, same polygon offset by -normal * thickness
3. **Side faces**: One PLANE per outline edge, connecting top to bottom

For a region with N outline vertices → N+2 faces total.

Construction:
```python
def build_flat_solid(region, recipe, thickness):
    # 1. Transform 2D outline → 3D top vertices
    top_verts = [transform_point(v, recipe) for v in region.outline]
    normal = compute_normal(region.outline[0], recipe)

    # 2. Compute bottom vertices
    bot_verts = [offset(v, -normal, thickness) for v in top_verts]

    # 3. Create faces
    top_face = planar_face(top_verts, normal)         # CCW from outside
    bot_face = planar_face(reversed(bot_verts), -normal)  # CW = CCW from outside
    side_faces = [planar_face(quad, side_normal)
                  for each edge pair (top[i]→top[i+1], bot[i+1]→bot[i])]

    # 4. Assemble into closed shell
    return closed_shell([top_face, bot_face] + side_faces)
```

Holes in the region outline create inner edge loops on top/bottom faces and
additional side faces (inner walls).

### Bend Region → Cylindrical Shell Section

A bend region wraps around a cylinder. The fold gives us:
- `axis`: direction along the fold line (cylinder axis)
- `center`: reference point on the fold line
- `radius`: R = zone_width / |angle|
- `angle`: rotation in radians

```
         outer cylinder (R + thickness)
        ╱‾‾‾‾‾‾‾‾‾‾‾‾╲
       ╱                ╲
      │    inner cyl (R)  │
      │   ╱‾‾‾‾‾‾‾‾‾╲   │  ← cross section
      │  │             │  │
      ╰──╯             ╰──╯
     start             end
     side              side
     (PLANE)           (PLANE)
```

Faces:
1. **Inner cylindrical face**: CYLINDRICAL_SURFACE(R), bounded by 2 arcs + 2 lines
2. **Outer cylindrical face**: CYLINDRICAL_SURFACE(R+t), bounded by 2 arcs + 2 lines
3. **Start side face**: PLANE, rectangular (connects inner→outer at angle=0)
4. **End side face**: PLANE, rectangular (connects inner→outer at angle=θ)

The two arc edges on each cylindrical face are `CIRCLE` entities (partial arcs
defined by start/end vertices on the circle).

Construction:
```python
def build_bend_solid(region, recipe, thickness):
    fold = get_in_zone_fold(recipe)
    R = fold.radius
    angle = fold.angle

    # Previous folds give us the 3D coordinate frame
    axis_3d, perp_3d, up_3d, origin_3d = compute_bend_frame(recipe)

    # Cylinder axis placement (AXIS2_PLACEMENT_3D)
    # The cylinder axis is at the BEFORE boundary of the bend zone
    cyl_origin = origin_3d + (-hw) * perp_3d   # fold center - half_width along perp
    cyl_axis = axis_3d                          # along the fold line
    cyl_ref_dir = perp_3d                       # reference direction on cylinder

    # Compute key 3D points (4 corners × 2 surfaces = 8 points)
    # Using the region outline's along-axis extent for width
    min_along, max_along = axis_extent(region.outline, fold)

    # Inner surface points (radius = R)
    inner_start_near = cyl_origin + min_along * axis_3d    # θ=0, near edge
    inner_start_far  = cyl_origin + max_along * axis_3d    # θ=0, far edge
    inner_end_near   = rotate(inner_start_near, angle)     # θ=angle, near edge
    inner_end_far    = rotate(inner_start_far, angle)      # θ=angle, far edge
    inner_mid_near   = rotate(inner_start_near, angle/2)   # midpoint for arc

    # Outer surface points (radius = R + thickness)
    # Same as inner but offset by thickness along cylinder radial direction
    ...

    # Create faces
    inner_face = cylindrical_face(cyl_placement, R, ...)
    outer_face = cylindrical_face(cyl_placement, R + thickness, ...)
    start_face = planar_face([inner_start_near, inner_start_far,
                              outer_start_far, outer_start_near], ...)
    end_face = planar_face([inner_end_near, outer_end_near,
                            outer_end_far, inner_end_far], ...)

    return closed_shell([inner_face, outer_face, start_face, end_face])
```

Note: "start" and "end" side faces connect the inner and outer cylinders.
The two long edges (along axis) of each cylindrical face are shared with
adjacent flat regions' side edges. This is how the solid is watertight.

---

## STEP File Structure

### Boilerplate (~25 entities, fixed)

```step
ISO-10303-21;
HEADER;
  FILE_DESCRIPTION(('Flex PCB STEP Export'), '2;1');
  FILE_NAME('{filename}', '{timestamp}', ('FlexViz'), (''),
    'FlexViz STEP Writer', 'KiCad Flex Viewer', '');
  FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
  /* Units: mm, radians */
  #1 = (LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.));
  #2 = (NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.));
  #3 = (NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT());
  #4 = UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-006),#1,...);
  #5 = (GEOMETRIC_REPRESENTATION_CONTEXT(3)
        GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#4))
        GLOBAL_UNIT_ASSIGNED_CONTEXT((#1,#2,#3))
        REPRESENTATION_CONTEXT('','3D'));
  /* Product definition chain */
  #6 = APPLICATION_CONTEXT('automotive design');
  #7 = APPLICATION_PROTOCOL_DEFINITION('international standard',
        'automotive_design',2000,#6);
  #8 = PRODUCT_CONTEXT('',#6,'mechanical');
  #9 = PRODUCT('{name}','{name}','',(#8));
  #10 = PRODUCT_DEFINITION_FORMATION('','',#9);
  #11 = PRODUCT_DEFINITION_CONTEXT('part definition',#6,'design');
  #12 = PRODUCT_DEFINITION('design','',#10,#11);
  #13 = PRODUCT_DEFINITION_SHAPE('','',#12);
  /* Then geometry entities follow... */
ENDSEC;
END-ISO-10303-21;
```

### Entity Count Estimate

For a board with N bends:
- Flat regions: N+1 regions × ~(6 faces × ~20 entities/face) ≈ 120(N+1)
- Bend regions: N × ~(4 faces × ~25 entities/face) ≈ 100N
- Shared edges reduce this by ~20%
- Boilerplate: ~25
- **Total: ~200N + 150 entities**

A 3-bend board: ~750 entities. A simple no-bend board: ~150 entities.

---

## Edge Sharing Strategy

Adjacent regions share edges at their boundary. This is critical for a valid
closed shell (watertight solid).

```
  Flat Region A          Bend Region          Flat Region B
 ┌──────────┐  shared  ╭──────────╮  shared  ┌──────────┐
 │           │  edge    │          │  edge    │          │
 │  PLANE    │◄────────►│ CYLINDER │◄────────►│  PLANE   │
 │           │          │          │          │          │
 └──────────┘          ╰──────────╯          └──────────┘
```

Implementation:
- Use a dict `{(v1_id, v2_id): edge_curve_id}` as edge cache
- When creating a face edge, check if the same vertex pair (in either order)
  already has an EDGE_CURVE
- If found, reuse it with `ORIENTED_EDGE(..., .F.)` (reversed orientation)
- Vertex IDs come from the point cache, so shared boundary points get the same ID

**However**: for the initial implementation, we can build each region as an
independent `CLOSED_SHELL` / `MANIFOLD_SOLID_BREP` and put them all in one
`ADVANCED_BREP_SHAPE_REPRESENTATION`. Each solid is valid on its own. Most CAD
tools handle this well (they show as one body with coincident faces at joints).
This avoids the complexity of cross-region edge sharing.

Later optimization: fuse into a single shell with shared edges.

---

## Implementation Plan

### Phase 1: StepWriter core (new file: `step_writer.py`)

Create the low-level STEP entity builder:
- Entity ID management with auto-increment
- Point/direction/vector primitives with dedup cache
- Topology builders (vertex, edge_curve, oriented_edge, edge_loop, face)
- PLANE and CYLINDRICAL_SURFACE surface types
- CIRCLE curve type (for arc edges on cylindrical faces)
- Closed shell and manifold solid brep assembly
- File output with header boilerplate and product definition chain

### Phase 2: Solid builders (in `step_writer.py` or separate)

High-level functions that use StepWriter to build solids:
- `build_flat_solid(writer, region, recipe, thickness)` → solid entity ID
- `build_bend_solid(writer, region, recipe, thickness)` → solid entity ID
- Handle holes in regions (inner edge loops + inner wall side faces)

### Phase 3: Integration (update `step_export.py`)

Replace `board_to_step_direct()` with new pipeline:
```python
def board_to_step_native(board_geometry, markers, filename, stiffeners=None):
    regions = split_board_into_regions(..., num_bend_subdivisions=1)
    writer = StepWriter()
    for region in regions:
        if is_bend_region(region):
            build_bend_solid(writer, region, recipe, thickness)
        else:
            build_flat_solid(writer, region, recipe, thickness)
    # Stiffeners as separate solids
    for stiffener in stiffeners:
        build_stiffener_solid(writer, stiffener, ...)
    writer.write(filename)
```

Keep `is_step_export_available()` → always True (no dependency).
Keep build123d path as optional fallback initially, remove later.

### Phase 4: Validation & Testing

- Write unit tests for StepWriter (entity generation, caching)
- Test with simple cases: flat board, one bend, multiple bends
- Validate output in FreeCAD / KiCad 3D viewer / online STEP viewer
- Compare against build123d output for correctness
- Test with boards that have holes/cutouts in bend zones

### Phase 5: Cleanup

- Remove build123d dependency from step_export.py
- Remove step_export_cli.py (no longer needed)
- Remove install_step_export.sh
- Update viewer.py STEP export button to call new writer directly
- Update documentation

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Topological errors (non-watertight) | Start with independent solids per region (always valid). Add shared edges later. |
| Arc edge precision | Use exact analytical formulas from bend_transform.py. Round to 6dp for cache. |
| Complex outlines with many vertices | planar_subdivision already simplifies regions. Each region is a simple polygon. |
| STEP import failures in CAD tools | Test with FreeCAD (open source, strict parser). Fix issues iteratively. |
| Stiffeners on bend zones | Initially flat-only stiffeners (current behavior). Bent stiffeners as future work. |

---

## File Changes Summary

| File | Action |
|------|--------|
| `step_writer.py` | **NEW** — Pure Python STEP B-Rep writer |
| `step_export.py` | **REWRITE** — Use StepWriter instead of build123d |
| `step_export_cli.py` | **DELETE** — No longer needed |
| `install_step_export.sh` | **DELETE** — No longer needed |
| `viewer.py` | **UPDATE** — STEP export calls new writer directly |
| `tests/test_step_writer.py` | **NEW** — Unit tests for STEP writer |
