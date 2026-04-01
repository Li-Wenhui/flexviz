# STEP Export Design Guide

## Goal

Replace the tessellated (triangle mesh) STEP export with one that uses **exact KiCad primitives** (lines, arcs, circles) to produce clean B-Rep geometry. The current mesh.py pipeline tessellates everything into triangles — good for STL/OpenGL, but STEP files should use native PLANE and CYLINDRICAL_SURFACE entities with exact edge curves.

## Architecture Overview

```
KiCadPCB
  ├── get_board_outline_with_arcs() → (vertices, segments[line|arc])
  ├── get_polygon_cutouts()         → linearized vertex lists
  ├── get_circle_cutouts()          → CircleCutout(cx, cy, r)
  └── get_drill_holes()             → DrillHole(x, y, diameter)

extract_geometry(pcb) → BoardGeometry
  ├── outline: Polygon (vertices + OutlineSegment[line|arc])
  ├── cutouts: list[Polygon] (linearized — circles are 24-gon, arcs lost)
  ├── thickness: float (mm)
  └── traces, pads, components (not needed for STEP board shape)

detect_fold_markers(pcb) → FoldMarker[]
  └── Each: center, axis, perp, zone_width, angle_degrees, radius

split_board_into_regions(outline, holes, markers, num_bend_subdivisions=1)
  → Region[] each with: outline, holes, fold_recipe, index

For each Region:
  fold_recipe = [(FoldMarker, "IN_ZONE"|"AFTER", entered_from_back), ...]
  ├── Flat region: last entry is "AFTER" or recipe is empty
  └── Bend region: last entry is "IN_ZONE"
```

## Key Data Structures

### OutlineSegment (geometry.py)
```python
@dataclass
class OutlineSegment:
    type: str                        # "line" or "arc"
    start: (float, float)
    end: (float, float)
    center: (float, float) = None    # arcs only
    radius: float = 0.0             # arcs only
    mid: (float, float) = None      # 3-point arc definition
```

### FoldMarker (markers.py)
```python
@dataclass
class FoldMarker:
    line_a_start/end, line_b_start/end  # The two parallel marker lines
    angle_degrees: float                 # +ve = toward viewer
    zone_width: float                    # distance between marker lines
    radius: float                        # = zone_width / |angle_radians|
    axis: (float, float)                 # unit vector along fold lines
    center: (float, float)               # midpoint between line midpoints
    # Derived:
    perp = (-axis[1], axis[0])           # perpendicular direction (BEFORE→AFTER)
```

### Region (planar_subdivision.py)
```python
@dataclass
class Region:
    outline: list[(float, float)]        # CCW boundary vertices (linearized)
    holes: list[list[(float, float)]]    # CW holes
    index: int
    fold_before: Optional[FoldMarker]
    fold_after: Optional[FoldMarker]
    fold_recipe: list[(FoldMarker, classification, entered_from_back)]
    representative_point: (float, float)
```

### FoldDefinition (bend_transform.py)
```python
@dataclass
class FoldDefinition:
    center, axis, zone_width, angle (radians)
    # Derived:
    radius = zone_width / |angle|
    perp = (-axis[1], axis[0])
```

## Existing Pipeline: What It Does

### Planar Subdivision
1. Cutting lines are created at fold zone boundaries (perpendicular to fold axis)
2. With `num_bend_subdivisions=1` for STEP, each fold zone creates 2 cutting lines (at ±hw from center)
3. The board outline + holes + cutting lines form a planar graph
4. Regions are traced using rightmost-turn algorithm
5. Each region gets a `fold_recipe` via BFS from an anchor region (one BEFORE all folds)

### Region Classification
- **Flat region**: `fold_recipe` is empty OR last entry has classification `"AFTER"`
- **Bend region**: last entry has classification `"IN_ZONE"`

### 3D Transformation (bend_transform.py)
For a 2D point with a fold recipe, iterate through each entry:

**IN_ZONE** (cylindrical mapping):
```
dist_into_zone = clamp(perp_dist + hw, 0, zone_width)
theta = (dist_into_zone / zone_width) * angle
R = zone_width / |angle|
local_perp = R * sin(|theta|) - hw
local_up = R * (1 - cos(|theta|)) * sign(angle)
```

**AFTER** (rigid rotation):
```
zone_end computed from full IN_ZONE at perp_dist=hw
excess = perp_dist - hw
rotation_angle = angle (or -angle+π for back entry)
local_perp = zone_end_perp + excess * cos(rotation_angle)
local_up = zone_end_up + excess * sin(rotation_angle)
```

Cumulative: maintains 3×3 rotation matrix + 3D origin for multi-fold chains.

### Back Entry
When a region enters a fold from the AFTER side (traveled backward through the fold column), `entered_from_back=True`. This mirrors the perpendicular coordinate and adjusts the rotation angle to `(-angle + π)`.

## Design: STEP Exporter

### Input
Same as current: `board_to_step(board_geometry, markers, filename, config, pcb, stiffeners)`

Available from inputs:
- `board_geometry.outline.segments` — **exact** line/arc segments (NOT tessellated)
- `board_geometry.outline.vertices` — linearized vertices (arcs are densified)
- `board_geometry.cutouts` — **linearized only** (circles→24-gon, arcs→polyline)
- `pcb.get_circle_cutouts()` — **exact** circles with center+radius
- `pcb.get_drill_holes()` — **exact** circles with center+diameter

### Strategy

#### Phase 1: Region Computation (reuse existing)
```python
regions = split_board_into_regions(
    outline_2d, holes_2d, markers, num_bend_subdivisions=1
)
```
This gives us regions with linearized polygon outlines. For STEP, we need to:
1. Keep the region partition (it determines which fold recipe applies)
2. But also preserve arc info for edges that come from the original outline

#### Phase 2: Edge Primitive Recovery

**Problem**: `split_board_into_regions` only outputs linearized vertex lists — arc info is lost at the cutting stage.

**Solution**: After subdivision, match region edges back to original outline segments:
- For each consecutive edge (v[i], v[i+1]) in a region outline
- Check if both endpoints lie on an original arc segment (within tolerance)
- If yes, tag that edge as an arc with the original center/radius
- Otherwise it's a line (either from the original outline or from a cutting line)

This is the **critical bridge** between the region polygons and exact STEP geometry.

Algorithm for arc recovery:
```python
def recover_arcs(region_outline, original_segments, tolerance=0.01):
    """Tag region edges that correspond to original arc segments."""
    tagged_edges = []
    for i in range(len(region_outline)):
        p1 = region_outline[i]
        p2 = region_outline[(i+1) % len(region_outline)]

        arc_seg = find_matching_arc(p1, p2, original_segments, tolerance)
        if arc_seg:
            tagged_edges.append({
                'type': 'arc',
                'start': p1, 'end': p2,
                'center': arc_seg.center,
                'radius': arc_seg.radius
            })
        else:
            tagged_edges.append({
                'type': 'line',
                'start': p1, 'end': p2
            })
    return tagged_edges
```

**For cutouts**: recover circles from `pcb.get_circle_cutouts()` and `pcb.get_drill_holes()`.

#### Phase 3: Build STEP Solids per Region

For each region, based on its fold_recipe:

##### Flat Region Solid
A flat region is an extruded slab:
- **Top face**: PLANE with edges from the region outline (lines + arcs from Phase 2)
- **Bottom face**: Same outline offset by `-normal * thickness`
- **Side faces**: One per outline edge
  - Line edge → planar quad side face
  - Arc edge → cylindrical surface side face
- **Hole faces** (if cutouts): Same as above but inward-facing

Transform all 2D points to 3D using `transform_point(pt, fold_recipe)`.
Compute face normal using `compute_normal(representative_pt, fold_recipe)`.

##### Bend Region Solid
A bend region is a cylindrical slab between inner radius R and outer radius R+thickness:
- **Inner cylindrical face**: CYLINDRICAL_SURFACE at radius R
- **Outer cylindrical face**: CYLINDRICAL_SURFACE at radius R+thickness
- **2 end cap faces**: Planar radial cross-sections (line edges only)
- **2 side faces**: Planar annular sectors with arc edges (shared with cylindrical faces)

The bend geometry is computed from the fold definition:
```
R = fold_def.radius
theta_min, theta_max from region extent in fold zone
cyl_axis, cyl_ref, cyl_origin from prior recipe transformations
```

### Phase 4: Stiffeners
For each stiffener:
1. Find containing region via `find_containing_region(centroid, regions)`
2. Use region's fold_recipe to transform stiffener outline to 3D
3. Offset by PCB thickness for bottom-side stiffeners
4. Build flat extruded solid (stiffeners are always flat)

### STEP File Structure

Each solid is a `MANIFOLD_SOLID_BREP` → `CLOSED_SHELL` → `ADVANCED_FACE[]`.

All solids go into one `ADVANCED_BREP_SHAPE_REPRESENTATION` with:
- Units: mm (SI_UNIT(.MILLI.,.METRE.)) and radians
- One PRODUCT per named body (FLEX_PCB_0, FLEX_PCB_1, ..., STIFFENER_0, ...)

### Edge Sharing for Valid Topology

Adjacent faces within a solid MUST share EDGE_CURVE entities:
- Use a vertex cache (dedup by rounded coordinates)
- Use an edge cache (dedup by vertex pair)
- Line edges shared between top/bottom/side faces
- Arc/circle edges shared between cylindrical faces and annular side faces

## Edge Cases to Handle

### 1. Negative Fold Angles
`fold_def.angle < 0` means folding away from viewer. Theta values will be negative. Ensure arc sweep direction is always short (CCW on STEP circle). Fix: swap `theta_min/theta_max` if `theta_min > theta_max`.

### 2. Back Entry
When `entered_from_back=True`, the perpendicular coordinate is mirrored and rotation angle becomes `(-angle + π)`. The cylinder axis position changes from `-hw` to `+hw` side.

### 3. Multi-Fold Chains
Regions after multiple folds have compound recipes. The 3D transformation is cumulative: each AFTER entry rotates the coordinate frame. Cylinder geometry for a bend-after-bend must account for the rotated frame.

### 4. Board Outline Arcs in Bend Zones
If the original board outline has arcs that cross a fold zone boundary, the subdivision splits them. The arc recovery step must handle partial arcs (sub-arcs of the original).

### 5. Cutout Circles vs Polygons
Circle cutouts should be CIRCLE edges in STEP (exact), not 24-gon approximations. Polygon cutouts from `get_polygon_cutouts()` are linearized — can recover arcs if they came from `gr_arc` elements.

### 6. Zero-Length Edges
Degenerate edges (length < 1e-12) from numerical coincidence at cutting intersections. Skip these in edge loop construction.

### 7. Coplanar Face Snapping
For flat regions, snap all transformed vertices to a best-fit plane to eliminate floating-point drift. Critical for STEP tolerance.

## Implementation Modules

### `step_writer.py` — Low-Level STEP Entities
Already exists. Provides:
- Entity dedup (points, directions, vertices, edges)
- `add_planar_face()`, `add_cylindrical_face()`
- `build_flat_solid()`, `build_bend_solid()`
- `write()` — complete STEP AP214 output

Needs extension for:
- Faces with mixed line+arc edge loops
- Arc edge creation (CIRCLE EDGE_CURVE)

### `step_export.py` — Orchestrator
Ties together:
1. Region computation from planar_subdivision
2. Arc recovery from original outline segments
3. 3D transformation from bend_transform
4. Solid construction via step_writer
5. File output

### No Changes Needed To
- `kicad_parser.py` — already provides arc-aware outline
- `geometry.py` — already stores OutlineSegments
- `markers.py` — fold detection is independent
- `bend_transform.py` — transformation math is reusable
- `planar_subdivision.py` — region computation is reusable
- `viewer.py` — just calls `board_to_step_native()`

## Verification Checklist

1. Flat board (no folds): single extruded solid, exact outline edges
2. Board with arcs in outline: CIRCLE edges on side faces
3. Single fold (positive angle): 3 solids, correct cylinder geometry
4. Single fold (negative angle): same, arcs go correct direction
5. Multi-fold: compound transformations, all regions connected
6. Back-entry fold: mirrored geometry correct
7. Cutout circles: exact CIRCLE edges, not polygons
8. Stiffeners: positioned on correct side with correct offset
9. All entity references resolve (no dangling #N)
10. FreeCAD imports with Solids > 0, all shells closed
