# Stage 4: 3D Transformation

> The mathematics of transforming 2D flat PCB geometry into 3D bent geometry using cylindrical mapping and rotated plane continuation.

Each region's fold recipe determines exactly how its 2D vertices are mapped to 3D. The transformation has two cases: **IN_ZONE** (cylindrical bending) and **AFTER** (rotated flat plane).

## IN_ZONE: Cylindrical Mapping

Points inside a bend zone are mapped onto a **cylindrical arc**. The fold axis (at the BEFORE boundary) serves as the cylinder's rotation axis:

![Cylindrical Mapping](figures/04_cylindrical_mapping.png)

### Formulas

Given a point with perpendicular distance `perp_dist` from the fold center:

```
R     = zone_width / |angle|          (bend radius)
hw    = zone_width / 2                (half-width)
dist  = perp_dist + hw               (distance from BEFORE boundary, in [0, zone_width])
f     = dist / zone_width            (arc fraction, in [0, 1])
theta = f * angle                    (rotation angle at this point)
```

The transformed local coordinates are:

```
perp' = R * sin(|theta|) - hw
up'   = R * (1 - cos(|theta|))       (negated if angle < 0)
```

**Boundary conditions:**
- At BEFORE boundary (theta = 0): `perp' = -hw`, `up' = 0` (original position, no rotation)
- At AFTER boundary (theta = angle): `perp' = R*sin(|angle|) - hw`, `up' = R*(1-cos(|angle|))`

## AFTER: Rotated Plane Continuation

Points past the bend zone continue on a **flat plane tangent to the cylinder** at its end:

![AFTER Plane](figures/04_after_plane.png)

```
zone_end_perp = R * sin(|angle|) - hw
zone_end_up   = R * (1 - cos(|angle|))
excess        = perp_dist - hw               (distance past the zone)

perp' = zone_end_perp + excess * cos(angle)
up'   = zone_end_up   + excess * sin(angle)
```

The excess distance is projected along the **tangent direction** at the arc endpoint. This ensures the flat plane is tangent to the cylinder -- no discontinuity at the boundary.

## Continuity Proof

The transformation is **C0-continuous** (position-continuous) across zone boundaries. Both `perp'` and `up'` match exactly at the BEFORE and AFTER boundaries:

![Continuity](figures/04_continuity.png)

- **Blue curve**: `perp'` as a function of `perp_dist`
- **Red curve**: `up'` as a function of `perp_dist`
- **Green zone**: BEFORE (flat, no transform)
- **Orange zone**: IN_ZONE (cylindrical arc)
- **Blue zone**: AFTER (rotated plane)
- **Black dots**: boundary points showing exact matching

## Back-Entry Transformation

When a region is entered from the AFTER side (`entered_from_back = True`), the transformation is **mirrored**:

1. `perp_dist` is negated (measure from opposite boundary)
2. `local_perp` is negated for IN_ZONE (cylinder axis at opposite side)
3. For AFTER: `tangent_angle = -angle + pi` (tangent flips), but `cumulative_angle = -angle` (rotation matrix uses the physical angle)

The key insight: **tangent angle** (for position) and **cumulative angle** (for rotation of subsequent folds) are different for back-entry. This separation was critical for fixing -90 degree fold discontinuities.

## Multi-Fold Cumulative Transform

For regions affected by multiple folds, transformations are composed via an **affine transformation** tracked through each fold:

```
pos_3d = rot @ point_2d + origin
```

Each AFTER fold updates the cumulative rotation (`rot`) and translation (`origin`). The fold's local coordinate frame is transformed through the cumulative rotation using **Rodrigues' rotation formula**:

```
R(v) = v*cos(theta) + (k x v)*sin(theta) + k*(k.v)*(1 - cos(theta))
```

where `k` is the fold axis (unit vector) and `theta` is the fold angle.

### H-Shape Board: Full 3D Folding

All 73 regions transformed to 3D using their fold recipes:

![Multi-Fold 3D](figures/04_multi_fold_3d.png)

Each colored patch is one region. The H-shape folds into a complex 3D structure with up to 5 nested folds per region.

## Grid Deformation

Visualizing how a regular 2D grid deforms through the folding pipeline:

![Grid Deformation](figures/04_grid_deformation.png)

- **Left**: flat 2D grid points on the left arm of the H
- **Right**: same points after 3D transformation
- **Color**: original Y coordinate (shows correspondence between flat and bent)

Points that were uniformly spaced on the flat board follow smooth curves through the bend zones and maintain their relative positions on the rotated planes.

## Surface Normals

The surface normal at each point is computed alongside the position. Normals rotate through the same fold recipe:

![Surface Normals](figures/04_normals.png)

- **Green arrows**: flat regions (normal = (0, 0, 1), pointing straight up)
- **Orange arrows**: bend zones (normal rotates smoothly through the arc)
- **Blue arrows**: folded regions (normal fully rotated by the fold angle)

Normals are essential for correct lighting in the OpenGL viewer and for face orientation in STEP export.

---

## Summary: Complete Transform Pipeline

```
For each vertex (x, y) with fold recipe [(F1, AFTER), (F2, IN_ZONE)]:

1. Initialize: rot = Identity, origin = (0,0,0)

2. Process F1 (AFTER):
   - Compute perp_dist, along from fold center
   - Map to 3D: zone_end + excess * tangent
   - Update rot via Rodrigues around fold axis
   - Update origin to maintain consistency

3. Process F2 (IN_ZONE):
   - Compute perp_dist, along from fold center
   - Map to cylinder: R*sin(theta) - hw, R*(1-cos(theta))
   - Return 3D position (IN_ZONE is terminal)
```

**Previous**: [Stage 3: Fold Recipes](03_fold_recipes.md)

*Generated by `04_3d_transformation.py`*
