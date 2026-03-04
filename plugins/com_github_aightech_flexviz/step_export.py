"""
STEP file export for flex PCB visualization.

Pure Python — no external dependencies. Uses StepWriter to emit B-Rep geometry
(PLANE + CYLINDRICAL_SURFACE) directly to STEP AP214 format.
"""

import math
from dataclasses import dataclass

try:
    from .step_writer import StepWriter, _normalize, _cross, _sub, _add, _scale, _dot
    from .bend_transform import (
        FoldDefinition, transform_point, compute_normal,
        _rotation_matrix_around_axis, _apply_rotation, _multiply_matrices,
    )
    from .planar_subdivision import split_board_into_regions, find_containing_region
    from .stiffener import extract_stiffeners
except ImportError:
    from step_writer import StepWriter, _normalize, _cross, _sub, _add, _scale, _dot
    from bend_transform import (
        FoldDefinition, transform_point, compute_normal,
        _rotation_matrix_around_axis, _apply_rotation, _multiply_matrices,
    )
    from planar_subdivision import split_board_into_regions, find_containing_region
    from stiffener import extract_stiffeners


# =====================================================================
# TaggedEdge: edge with arc/line type info preserved after subdivision
# =====================================================================

@dataclass
class TaggedEdge:
    """An edge that knows whether it's a line or arc segment."""
    type: str          # "line" or "arc"
    start: tuple       # (x, y) 2D
    end: tuple         # (x, y) 2D
    center: tuple = None  # (x, y) for arcs
    radius: float = 0.0
    ccw: bool = True       # True = CCW arc from start to end


def _point_on_arc(pt, arc_segment, tolerance=0.01):
    """
    Check if a 2D point lies on an OutlineSegment arc.

    Verifies:
    1. Distance from point to arc center matches radius (within tolerance)
    2. Point's angle is within the arc's angular span (using mid point for sweep direction)
    """
    if arc_segment.type != "arc" or arc_segment.center is None:
        return False

    cx, cy = arc_segment.center
    dx = pt[0] - cx
    dy = pt[1] - cy
    dist = math.sqrt(dx * dx + dy * dy)

    if abs(dist - arc_segment.radius) > tolerance:
        return False

    # Check angular span using start, mid, end points
    angle_pt = math.atan2(dy, dx)
    angle_start = math.atan2(arc_segment.start[1] - cy, arc_segment.start[0] - cx)
    angle_end = math.atan2(arc_segment.end[1] - cy, arc_segment.end[0] - cx)

    if arc_segment.mid is not None:
        angle_mid = math.atan2(arc_segment.mid[1] - cy, arc_segment.mid[0] - cx)
    else:
        # Without mid, assume short arc from start to end CCW
        angle_mid = (angle_start + angle_end) / 2

    # Determine sweep direction from start -> mid -> end
    # Normalize all angles to [0, 2*pi)
    def norm_angle(a):
        return a % (2 * math.pi)

    a_s = norm_angle(angle_start)
    a_m = norm_angle(angle_mid)
    a_e = norm_angle(angle_end)
    a_p = norm_angle(angle_pt)

    # Check if mid is in the CCW sweep from start to end
    def in_ccw_sweep(start, end, test):
        """Check if test angle is in CCW sweep from start to end."""
        if start <= end:
            return start <= test <= end
        else:  # wraps around 2*pi
            return test >= start or test <= end

    # Determine if the arc goes CCW or CW by checking mid
    if in_ccw_sweep(a_s, a_e, a_m):
        # CCW from start to end
        return in_ccw_sweep(a_s, a_e, a_p)
    else:
        # CW from start to end = CCW from end to start
        return in_ccw_sweep(a_e, a_s, a_p)


def _recover_arcs(region_outline, original_segments, tolerance=0.01):
    """
    Recover arc info for edges in a region outline after planar subdivision.

    For each consecutive edge (v[i] -> v[i+1]), check if both endpoints lie on
    the same original arc segment. If yes, tag as arc with parent's center/radius.

    Args:
        region_outline: list of (x, y) vertices from subdivision
        original_segments: list of OutlineSegment from the original board outline

    Returns:
        list of TaggedEdge
    """
    n = len(region_outline)
    tagged = []

    # Filter to just arc segments for efficiency
    arc_segments = [s for s in original_segments if s.type == "arc" and s.center is not None]

    for i in range(n):
        p1 = region_outline[i]
        p2 = region_outline[(i + 1) % n]

        # Check if this edge lies on any original arc
        matched_arc = None
        for arc in arc_segments:
            if _point_on_arc(p1, arc, tolerance) and _point_on_arc(p2, arc, tolerance):
                matched_arc = arc
                break

        if matched_arc is not None:
            # Determine sweep direction for THIS edge (p1→p2), not the original arc.
            # The recovered edge may traverse the arc in the opposite direction from
            # the original, so we use cross(p1-C, p2-C) to find the short-arc direction.
            # Positive cross → end is CCW from start → short arc is CCW → same_sense=True.
            cx, cy = matched_arc.center
            sx = p1[0] - cx
            sy = p1[1] - cy
            ex = p2[0] - cx
            ey = p2[1] - cy
            ccw = (sx * ey - sy * ex) > 0
            tagged.append(TaggedEdge(
                type="arc",
                start=p1,
                end=p2,
                center=matched_arc.center,
                radius=matched_arc.radius,
                ccw=ccw,
            ))
        else:
            tagged.append(TaggedEdge(
                type="line",
                start=p1,
                end=p2,
            ))

    # Merge consecutive arcs on the same circle into single arcs
    merged = _merge_consecutive_arcs(tagged, tolerance)
    return merged


def _merge_consecutive_arcs(tagged, tolerance=0.01):
    """Merge consecutive TaggedEdges that are arcs on the same circle.

    Adjacent arcs with matching center and radius are combined into a single
    arc edge spanning from the first arc's start to the last arc's end.
    """
    if not tagged:
        return tagged

    merged = []
    i = 0
    while i < len(tagged):
        e = tagged[i]
        if e.type != 'arc':
            merged.append(e)
            i += 1
            continue

        # Start a run of arcs on the same circle
        run_start = e.start
        run_end = e.end
        cx, cy, r = e.center[0], e.center[1], e.radius
        ccw = e.ccw
        j = i + 1
        while j < len(tagged):
            nxt = tagged[j]
            if (nxt.type == 'arc' and
                    abs(nxt.center[0] - cx) < tolerance and
                    abs(nxt.center[1] - cy) < tolerance and
                    abs(nxt.radius - r) < tolerance and
                    nxt.ccw == ccw):
                run_end = nxt.end
                j += 1
            else:
                break

        merged.append(TaggedEdge(
            type='arc',
            start=run_start,
            end=run_end,
            center=e.center,
            radius=e.radius,
            ccw=ccw,
        ))
        i = j

    return merged


def _recover_circle_holes(region_holes, circle_cutouts, drill_holes, tolerance=0.01):
    """
    Identify which region holes are actually circles (from cutouts or drill holes).

    For each hole polygon, check if all vertices are equidistant from a known circle
    center. If so, return it as a circle descriptor instead of a polygon.

    Args:
        region_holes: list of hole vertex lists from subdivision
        circle_cutouts: list of CircleCutout objects (center_x, center_y, radius)
        drill_holes: list of DrillHole objects (center_x, center_y, diameter)

    Returns:
        list of ('circle', (cx, cy), radius) or ('polygon', [TaggedEdge...])
    """
    # Build list of known circles: (cx, cy, radius)
    known_circles = []
    if circle_cutouts:
        for c in circle_cutouts:
            known_circles.append((c.center_x, c.center_y, c.radius))
    if drill_holes:
        for h in drill_holes:
            known_circles.append((h.center_x, h.center_y, h.diameter / 2))

    result = []
    for hole in region_holes:
        if len(hole) < 3:
            result.append(('polygon', [TaggedEdge(type="line", start=hole[i], end=hole[(i+1) % len(hole)]) for i in range(len(hole))]))
            continue

        matched = False
        for (cx, cy, r) in known_circles:
            all_match = True
            for v in hole:
                dx = v[0] - cx
                dy = v[1] - cy
                dist = math.sqrt(dx * dx + dy * dy)
                if abs(dist - r) > tolerance:
                    all_match = False
                    break
            if all_match:
                result.append(('circle', (cx, cy), r))
                matched = True
                break

        if not matched:
            # Return as polygon with line edges
            n = len(hole)
            edges = [TaggedEdge(type="line", start=hole[i], end=hole[(i+1) % n]) for i in range(n)]
            result.append(('polygon', edges))

    return result


def is_step_export_available():
    """STEP export is always available (pure Python)."""
    return True


def _marker_to_fold_def(marker):
    """Convert a FoldMarker to a FoldDefinition."""
    return FoldDefinition.from_marker(marker)


def _recipe_with_fold_defs(recipe):
    """Convert a region fold_recipe (with FoldMarker objects) to use FoldDefinitions."""
    result = []
    for entry in recipe:
        marker = entry[0]
        classification = entry[1]
        entered_from_back = entry[2] if len(entry) > 2 else False
        fold_def = _marker_to_fold_def(marker)
        result.append((fold_def, classification, entered_from_back))
    return result


def _transform_polygon_3d(polygon_2d, recipe):
    """Transform a 2D polygon to 3D using a fold recipe."""
    return [transform_point(p, recipe) for p in polygon_2d]


def _transform_tagged_edges_3d(tagged_edges_2d, recipe, face_normal):
    """Transform 2D tagged edges to 3D edge dicts for StepWriter.

    For arcs, also transforms center and computes axis/ref_dir.
    Applies coplanar snapping to eliminate floating-point drift.

    Args:
        tagged_edges_2d: list of TaggedEdge (2D)
        recipe: fold recipe for 3D transformation
        face_normal: 3D face normal (used as cylinder axis for arcs)

    Returns:
        list of edge dicts suitable for StepWriter._make_mixed_loop
    """
    n = _normalize(face_normal)

    # First pass: transform all points to 3D
    raw_points = []
    for edge in tagged_edges_2d:
        raw_points.append(transform_point(edge.start, recipe))
    # Snap line endpoints to plane (arc endpoints stay on the arc)
    snapped = _snap_to_plane(raw_points, n)

    result = []
    for i, edge in enumerate(tagged_edges_2d):
        start_3d = snapped[i]
        end_3d = snapped[(i + 1) % len(snapped)]

        if edge.type == 'arc' and edge.center is not None:
            center_3d = transform_point(edge.center, recipe)
            # Snap center to same plane
            count = len(snapped)
            cx = sum(v[0] for v in snapped) / count
            cy = sum(v[1] for v in snapped) / count
            cz = sum(v[2] for v in snapped) / count
            plane_center = (cx, cy, cz)
            diff = _sub(center_3d, plane_center)
            dist = _dot(diff, n)
            center_3d = _sub(center_3d, _scale(n, dist))

            ref_dir = _normalize(_sub(start_3d, center_3d))
            result.append({
                'type': 'arc',
                'start': start_3d,
                'end': end_3d,
                'center': center_3d,
                'axis': n,
                'ref_dir': ref_dir,
                'radius': edge.radius,
                'ccw': edge.ccw,
            })
        else:
            result.append({
                'type': 'line',
                'start': start_3d,
                'end': end_3d,
            })
    return result


def _compute_region_normal(region, recipe):
    """Compute the surface normal for a region."""
    if region.representative_point:
        return compute_normal(region.representative_point, recipe)
    # Fallback: use centroid
    if region.outline:
        n = len(region.outline)
        cx = sum(p[0] for p in region.outline) / n
        cy = sum(p[1] for p in region.outline) / n
        return compute_normal((cx, cy), recipe)
    return (0.0, 0.0, 1.0)


def _is_bend_region(recipe):
    """Check if the last entry in the recipe is IN_ZONE (bend region)."""
    if not recipe:
        return False
    last_entry = recipe[-1]
    return last_entry[1] == "IN_ZONE"


def _get_bend_info(recipe):
    """Get the fold definition and entry info for a bend region's active fold."""
    if not recipe:
        return None, None, False
    last_entry = recipe[-1]
    if last_entry[1] != "IN_ZONE":
        return None, None, False
    marker = last_entry[0]
    entered_from_back = last_entry[2] if len(last_entry) > 2 else False
    return _marker_to_fold_def(marker), last_entry[1], entered_from_back


def _build_bend_region_solid(writer, region, recipe_with_defs, thickness):
    """
    Build a cylindrical solid for a bend zone region.

    For a bend region, the last recipe entry is IN_ZONE. We compute the
    cylinder geometry from the fold definition and region outline.
    """
    fold_def = None
    entered_from_back = False
    for entry in reversed(recipe_with_defs):
        if entry[1] == "IN_ZONE":
            fold_def = entry[0]
            entered_from_back = entry[2] if len(entry) > 2 else False
            break

    if fold_def is None:
        # Fallback to flat
        return _build_flat_region_solid(writer, region, recipe_with_defs, thickness)

    # The recipe up to (but not including) the last IN_ZONE entry gives us
    # the cumulative transformation before entering the bend zone.
    prior_recipe = recipe_with_defs[:-1]

    hw = fold_def.zone_width / 2
    R = fold_def.radius

    # Find the min/max perpendicular distances of region vertices from fold center
    perp_dists = []
    along_dists = []
    for p in region.outline:
        dx = p[0] - fold_def.center[0]
        dy = p[1] - fold_def.center[1]
        perp = dx * fold_def.perp[0] + dy * fold_def.perp[1]
        along = dx * fold_def.axis[0] + dy * fold_def.axis[1]
        perp_dists.append(perp)
        along_dists.append(along)

    perp_min = min(perp_dists)
    perp_max = max(perp_dists)
    along_min = min(along_dists)
    along_max = max(along_dists)

    # For back entry, mirror perpendicular distances
    if entered_from_back:
        perp_min_eff = -perp_max
        perp_max_eff = -perp_min
    else:
        perp_min_eff = perp_min
        perp_max_eff = perp_max

    # Arc fractions
    dist_into_zone_min = max(0, perp_min_eff + hw)
    dist_into_zone_max = min(fold_def.zone_width, perp_max_eff + hw)
    frac_min = dist_into_zone_min / fold_def.zone_width if fold_def.zone_width > 0 else 0
    frac_max = dist_into_zone_max / fold_def.zone_width if fold_def.zone_width > 0 else 0
    # Always use positive theta: the cyl_axis negation already handles
    # the sweep direction for positive angles. For negative angles the
    # physical sweep is opposite, but abs(angle) with the same negated axis
    # produces the correct geometry (tangent goes in the right direction).
    theta_min = frac_min * abs(fold_def.angle)
    theta_max = frac_max * abs(fold_def.angle)

    # Compute cylinder axis in 3D using the prior recipe
    rot = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    origin = (0.0, 0.0, 0.0)
    for entry in prior_recipe:
        f = entry[0]
        c = entry[1]
        back = entry[2] if len(entry) > 2 else False
        if c == "AFTER":
            fold_axis_3d = _apply_rotation(rot, (f.axis[0], f.axis[1], 0.0))
            rotation_angle = (-f.angle + math.pi) if back else f.angle
            fold_rot = _rotation_matrix_around_axis(fold_axis_3d, rotation_angle)
            rot = _multiply_matrices(fold_rot, rot)
            # We don't need to track origin precisely here - we'll use transform_point

    fold_axis_3d = _apply_rotation(rot, (fold_def.axis[0], fold_def.axis[1], 0.0))
    fold_perp_3d = _apply_rotation(rot, (fold_def.perp[0], fold_def.perp[1], 0.0))
    up_3d = _apply_rotation(rot, (0.0, 0.0, 1.0))

    # Cylinder axis position: at the BEFORE boundary of the fold
    # The fold axis (rotation center) is at perp_dist = -hw from fold center
    # For back entry, the axis is at the opposite side (+hw)
    if entered_from_back:
        axis_perp_offset = hw
    else:
        axis_perp_offset = -hw

    # Get the fold center in 3D
    # Use a reference point at fold center with prior recipe to get 3D position
    fold_center_2d = fold_def.center
    center_ref_2d = (
        fold_center_2d[0] + axis_perp_offset * fold_def.perp[0],
        fold_center_2d[1] + axis_perp_offset * fold_def.perp[1],
    )
    # surface_before_3d: the PCB surface point at the BEFORE boundary of the fold zone
    surface_before_3d = transform_point(center_ref_2d, prior_recipe)

    # The rotation center is displaced from the surface point by R * up_3d
    # (sign of fold angle determines which side: positive angle folds upward)
    sign_angle = 1.0 if fold_def.angle >= 0 else -1.0
    cyl_origin_3d = _add(surface_before_3d, _scale(up_3d, sign_angle * R))

    # cyl_ref points FROM rotation center TO the initial surface point (theta=0)
    cyl_ref = _normalize(_scale(up_3d, -sign_angle))

    # Cylinder axis direction depends on angle sign.
    # We need cross(cyl_axis, cyl_ref) = fold_perp_3d for correct sweep direction.
    # With cyl_ref = -sign*up, this requires cyl_axis = sign*fold_axis.
    cyl_axis = _normalize(_scale(fold_axis_3d, sign_angle))

    # For positive angle (center above surface): bottom is further from center
    # For negative angle (center below surface): bottom is closer to center
    if sign_angle >= 0:
        inner_radius = R
        outer_radius = R + thickness
    else:
        inner_radius = R - thickness
        outer_radius = R

    # Along values are in fold_axis direction; project onto cyl_axis = sign*fold_axis
    if sign_angle >= 0:
        along_min_cyl = along_min
        along_max_cyl = along_max
    else:
        along_min_cyl = -along_max
        along_max_cyl = -along_min

    # _cyl_point: center + along*cyl_axis + r*(cos(theta)*cyl_ref + sin(theta)*tangent)
    # where tangent = cross(cyl_axis, cyl_ref) now points in the fold_perp direction
    def _cyl_point(along, theta, radius):
        tangent = _normalize(_cross(cyl_axis, cyl_ref))
        return _add(
            _add(cyl_origin_3d, _scale(cyl_axis, along)),
            _add(_scale(cyl_ref, radius * math.cos(theta)),
                 _scale(tangent, radius * math.sin(theta)))
        )

    # Inner surface (PCB top, at radius R from rotation center)
    inner_corners = [
        _cyl_point(along_min_cyl, theta_min, inner_radius),
        _cyl_point(along_min_cyl, theta_max, inner_radius),
        _cyl_point(along_max_cyl, theta_max, inner_radius),
        _cyl_point(along_max_cyl, theta_min, inner_radius),
    ]

    # Outer surface (PCB bottom, at radius R + thickness)
    outer_corners = [
        _cyl_point(along_min_cyl, theta_min, outer_radius),
        _cyl_point(along_min_cyl, theta_max, outer_radius),
        _cyl_point(along_max_cyl, theta_max, outer_radius),
        _cyl_point(along_max_cyl, theta_min, outer_radius),
    ]

    return writer.build_bend_solid(
        inner_corners, outer_corners,
        cyl_origin_3d, cyl_axis, cyl_ref,
        inner_radius, outer_radius,
        end_cap_pairs=None  # end caps computed inside build_bend_solid
    )


def _snap_to_plane(vertices_3d, normal):
    """Snap 3D vertices to a best-fit plane to eliminate floating-point drift.

    Uses the centroid and normal to define the plane, then projects all points onto it.
    """
    if not vertices_3d or len(vertices_3d) < 3:
        return vertices_3d

    n = _normalize(normal)
    # Compute centroid
    count = len(vertices_3d)
    cx = sum(v[0] for v in vertices_3d) / count
    cy = sum(v[1] for v in vertices_3d) / count
    cz = sum(v[2] for v in vertices_3d) / count
    center = (cx, cy, cz)

    # Project each vertex onto the plane defined by (center, n)
    result = []
    for v in vertices_3d:
        diff = _sub(v, center)
        dist = _dot(diff, n)
        projected = _sub(v, _scale(n, dist))
        result.append(projected)
    return result


def _build_flat_region_solid(writer, region, recipe_with_defs, thickness,
                             original_segments=None, circle_cutouts=None,
                             drill_holes=None):
    """Build a flat extruded solid for a region.

    When original_segments are provided, recovers arc info for exact geometry.
    """
    normal = _compute_region_normal(region, recipe_with_defs)

    # Try mixed (exact arc) path if original segment info is available
    if original_segments:
        tagged_edges = _recover_arcs(region.outline, original_segments)
        has_arcs = any(e.type == 'arc' for e in tagged_edges)

        if has_arcs:
            top_edges_3d = _transform_tagged_edges_3d(tagged_edges, recipe_with_defs, normal)

            # Holes stay as tessellated polygons (no circle hole recovery for now)
            hole_data = None
            if region.holes:
                hole_data = []
                for hole in region.holes:
                    n_h = len(hole)
                    hole_edges_2d = [TaggedEdge(type="line", start=hole[k],
                                                end=hole[(k+1) % n_h])
                                     for k in range(n_h)]
                    hole_edges_3d = _transform_tagged_edges_3d(
                        hole_edges_2d, recipe_with_defs, normal)
                    hole_data.append(('polygon', hole_edges_3d))

            return writer.build_flat_solid_mixed(
                top_edges_3d, normal, thickness,
                hole_data=hole_data
            )

    # Fallback: all-line path (original behavior)
    outline_3d = _snap_to_plane(
        _transform_polygon_3d(region.outline, recipe_with_defs), normal
    )
    holes_3d = [_snap_to_plane(_transform_polygon_3d(h, recipe_with_defs), normal)
                for h in region.holes]
    return writer.build_flat_solid(outline_3d, holes_3d if holes_3d else None, normal, thickness)


def board_to_step_native(board_geometry, markers, filename, config=None,
                         pcb=None, stiffeners=None):
    """
    Export board geometry to STEP file using pure Python.

    Args:
        board_geometry: BoardGeometry from geometry.py
        markers: list of FoldMarker objects
        filename: output .step file path
        config: FlexConfig (optional, for stiffener settings)
        pcb: KiCadPCB (optional, for stiffener extraction)
        stiffeners: pre-extracted stiffener list (optional)

    Returns:
        True on success, False on error
    """
    try:
        writer = StepWriter()
        thickness = board_geometry.thickness

        # Capture original arc segments for arc recovery
        original_segments = board_geometry.outline.segments if board_geometry.outline.segments else []

        # Capture circle cutout and drill hole info for exact circle recovery
        circle_cutouts = None
        drill_holes = None
        if pcb is not None:
            try:
                circle_cutouts = pcb.get_circle_cutouts()
                drill_holes = pcb.get_drill_holes()
            except Exception:
                pass  # Non-critical: falls back to polygon path

        # Get outline as list of (x,y) tuples
        outline_2d = list(board_geometry.outline.vertices)
        holes_2d = [list(c.vertices) for c in board_geometry.cutouts]

        # Split board into regions (1 subdivision for STEP — smooth cylinders)
        regions = split_board_into_regions(
            outline_2d, holes_2d, markers,
            num_bend_subdivisions=1
        )

        # Build a solid for each region
        for region in regions:
            # Convert fold_recipe markers to FoldDefinitions
            recipe_defs = _recipe_with_fold_defs(region.fold_recipe)

            if _is_bend_region(region.fold_recipe):
                brep_id = _build_bend_region_solid(writer, region, recipe_defs, thickness)
            else:
                brep_id = _build_flat_region_solid(
                    writer, region, recipe_defs, thickness,
                    original_segments=original_segments,
                    circle_cutouts=circle_cutouts,
                    drill_holes=drill_holes,
                )

            writer.add_body(brep_id, f"FLEX_PCB_{region.index}")

        # Handle stiffeners
        if stiffeners is None and config is not None and pcb is not None:
            if config.has_stiffener:
                stiffeners = extract_stiffeners(pcb, config)

        if stiffeners:
            for stiff_idx, stiff in enumerate(stiffeners):
                # Find which region contains the stiffener
                containing_region = find_containing_region(stiff.centroid, regions)
                if containing_region is None:
                    continue

                recipe_defs = _recipe_with_fold_defs(containing_region.fold_recipe)
                stiff_normal = _compute_region_normal(containing_region, recipe_defs)

                # Stiffener is on top or bottom of PCB
                if stiff.side == "bottom":
                    # Offset below PCB (in -normal direction by pcb thickness + stiffener thickness)
                    n = _normalize(stiff_normal)
                    offset_vec = _scale(n, -(thickness + stiff.thickness))
                    outline_3d = [_add(transform_point(p, recipe_defs), offset_vec)
                                  for p in stiff.outline]
                    cutouts_3d = [[_add(transform_point(p, recipe_defs), offset_vec)
                                   for p in cut] for cut in stiff.cutouts]
                    stiff_normal_out = _scale(n, -1.0)
                else:
                    # On top - offset above PCB
                    outline_3d = [transform_point(p, recipe_defs) for p in stiff.outline]
                    cutouts_3d = [[transform_point(p, recipe_defs) for p in cut]
                                  for cut in stiff.cutouts]
                    stiff_normal_out = stiff_normal

                brep_id = writer.build_flat_solid(
                    outline_3d,
                    cutouts_3d if cutouts_3d else None,
                    stiff_normal_out,
                    stiff.thickness
                )
                writer.add_body(brep_id, f"STIFFENER_{stiff_idx}")

        writer.write(filename)
        return True

    except Exception as e:
        print(f"STEP export error: {e}")
        import traceback
        traceback.print_exc()
        return False
