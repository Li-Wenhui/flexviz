"""
Pure-Python STEP AP214 file writer for flex PCB geometry.

Writes B-Rep (Boundary Representation) STEP files directly — no external
dependencies. Supports PLANE and CYLINDRICAL_SURFACE geometry, which is
all that's needed for flat + bent flex PCB regions.

Each solid is a MANIFOLD_SOLID_BREP containing an ADVANCED_BREP_SHAPE_REPRESENTATION.
Multiple named bodies share a single shape representation.
"""

import math
from datetime import datetime


def _round_pt(vals, dp=6):
    """Round a tuple of floats for dedup."""
    return tuple(round(v, dp) for v in vals)


def _normalize(v):
    """Normalize a 3-vector, return tuple."""
    length = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    if length < 1e-15:
        return (0.0, 0.0, 1.0)
    return (v[0] / length, v[1] / length, v[2] / length)


def _cross(a, b):
    return (
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    )


def _dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def _sub(a, b):
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])


def _add(a, b):
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])


def _scale(v, s):
    return (v[0]*s, v[1]*s, v[2]*s)


class StepWriter:
    """
    Builds STEP AP214 files from B-Rep geometry.

    Usage:
        w = StepWriter()
        brep_id = w.build_flat_solid(outline_3d, holes_3d, normal, thickness)
        w.add_body(brep_id, "FLEX_PCB")
        w.write("output.step")
    """

    def __init__(self):
        self._next_id = 1
        self._entities = []  # list of (id, step_text)

        # Dedup caches: key -> entity_id
        self._point_cache = {}      # _round_pt(coords) -> id
        self._direction_cache = {}   # _round_pt(normalized) -> id
        self._vertex_cache = {}      # point_id -> vertex_id
        self._edge_cache = {}        # frozenset((v1_id, v2_id)) -> (ec_id, start_v, end_v)

        # Bodies: list of (brep_id, name)
        self._bodies = []

    # -----------------------------------------------------------------
    # Entity ID management
    # -----------------------------------------------------------------

    def _new_id(self):
        eid = self._next_id
        self._next_id += 1
        return eid

    def _add(self, text):
        """Add entity, return its ID."""
        eid = self._new_id()
        self._entities.append((eid, text))
        return eid

    # -----------------------------------------------------------------
    # Low-level STEP entities (with dedup where appropriate)
    # -----------------------------------------------------------------

    def cartesian_point(self, coords):
        """CARTESIAN_POINT with dedup on rounded coordinates."""
        key = _round_pt(coords)
        if key in self._point_cache:
            return self._point_cache[key]
        vals = ",".join(f"{v:.10g}" for v in key)
        eid = self._add(f"CARTESIAN_POINT('',(  {vals}))")
        self._point_cache[key] = eid
        return eid

    def direction(self, vec):
        """DIRECTION with dedup on normalized+rounded vector.

        Components smaller than 1e-5 are snapped to zero to prevent
        floating-point noise from creating distinct direction entities
        for what is logically the same vector.
        """
        n = _normalize(vec)
        # Snap near-zero components to exactly 0 for clean dedup
        cleaned = tuple(0.0 if abs(c) < 1e-5 else c for c in n)
        # Re-normalize after snapping (in case snapping changed length)
        cleaned = _normalize(cleaned)
        key = _round_pt(cleaned)
        if key in self._direction_cache:
            return self._direction_cache[key]
        vals = ",".join(f"{v:.10g}" for v in key)
        eid = self._add(f"DIRECTION('',(  {vals}))")
        self._direction_cache[key] = eid
        return eid

    def vector(self, dir_id, magnitude):
        """VECTOR entity."""
        return self._add(f"VECTOR('',#{dir_id},{magnitude:.10g})")

    def axis2_placement_3d(self, origin, axis=None, ref_dir=None):
        """AXIS2_PLACEMENT_3D. axis = Z direction, ref_dir = X direction."""
        pt_id = self.cartesian_point(origin)
        if axis is None:
            axis = (0.0, 0.0, 1.0)
        if ref_dir is None:
            ref_dir = (1.0, 0.0, 0.0)
        ax_id = self.direction(axis)
        ref_id = self.direction(ref_dir)
        return self._add(f"AXIS2_PLACEMENT_3D('',#{pt_id},#{ax_id},#{ref_id})")

    def line(self, origin, direction):
        """LINE entity through origin in direction."""
        pt_id = self.cartesian_point(origin)
        dir_id = self.direction(direction)
        vec_id = self.vector(dir_id, 1.0)
        return self._add(f"LINE('',#{pt_id},#{vec_id})")

    def circle(self, center, axis, ref_dir, radius):
        """CIRCLE entity."""
        ax_id = self.axis2_placement_3d(center, axis, ref_dir)
        return self._add(f"CIRCLE('',#{ax_id},{radius:.10g})")

    def plane(self, origin, normal, ref_dir=None):
        """PLANE surface."""
        if ref_dir is None:
            ref_dir = self._make_ref_dir(normal)
        ax_id = self.axis2_placement_3d(origin, normal, ref_dir)
        return self._add(f"PLANE('',#{ax_id})")

    def cylindrical_surface(self, origin, axis, ref_dir, radius):
        """CYLINDRICAL_SURFACE."""
        ax_id = self.axis2_placement_3d(origin, axis, ref_dir)
        return self._add(f"CYLINDRICAL_SURFACE('',#{ax_id},{radius:.10g})")

    def vertex_point(self, point_3d):
        """VERTEX_POINT with dedup per cartesian point."""
        pt_id = self.cartesian_point(point_3d)
        if pt_id in self._vertex_cache:
            return self._vertex_cache[pt_id]
        eid = self._add(f"VERTEX_POINT('',#{pt_id})")
        self._vertex_cache[pt_id] = eid
        return eid

    def edge_curve(self, v1_id, v2_id, curve_id, same_sense=True):
        """EDGE_CURVE."""
        sense = ".T." if same_sense else ".F."
        return self._add(f"EDGE_CURVE('',#{v1_id},#{v2_id},#{curve_id},{sense})")

    def oriented_edge(self, edge_id, orientation=True):
        """ORIENTED_EDGE."""
        orient = ".T." if orientation else ".F."
        return self._add(f"ORIENTED_EDGE('',*,*,#{edge_id},{orient})")

    def edge_loop(self, oriented_edge_ids):
        """EDGE_LOOP from list of oriented edge IDs."""
        refs = ",".join(f"#{eid}" for eid in oriented_edge_ids)
        return self._add(f"EDGE_LOOP('',({refs}))")

    def face_outer_bound(self, loop_id, orientation=True):
        """FACE_OUTER_BOUND."""
        orient = ".T." if orientation else ".F."
        return self._add(f"FACE_OUTER_BOUND('',#{loop_id},{orient})")

    def face_bound(self, loop_id, orientation=True):
        """FACE_BOUND (for holes)."""
        orient = ".T." if orientation else ".F."
        return self._add(f"FACE_BOUND('',#{loop_id},{orient})")

    def advanced_face(self, bound_ids, surface_id, same_sense=True):
        """ADVANCED_FACE."""
        refs = ",".join(f"#{bid}" for bid in bound_ids)
        sense = ".T." if same_sense else ".F."
        return self._add(f"ADVANCED_FACE('',({refs}),#{surface_id},{sense})")

    def closed_shell(self, face_ids):
        """CLOSED_SHELL."""
        refs = ",".join(f"#{fid}" for fid in face_ids)
        return self._add(f"CLOSED_SHELL('',({refs}))")

    def manifold_solid_brep(self, shell_id):
        """MANIFOLD_SOLID_BREP."""
        return self._add(f"MANIFOLD_SOLID_BREP('',#{shell_id})")

    # -----------------------------------------------------------------
    # Helper: compute a ref_dir perpendicular to a given normal
    # -----------------------------------------------------------------

    def _make_ref_dir(self, normal):
        """Find a vector perpendicular to normal for use as ref_direction."""
        n = _normalize(normal)
        # Pick the axis least aligned with normal
        if abs(n[0]) <= abs(n[1]) and abs(n[0]) <= abs(n[2]):
            candidate = (1.0, 0.0, 0.0)
        elif abs(n[1]) <= abs(n[2]):
            candidate = (0.0, 1.0, 0.0)
        else:
            candidate = (0.0, 0.0, 1.0)
        # Gram-Schmidt
        d = _dot(candidate, n)
        ref = (candidate[0] - d*n[0], candidate[1] - d*n[1], candidate[2] - d*n[2])
        return _normalize(ref)

    # -----------------------------------------------------------------
    # High-level: planar face from polygon
    # -----------------------------------------------------------------

    def add_planar_face(self, vertices_3d, normal, holes_3d=None):
        """
        Build an ADVANCED_FACE on a PLANE from a 3D polygon.

        Args:
            vertices_3d: list of (x,y,z) tuples defining the outer boundary
            normal: face normal (x,y,z)
            holes_3d: optional list of hole polygons [[(x,y,z),...], ...]

        Returns:
            face entity ID
        """
        surface_id = self.plane(vertices_3d[0], normal)
        outer_loop_id = self._make_line_loop(vertices_3d)
        outer_bound_id = self.face_outer_bound(outer_loop_id, True)

        bound_ids = [outer_bound_id]
        if holes_3d:
            for hole in holes_3d:
                hole_loop_id = self._make_line_loop(hole)
                hole_bound_id = self.face_bound(hole_loop_id, False)
                bound_ids.append(hole_bound_id)

        return self.advanced_face(bound_ids, surface_id, True)

    def _get_or_create_line_edge(self, p1, p2):
        """Get or create a shared LINE EDGE_CURVE between two points.

        Returns (edge_curve_id, same_direction) where same_direction is True
        if the edge goes from p1 to p2, False if it goes from p2 to p1.
        """
        v1 = self.vertex_point(p1)
        v2 = self.vertex_point(p2)
        key = frozenset((v1, v2))
        if key in self._edge_cache:
            ec_id, start_v, _ = self._edge_cache[key]
            return ec_id, (start_v == v1)
        direction = _sub(p2, p1)
        line_id = self.line(p1, direction)
        ec_id = self.edge_curve(v1, v2, line_id, True)
        self._edge_cache[key] = (ec_id, v1, v2)
        return ec_id, True

    def _make_line_loop(self, vertices_3d):
        """Make an EDGE_LOOP of LINE edges connecting sequential vertices."""
        n = len(vertices_3d)
        oriented_edges = []
        for i in range(n):
            p1 = vertices_3d[i]
            p2 = vertices_3d[(i + 1) % n]
            direction = _sub(p2, p1)
            length = math.sqrt(_dot(direction, direction))
            if length < 1e-12:
                continue
            ec_id, same_dir = self._get_or_create_line_edge(p1, p2)
            oe_id = self.oriented_edge(ec_id, same_dir)
            oriented_edges.append(oe_id)
        return self.edge_loop(oriented_edges)

    # -----------------------------------------------------------------
    # Arc edge creation + mixed loops (line + arc)
    # -----------------------------------------------------------------

    def _get_or_create_arc_edge(self, p1, p2, center_3d, axis, ref_dir, radius,
                                ccw=True):
        """Get or create a shared CIRCLE EDGE_CURVE between two points.

        Args:
            ccw: If True, the arc from p1 to p2 goes CCW (same_sense=True).
                 If False, the arc goes CW (same_sense=False).

        Returns (edge_curve_id, same_direction) where same_direction is True
        if the edge goes from p1 to p2.
        """
        v1 = self.vertex_point(p1)
        v2 = self.vertex_point(p2)
        key = frozenset((v1, v2))
        if key in self._edge_cache:
            ec_id, start_v, _ = self._edge_cache[key]
            return ec_id, (start_v == v1)
        circ_id = self.circle(center_3d, axis, ref_dir, radius)
        ec_id = self.edge_curve(v1, v2, circ_id, ccw)
        self._edge_cache[key] = (ec_id, v1, v2)
        return ec_id, True

    def _make_mixed_loop(self, tagged_edges_3d):
        """Make an EDGE_LOOP from a list of edge dicts with type info.

        Each element is a dict with:
            type: "line" or "arc"
            start: (x,y,z)
            end: (x,y,z)
        For arcs, also:
            center: (x,y,z)
            axis: (x,y,z)
            ref_dir: (x,y,z)
            radius: float
        """
        oriented_edges = []
        for edge in tagged_edges_3d:
            p1 = edge['start']
            p2 = edge['end']
            direction = _sub(p2, p1)
            length = math.sqrt(_dot(direction, direction))
            if length < 1e-12:
                continue

            if edge['type'] == 'arc':
                ec_id, same_dir = self._get_or_create_arc_edge(
                    p1, p2, edge['center'], edge['axis'], edge['ref_dir'], edge['radius'],
                    ccw=edge.get('ccw', True)
                )
            else:
                ec_id, same_dir = self._get_or_create_line_edge(p1, p2)

            oe_id = self.oriented_edge(ec_id, same_dir)
            oriented_edges.append(oe_id)
        return self.edge_loop(oriented_edges)

    def add_planar_face_mixed(self, tagged_edges_3d, normal, hole_loop_ids=None):
        """
        Build an ADVANCED_FACE on a PLANE using a mixed line+arc outer boundary.

        Args:
            tagged_edges_3d: list of edge dicts (see _make_mixed_loop)
            normal: face normal (x,y,z)
            hole_loop_ids: optional list of pre-built edge_loop IDs for holes

        Returns:
            face entity ID
        """
        origin = tagged_edges_3d[0]['start']
        surface_id = self.plane(origin, normal)
        outer_loop_id = self._make_mixed_loop(tagged_edges_3d)
        outer_bound_id = self.face_outer_bound(outer_loop_id, True)

        bound_ids = [outer_bound_id]
        if hole_loop_ids:
            for loop_id in hole_loop_ids:
                hole_bound_id = self.face_bound(loop_id, False)
                bound_ids.append(hole_bound_id)

        return self.advanced_face(bound_ids, surface_id, True)

    # -----------------------------------------------------------------
    # High-level: cylindrical face from 4 corners
    # -----------------------------------------------------------------

    def add_cylindrical_face(self, cyl_origin, cyl_axis, cyl_ref, radius, corners,
                             face_same_sense=True):
        """
        Build an ADVANCED_FACE on a CYLINDRICAL_SURFACE.

        corners: [bottom_start, bottom_end, top_end, top_start]
                 where bottom/top are the straight edges (along axis)
                 and start/end are connected by arcs.

        The face has:
          - 2 arc edges (CIRCLE) at start and end (circumferential)
          - 2 line edges (LINE) at bottom and top (axial)

        Args:
            cyl_origin: cylinder axis origin
            cyl_axis: cylinder axis direction (unit)
            cyl_ref: reference direction for cylinder (radial at angle=0)
            radius: cylinder radius
            corners: [p0, p1, p2, p3] - 4 corner points of the patch
            face_same_sense: if True, face normal = surface outward normal;
                            if False, face normal = -surface outward normal

        Returns:
            face entity ID
        """
        surf_id = self.cylindrical_surface(cyl_origin, cyl_axis, cyl_ref, radius)

        p0, p1, p2, p3 = corners  # bottom_start, bottom_end, top_end, top_start

        v0 = self.vertex_point(p0)
        v1 = self.vertex_point(p1)
        v2 = self.vertex_point(p2)
        v3 = self.vertex_point(p3)

        # Arc circle centers must be at the correct axial position on the cylinder axis
        along_0 = _dot(_sub(p0, cyl_origin), cyl_axis)
        arc_center_bottom = _add(cyl_origin, _scale(cyl_axis, along_0))
        along_3 = _dot(_sub(p3, cyl_origin), cyl_axis)
        arc_center_top = _add(cyl_origin, _scale(cyl_axis, along_3))

        # Bottom arc: p0 -> p1 (circumferential, positive parameter direction)
        circ_bottom = self.circle(arc_center_bottom, cyl_axis, cyl_ref, radius)
        ec_bottom = self.edge_curve(v0, v1, circ_bottom, True)

        # Right line: p1 -> p2 (axial) — shared with adjacent faces
        ec_right, right_same_dir = self._get_or_create_line_edge(p1, p2)

        # Top arc: p2 -> p3 (circumferential, negative parameter direction)
        # Use same_sense=False so edge goes v2->v3 in decreasing angle = short CW arc
        circ_top = self.circle(arc_center_top, cyl_axis, cyl_ref, radius)
        ec_top = self.edge_curve(v2, v3, circ_top, False)

        # Left line: p3 -> p0 (axial) — shared with adjacent faces
        ec_left, left_same_dir = self._get_or_create_line_edge(p3, p0)

        # Edge loop: p0->p1 (arc), p1->p2 (line), p2->p3 (arc), p3->p0 (line)
        oe0 = self.oriented_edge(ec_bottom, True)
        oe1 = self.oriented_edge(ec_right, right_same_dir)
        oe2 = self.oriented_edge(ec_top, True)
        oe3 = self.oriented_edge(ec_left, left_same_dir)

        loop_id = self.edge_loop([oe0, oe1, oe2, oe3])
        bound_id = self.face_outer_bound(loop_id, True)

        return self.advanced_face([bound_id], surf_id, face_same_sense)

    # -----------------------------------------------------------------
    # High-level: build a flat (extruded) solid
    # -----------------------------------------------------------------

    def build_flat_solid(self, outline_3d, holes_3d, normal, thickness):
        """
        Build a flat box-like solid from a polygon outline.

        Creates top face, bottom face (offset by -normal*thickness),
        and side faces connecting them.

        Args:
            outline_3d: list of (x,y,z) for the top face boundary
            holes_3d: list of hole polygons [[(x,y,z),...], ...] or None
            normal: outward surface normal of the top face
            thickness: extrusion distance along -normal

        Returns:
            MANIFOLD_SOLID_BREP entity ID
        """
        n = _normalize(normal)
        offset = _scale(n, -thickness)

        # Bottom face vertices (offset along -normal)
        bottom_outline = [_add(v, offset) for v in outline_3d]
        bottom_holes = None
        if holes_3d:
            bottom_holes = [[_add(v, offset) for v in hole] for hole in holes_3d]

        # Top face (normal pointing outward)
        top_face_id = self.add_planar_face(outline_3d, n, holes_3d)

        # Bottom face (normal pointing inward = -n, but reversed polygon)
        neg_n = _scale(n, -1.0)
        bottom_reversed = list(reversed(bottom_outline))
        bottom_holes_reversed = None
        if bottom_holes:
            bottom_holes_reversed = [list(reversed(h)) for h in bottom_holes]
        bottom_face_id = self.add_planar_face(bottom_reversed, neg_n, bottom_holes_reversed)

        # Side faces - connect each edge of outline to corresponding bottom edge
        side_face_ids = self._build_side_faces(outline_3d, bottom_outline, n)

        # Side faces for holes (inside surfaces)
        if holes_3d and bottom_holes:
            for hole_top, hole_bot in zip(holes_3d, bottom_holes):
                # Hole sides face inward, so reverse winding
                hole_sides = self._build_side_faces(hole_bot, hole_top, neg_n)
                side_face_ids.extend(hole_sides)

        all_faces = [top_face_id, bottom_face_id] + side_face_ids
        shell_id = self.closed_shell(all_faces)
        return self.manifold_solid_brep(shell_id)

    def _build_side_faces(self, top_verts, bottom_verts, top_normal):
        """Build side faces connecting top and bottom polygon outlines."""
        n_verts = len(top_verts)
        face_ids = []
        for i in range(n_verts):
            j = (i + 1) % n_verts
            # Quad: top[i], top[j], bottom[j], bottom[i]
            quad = [top_verts[i], top_verts[j], bottom_verts[j], bottom_verts[i]]

            # Compute face normal (outward from the solid)
            edge1 = _sub(quad[1], quad[0])
            edge2 = _sub(quad[3], quad[0])
            face_normal = _normalize(_cross(edge1, edge2))

            face_id = self.add_planar_face(quad, face_normal)
            face_ids.append(face_id)
        return face_ids

    # -----------------------------------------------------------------
    # Mixed side faces (line -> PLANE, arc -> CYLINDRICAL_SURFACE)
    # -----------------------------------------------------------------

    def _build_side_faces_mixed(self, top_edges_3d, bottom_edges_3d, top_normal):
        """Build side faces for mixed line+arc edges.

        For each corresponding top/bottom edge pair:
        - Line edge -> PLANE side face (quad)
        - Arc edge -> CYLINDRICAL_SURFACE side face

        Uses _get_or_create_arc_edge / _get_or_create_line_edge so that
        all edges are shared with the top/bottom faces (watertight shell).

        Args:
            top_edges_3d: list of edge dicts (type, start, end, and for arcs: center, axis, ref_dir, radius)
            bottom_edges_3d: list of edge dicts (same structure, offset from top)
            top_normal: outward normal of the top face (used for extrusion direction)

        Returns:
            list of face entity IDs
        """
        face_ids = []
        for top_e, bot_e in zip(top_edges_3d, bottom_edges_3d):
            # Skip degenerate edges
            d = _sub(top_e['end'], top_e['start'])
            if math.sqrt(_dot(d, d)) < 1e-12:
                continue

            if top_e['type'] == 'arc':
                # Cylindrical side face — manually built to reuse cached edges
                n = _normalize(top_normal)
                cyl_center = top_e['center']
                radius = top_e['radius']
                arc_ccw = top_e.get('ccw', True)
                ref = _normalize(_sub(top_e['start'], cyl_center))

                surf_id = self.cylindrical_surface(cyl_center, n, ref, radius)

                # Top arc: top_e start -> top_e end (reuses cached edge from top face)
                ec_top, top_same = self._get_or_create_arc_edge(
                    top_e['start'], top_e['end'],
                    top_e['center'], top_e['axis'], top_e['ref_dir'], radius,
                    ccw=arc_ccw
                )
                # Bottom arc: bot_e start -> bot_e end (reuses cached edge from bottom face)
                ec_bot, bot_same = self._get_or_create_arc_edge(
                    bot_e['start'], bot_e['end'],
                    bot_e['center'], bot_e['axis'], bot_e['ref_dir'], radius,
                    ccw=arc_ccw
                )
                # Axial lines connecting top to bottom (shared with adjacent side faces)
                ec_right, right_same = self._get_or_create_line_edge(top_e['end'], bot_e['end'])
                ec_left, left_same = self._get_or_create_line_edge(bot_e['start'], top_e['start'])

                # Loop: top_start->top_end (arc), top_end->bot_end (line),
                #        bot_end->bot_start (arc reversed), bot_start->top_start (line)
                oe_top = self.oriented_edge(ec_top, top_same)
                oe_right = self.oriented_edge(ec_right, right_same)
                oe_bot = self.oriented_edge(ec_bot, not bot_same)  # reversed
                oe_left = self.oriented_edge(ec_left, left_same)

                loop_id = self.edge_loop([oe_top, oe_right, oe_bot, oe_left])
                bound_id = self.face_outer_bound(loop_id, True)
                # CCW arc = convex corner, cylinder outward = solid outward → True
                # CW arc = concave corner, cylinder outward = solid inward → False
                face_id = self.advanced_face([bound_id], surf_id, arc_ccw)
                face_ids.append(face_id)
            else:
                # Planar side face (quad)
                quad = [top_e['start'], top_e['end'], bot_e['end'], bot_e['start']]
                edge1 = _sub(quad[1], quad[0])
                edge2 = _sub(quad[3], quad[0])
                face_normal = _normalize(_cross(edge1, edge2))
                face_id = self.add_planar_face(quad, face_normal)
                face_ids.append(face_id)
        return face_ids

    def _build_circle_hole_faces(self, center_top, center_bot, radius, normal):
        """Build faces for a full-circle hole through the PCB.

        Splits the circle into 2 semicircular arcs to avoid seam issues.

        Returns:
            (top_loop_id, bottom_loop_id, [side_face_ids])
        """
        n = _normalize(normal)
        ref_dir = self._make_ref_dir(n)
        # Compute perpendicular to both normal and ref_dir
        perp = _normalize(_cross(n, ref_dir))

        # Two points on the circle (diametrically opposite)
        t1 = _add(center_top, _scale(ref_dir, radius))
        t2 = _add(center_top, _scale(ref_dir, -radius))
        b1 = _add(center_bot, _scale(ref_dir, radius))
        b2 = _add(center_bot, _scale(ref_dir, -radius))

        # Top loop: two semicircular arcs (t1->t2 and t2->t1)
        circ_top = self.circle(center_top, n, ref_dir, radius)
        vt1 = self.vertex_point(t1)
        vt2 = self.vertex_point(t2)
        ec_top_1 = self.edge_curve(vt1, vt2, circ_top, True)   # CCW first half
        ec_top_2 = self.edge_curve(vt2, vt1, circ_top, True)   # CCW second half

        oe_top_1 = self.oriented_edge(ec_top_1, True)
        oe_top_2 = self.oriented_edge(ec_top_2, True)
        top_loop_id = self.edge_loop([oe_top_1, oe_top_2])

        # Bottom loop: two semicircular arcs (reversed winding for bottom face)
        circ_bot = self.circle(center_bot, n, ref_dir, radius)
        vb1 = self.vertex_point(b1)
        vb2 = self.vertex_point(b2)
        ec_bot_1 = self.edge_curve(vb1, vb2, circ_bot, True)
        ec_bot_2 = self.edge_curve(vb2, vb1, circ_bot, True)

        oe_bot_1 = self.oriented_edge(ec_bot_1, False)
        oe_bot_2 = self.oriented_edge(ec_bot_2, False)
        bottom_loop_id = self.edge_loop([oe_bot_1, oe_bot_2])

        # Wall: 2 cylindrical patches connecting top arcs to bottom arcs
        # Each patch: top_arc, axial_line, bottom_arc(reversed), axial_line(reversed)
        side_face_ids = []

        # Axial line edges (shared between the two patches)
        ec_ax1, _ = self._get_or_create_line_edge(t1, b1)
        ec_ax2, _ = self._get_or_create_line_edge(t2, b2)

        # Cylinder surface for the wall (axis = normal, center = center_top works for both)
        # Actually use center on the axis line (doesn't matter for cylinder definition)
        cyl_surf = self.cylindrical_surface(center_top, n, ref_dir, radius)

        # Patch 1: t1->t2 (top arc), t2->b2 (line), b2->b1 (bottom arc reversed), b1->t1 (line)
        oe_p1_arc_t = self.oriented_edge(ec_top_1, True)      # t1->t2
        oe_p1_line_r = self.oriented_edge(ec_ax2, True)        # t2->b2
        oe_p1_arc_b = self.oriented_edge(ec_bot_1, False)      # b2->b1 (reverse of b1->b2)
        oe_p1_line_l = self.oriented_edge(ec_ax1, False)       # b1->t1 (reverse of t1->b1)
        loop_p1 = self.edge_loop([oe_p1_arc_t, oe_p1_line_r, oe_p1_arc_b, oe_p1_line_l])
        bound_p1 = self.face_outer_bound(loop_p1, True)
        face_p1 = self.advanced_face([bound_p1], cyl_surf, False)  # inner wall faces inward
        side_face_ids.append(face_p1)

        # Patch 2: t2->t1 (top arc), t1->b1 (line), b1->b2 (bottom arc reversed), b2->t2 (line)
        oe_p2_arc_t = self.oriented_edge(ec_top_2, True)       # t2->t1
        oe_p2_line_r = self.oriented_edge(ec_ax1, True)        # t1->b1
        oe_p2_arc_b = self.oriented_edge(ec_bot_2, False)      # b1->b2 (reverse of b2->b1)
        oe_p2_line_l = self.oriented_edge(ec_ax2, False)       # b2->t2 (reverse of t2->b2)
        loop_p2 = self.edge_loop([oe_p2_arc_t, oe_p2_line_r, oe_p2_arc_b, oe_p2_line_l])
        bound_p2 = self.face_outer_bound(loop_p2, True)
        face_p2 = self.advanced_face([bound_p2], cyl_surf, False)
        side_face_ids.append(face_p2)

        return top_loop_id, bottom_loop_id, side_face_ids

    # -----------------------------------------------------------------
    # High-level: build a flat solid with mixed line+arc edges
    # -----------------------------------------------------------------

    def build_flat_solid_mixed(self, top_edges_3d, normal, thickness,
                                hole_data=None):
        """
        Build a flat solid using mixed line+arc edges for exact geometry.

        Args:
            top_edges_3d: list of edge dicts for the outer boundary
                         (type, start, end, and for arcs: center, axis, ref_dir, radius)
            normal: outward surface normal of the top face
            thickness: extrusion distance along -normal
            hole_data: list of hole descriptors, each is one of:
                       ('circle', (cx_top, cy_top, cz_top), (cx_bot, cy_bot, cz_bot), radius)
                       ('polygon', [edge_dicts_3d])

        Returns:
            MANIFOLD_SOLID_BREP entity ID
        """
        n = _normalize(normal)
        offset = _scale(n, -thickness)

        # Build bottom edges (offset from top)
        bottom_edges_3d = []
        for e in top_edges_3d:
            be = dict(e)
            be['start'] = _add(e['start'], offset)
            be['end'] = _add(e['end'], offset)
            if e['type'] == 'arc' and e.get('center'):
                be['center'] = _add(e['center'], offset)
            bottom_edges_3d.append(be)

        # Reverse bottom edges for opposite winding
        bottom_edges_reversed = []
        for e in reversed(bottom_edges_3d):
            re_e = dict(e)
            re_e['start'] = e['end']
            re_e['end'] = e['start']
            # Flip arc sense: CCW A→B becomes CW B→A (same physical arc)
            if e['type'] == 'arc':
                re_e['ccw'] = not e.get('ccw', True)
            bottom_edges_reversed.append(re_e)

        # Process holes
        top_hole_loop_ids = []
        bottom_hole_loop_ids = []
        all_side_face_ids = []

        if hole_data:
            for hd in hole_data:
                if hd[0] == 'circle':
                    _, center_top, center_bot, radius = hd
                    top_loop, bot_loop, cyl_sides = self._build_circle_hole_faces(
                        center_top, center_bot, radius, n
                    )
                    top_hole_loop_ids.append(top_loop)
                    bottom_hole_loop_ids.append(bot_loop)
                    all_side_face_ids.extend(cyl_sides)
                else:
                    # 'polygon' hole with tagged edges
                    _, hole_top_edges = hd
                    # Build bottom hole edges
                    hole_bot_edges = []
                    for e in hole_top_edges:
                        be = dict(e)
                        be['start'] = _add(e['start'], offset)
                        be['end'] = _add(e['end'], offset)
                        if e['type'] == 'arc' and e.get('center'):
                            be['center'] = _add(e['center'], offset)
                        hole_bot_edges.append(be)

                    # Hole side faces (inward-facing)
                    neg_n = _scale(n, -1.0)
                    hole_sides = self._build_side_faces_mixed(
                        hole_bot_edges, hole_top_edges, neg_n
                    )
                    all_side_face_ids.extend(hole_sides)

                    # Hole loops for top and bottom faces
                    top_hole_loop_id = self._make_mixed_loop(hole_top_edges)
                    top_hole_loop_ids.append(top_hole_loop_id)

                    hole_bot_reversed = []
                    for e in reversed(hole_bot_edges):
                        re_e = dict(e)
                        re_e['start'] = e['end']
                        re_e['end'] = e['start']
                        hole_bot_reversed.append(re_e)
                    bottom_hole_loop_id = self._make_mixed_loop(hole_bot_reversed)
                    bottom_hole_loop_ids.append(bottom_hole_loop_id)

        # Top face
        top_face_id = self.add_planar_face_mixed(
            top_edges_3d, n, top_hole_loop_ids if top_hole_loop_ids else None
        )

        # Bottom face (reversed winding, -normal)
        neg_n = _scale(n, -1.0)
        bottom_face_id = self.add_planar_face_mixed(
            bottom_edges_reversed, neg_n,
            bottom_hole_loop_ids if bottom_hole_loop_ids else None
        )

        # Outer side faces
        outer_side_ids = self._build_side_faces_mixed(top_edges_3d, bottom_edges_3d, n)
        all_side_face_ids.extend(outer_side_ids)

        all_faces = [top_face_id, bottom_face_id] + all_side_face_ids
        shell_id = self.closed_shell(all_faces)
        return self.manifold_solid_brep(shell_id)

    # -----------------------------------------------------------------
    # High-level: build a bend (cylindrical) solid
    # -----------------------------------------------------------------

    def build_bend_solid(self, inner_corners, outer_corners, cyl_origin, cyl_axis,
                         cyl_ref, inner_radius, outer_radius, end_cap_pairs):
        """
        Build a curved solid for a bend zone with proper shared-edge topology.

        The solid has 6 faces sharing 12 edges:
          - Inner cylindrical face (radius R)
          - Outer cylindrical face (radius R + thickness)
          - 2 end cap faces (planar, radial cross-sections)
          - 2 side faces (planar annular sectors perpendicular to axis)

        Corner layout (inner and outer have same topology):
            p3 ---arc_top--- p2
            |                 |
          axial_left      axial_right
            |                 |
            p0 ---arc_bot--- p1

        Returns:
            MANIFOLD_SOLID_BREP entity ID
        """
        ip0, ip1, ip2, ip3 = inner_corners
        op0, op1, op2, op3 = outer_corners

        # --- Vertices (8 total) ---
        vi0 = self.vertex_point(ip0)
        vi1 = self.vertex_point(ip1)
        vi2 = self.vertex_point(ip2)
        vi3 = self.vertex_point(ip3)
        vo0 = self.vertex_point(op0)
        vo1 = self.vertex_point(op1)
        vo2 = self.vertex_point(op2)
        vo3 = self.vertex_point(op3)

        # --- Arc circle centers ---
        along_bot = _dot(_sub(ip0, cyl_origin), cyl_axis)
        along_top = _dot(_sub(ip3, cyl_origin), cyl_axis)
        center_bot = _add(cyl_origin, _scale(cyl_axis, along_bot))
        center_top = _add(cyl_origin, _scale(cyl_axis, along_top))

        # --- 12 shared edges ---
        # 4 arc edges (circle curves on cylindrical surfaces)
        circ_inner_bot = self.circle(center_bot, cyl_axis, cyl_ref, inner_radius)
        circ_inner_top = self.circle(center_top, cyl_axis, cyl_ref, inner_radius)
        circ_outer_bot = self.circle(center_bot, cyl_axis, cyl_ref, outer_radius)
        circ_outer_top = self.circle(center_top, cyl_axis, cyl_ref, outer_radius)

        ec_arc_ib = self.edge_curve(vi0, vi1, circ_inner_bot, True)   # inner bottom: CCW
        ec_arc_it = self.edge_curve(vi2, vi3, circ_inner_top, False)  # inner top: CW (v2→v3)
        ec_arc_ob = self.edge_curve(vo0, vo1, circ_outer_bot, True)   # outer bottom: CCW
        ec_arc_ot = self.edge_curve(vo2, vo3, circ_outer_top, False)  # outer top: CW (v2→v3)

        # 4 axial line edges (along cylinder axis, shared between cyl faces & end caps)
        ec_ax_i_left, _ = self._get_or_create_line_edge(ip3, ip0)    # inner left
        ec_ax_i_right, _ = self._get_or_create_line_edge(ip1, ip2)   # inner right
        ec_ax_o_left, _ = self._get_or_create_line_edge(op3, op0)    # outer left
        ec_ax_o_right, _ = self._get_or_create_line_edge(op1, op2)   # outer right

        # 4 radial line edges (connecting inner to outer, shared between end caps & side faces)
        ec_rad_00, _ = self._get_or_create_line_edge(ip0, op0)  # at theta_min, along_min
        ec_rad_11, _ = self._get_or_create_line_edge(ip1, op1)  # at theta_max, along_min
        ec_rad_22, _ = self._get_or_create_line_edge(ip2, op2)  # at theta_max, along_max
        ec_rad_33, _ = self._get_or_create_line_edge(ip3, op3)  # at theta_min, along_max

        # Helper to get oriented edge with correct direction from edge cache
        def _oe(ec_id, v_start, v_end):
            """Create oriented edge, checking if direction matches the edge_curve."""
            key = frozenset((self.vertex_point(v_start) if isinstance(v_start, tuple) else v_start,
                             self.vertex_point(v_end) if isinstance(v_end, tuple) else v_end))
            if key in self._edge_cache:
                _, start_v, _ = self._edge_cache[key]
                same_dir = (start_v == (self.vertex_point(v_start) if isinstance(v_start, tuple) else v_start))
                return self.oriented_edge(ec_id, same_dir)
            return self.oriented_edge(ec_id, True)

        # --- Face 1: Inner cylindrical (face normal inward = same_sense False) ---
        inner_surf = self.cylindrical_surface(cyl_origin, cyl_axis, cyl_ref, inner_radius)
        oe_ib_fwd = self.oriented_edge(ec_arc_ib, True)     # i0→i1 CCW
        oe_ir_fwd = self.oriented_edge(ec_ax_i_right, True)  # i1→i2
        oe_it_fwd = self.oriented_edge(ec_arc_it, True)      # i2→i3 (edge is CW, oriented .T.)
        oe_il_fwd = self.oriented_edge(ec_ax_i_left, True)   # i3→i0
        inner_loop = self.edge_loop([oe_ib_fwd, oe_ir_fwd, oe_it_fwd, oe_il_fwd])
        inner_bound = self.face_outer_bound(inner_loop, True)
        inner_face = self.advanced_face([inner_bound], inner_surf, False)

        # --- Face 2: Outer cylindrical (face normal outward = same_sense True) ---
        outer_surf = self.cylindrical_surface(cyl_origin, cyl_axis, cyl_ref, outer_radius)
        oe_ob_fwd = self.oriented_edge(ec_arc_ob, True)      # o0→o1 CCW
        oe_or_fwd = self.oriented_edge(ec_ax_o_right, True)   # o1→o2
        oe_ot_fwd = self.oriented_edge(ec_arc_ot, True)       # o2→o3 (edge is CW, oriented .T.)
        oe_ol_fwd = self.oriented_edge(ec_ax_o_left, True)    # o3→o0
        outer_loop = self.edge_loop([oe_ob_fwd, oe_or_fwd, oe_ot_fwd, oe_ol_fwd])
        outer_bound = self.face_outer_bound(outer_loop, True)
        outer_face = self.advanced_face([outer_bound], outer_surf, True)

        # --- Face 3: Start end cap (at theta_min): i0, o0, o3, i3 ---
        sc_e1 = _sub(op0, ip0)
        sc_e2 = _sub(ip3, ip0)
        sc_normal = _normalize(_cross(sc_e1, sc_e2))
        start_surf = self.plane(ip0, sc_normal)
        oe_sc0 = self.oriented_edge(ec_rad_00, True)          # i0→o0
        oe_sc1 = self.oriented_edge(ec_ax_o_left, False)      # o0→o3 (reverse of o3→o0)
        oe_sc2 = self.oriented_edge(ec_rad_33, False)         # o3→i3 (reverse of i3→o3)
        oe_sc3 = self.oriented_edge(ec_ax_i_left, False)      # i3→i0 (reverse of... wait)
        # ec_ax_i_left was created as i3→i0, so forward = i3→i0. We want i3→i0 here.
        # But loop is i0→o0→o3→i3→i0. The last edge is i3→i0 which IS the forward direction.
        oe_sc3 = self.oriented_edge(ec_ax_i_left, True)       # i3→i0 (forward)
        sc_loop = self.edge_loop([oe_sc0, oe_sc1, oe_sc2, oe_sc3])
        sc_bound = self.face_outer_bound(sc_loop, True)
        start_face = self.advanced_face([sc_bound], start_surf, True)

        # --- Face 4: End cap (at theta_max): i1, i2, o2, o1 ---
        ec_e1 = _sub(ip2, ip1)
        ec_e2 = _sub(op1, ip1)
        ec_normal = _normalize(_cross(ec_e1, ec_e2))
        end_surf = self.plane(ip1, ec_normal)
        oe_ec0 = self.oriented_edge(ec_ax_i_right, True)      # i1→i2
        oe_ec1 = self.oriented_edge(ec_rad_22, True)           # i2→o2
        oe_ec2 = self.oriented_edge(ec_ax_o_right, False)      # o2→o1 (reverse of o1→o2)
        oe_ec3 = self.oriented_edge(ec_rad_11, False)          # o1→i1 (reverse of i1→o1)
        ec_loop = self.edge_loop([oe_ec0, oe_ec1, oe_ec2, oe_ec3])
        ec_bound = self.face_outer_bound(ec_loop, True)
        end_face = self.advanced_face([ec_bound], end_surf, True)

        # --- Face 5: Bottom side (at along_min): i0, i1, o1, o0 ---
        # This is an annular sector face with arc edges on inner/outer radii
        bs_e1 = _sub(ip1, ip0)
        bs_e2 = _sub(op0, ip0)
        bs_normal = _normalize(_cross(bs_e1, bs_e2))
        bot_surf = self.plane(ip0, bs_normal)
        oe_bs0 = self.oriented_edge(ec_arc_ib, True)          # i0→i1 (inner arc, shared)
        oe_bs1 = self.oriented_edge(ec_rad_11, True)           # i1→o1
        oe_bs2 = self.oriented_edge(ec_arc_ob, False)          # o1→o0 (outer arc reversed)
        oe_bs3 = self.oriented_edge(ec_rad_00, False)          # o0→i0 (reverse of i0→o0)
        bs_loop = self.edge_loop([oe_bs0, oe_bs1, oe_bs2, oe_bs3])
        bs_bound = self.face_outer_bound(bs_loop, True)
        bot_face = self.advanced_face([bs_bound], bot_surf, True)

        # --- Face 6: Top side (at along_max): i3, o3, o2, i2 ---
        ts_e1 = _sub(op3, ip3)
        ts_e2 = _sub(ip2, ip3)
        ts_normal = _normalize(_cross(ts_e1, ts_e2))
        top_surf = self.plane(ip3, ts_normal)
        oe_ts0 = self.oriented_edge(ec_rad_33, True)           # i3→o3
        oe_ts1 = self.oriented_edge(ec_arc_ot, False)          # o3→o2 (reverse: edge is o2→o3)
        oe_ts2 = self.oriented_edge(ec_rad_22, False)          # o2→i2 (reverse of i2→o2)
        oe_ts3 = self.oriented_edge(ec_arc_it, False)          # i2→i3 (reverse: edge is i2→i3 CW)
        # Wait: ec_arc_it is EDGE_CURVE(vi2, vi3, circle, False). Forward = v2→v3.
        # We want i2→i3 direction which IS forward. But we're going i3→o3→o2→i2→(back to i3).
        # So last edge should be i2→i3 = forward direction of ec_arc_it.
        oe_ts3 = self.oriented_edge(ec_arc_it, True)           # i2→i3 (forward of CW edge)
        # Actually loop is i3→o3→o2→i2→i3. Last step i2→i3 is forward of ec_arc_it.
        ts_loop = self.edge_loop([oe_ts0, oe_ts1, oe_ts2, oe_ts3])
        ts_bound = self.face_outer_bound(ts_loop, True)
        top_face = self.advanced_face([ts_bound], top_surf, True)

        face_ids = [inner_face, outer_face, start_face, end_face, bot_face, top_face]
        shell_id = self.closed_shell(face_ids)
        return self.manifold_solid_brep(shell_id)

    # -----------------------------------------------------------------
    # Body management
    # -----------------------------------------------------------------

    def add_body(self, brep_id, name="Body"):
        """Register a named solid for output."""
        self._bodies.append((brep_id, name))

    # -----------------------------------------------------------------
    # STEP file output
    # -----------------------------------------------------------------

    def write(self, filename):
        """Write complete STEP AP214 file."""
        now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        lines = []
        lines.append("ISO-10303-21;")
        lines.append("HEADER;")
        lines.append(f"FILE_DESCRIPTION(('FreeForm'),'2;1');")
        lines.append(f"FILE_NAME('{filename}','{now}',('FlexViz'),(''),'FlexViz STEP Writer','FlexViz','');")
        lines.append("FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));")
        lines.append("ENDSEC;")
        lines.append("DATA;")

        # Write all geometry entities
        for eid, text in self._entities:
            lines.append(f"#{eid}={text};")

        # Write product structure for each body
        next_id = self._next_id

        # Collect all brep IDs for the shape representation
        brep_refs = ",".join(f"#{bid}" for bid, _ in self._bodies)

        # Allocate application context ID first so product chain can reference it directly
        app_ctx_id = next_id; next_id += 1
        app_proto = next_id; next_id += 1

        lines.append(f"#{app_ctx_id}=APPLICATION_CONTEXT("
                     f"'core data for automotive mechanical design processes');")
        lines.append(f"#{app_proto}=APPLICATION_PROTOCOL_DEFINITION('international standard',"
                     f"'automotive_design',2000,#{app_ctx_id});")

        # Context entities (shared)
        ctx_origin = next_id; next_id += 1
        ctx_dir_z = next_id; next_id += 1
        ctx_dir_x = next_id; next_id += 1
        ctx_axis = next_id; next_id += 1
        geom_ctx = next_id; next_id += 1
        length_unit = next_id; next_id += 1
        angle_unit = next_id; next_id += 1
        solid_angle = next_id; next_id += 1
        measure_len = next_id; next_id += 1
        measure_angle = next_id; next_id += 1
        measure_solid = next_id; next_id += 1
        uncert = next_id; next_id += 1
        shape_rep = next_id; next_id += 1

        lines.append(f"#{ctx_origin}=CARTESIAN_POINT('',(  0.,0.,0.));")
        lines.append(f"#{ctx_dir_z}=DIRECTION('',(  0.,0.,1.));")
        lines.append(f"#{ctx_dir_x}=DIRECTION('',(  1.,0.,0.));")
        lines.append(f"#{ctx_axis}=AXIS2_PLACEMENT_3D('',#{ctx_origin},#{ctx_dir_z},#{ctx_dir_x});")

        lines.append(f"#{geom_ctx}=("
                     f"GEOMETRIC_REPRESENTATION_CONTEXT(3)"
                     f"GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{uncert}))"
                     f"GLOBAL_UNIT_ASSIGNED_CONTEXT((#{length_unit},#{angle_unit},#{solid_angle}))"
                     f"REPRESENTATION_CONTEXT('Context3D','3D Context'));")

        lines.append(f"#{length_unit}=(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.));")
        lines.append(f"#{angle_unit}=(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.));")
        lines.append(f"#{solid_angle}=(NAMED_UNIT(*)SI_UNIT($,.STERADIAN.)SOLID_ANGLE_UNIT());")
        lines.append(f"#{measure_len}=LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-03),#{length_unit});")
        lines.append(f"#{measure_angle}=PLANE_ANGLE_MEASURE_WITH_UNIT(PLANE_ANGLE_MEASURE(1.E-03),#{angle_unit});")
        lines.append(f"#{measure_solid}=SOLID_ANGLE_MEASURE_WITH_UNIT(SOLID_ANGLE_MEASURE(1.E-03),#{solid_angle});")
        lines.append(f"#{uncert}=UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-07),#{length_unit},"
                     f"'distance_accuracy_value','confusion accuracy');")

        # Shape representation with all brep solids
        lines.append(f"#{shape_rep}=ADVANCED_BREP_SHAPE_REPRESENTATION('',({brep_refs},#{ctx_axis}),#{geom_ctx});")

        # Product chain per body (references app_ctx_id directly)
        for body_idx, (brep_id, body_name) in enumerate(self._bodies):
            prod = next_id; next_id += 1
            prod_ctx = next_id; next_id += 1
            pdf = next_id; next_id += 1
            pd = next_id; next_id += 1
            pds = next_id; next_id += 1
            sdr = next_id; next_id += 1

            lines.append(f"#{prod}=PRODUCT('{body_name}','{body_name}','',(#{prod_ctx}));")
            lines.append(f"#{prod_ctx}=PRODUCT_CONTEXT('',#{app_ctx_id},'mechanical');")
            lines.append(f"#{pdf}=PRODUCT_DEFINITION_FORMATION('','',#{prod});")
            lines.append(f"#{pd}=PRODUCT_DEFINITION('design','',#{pdf},#{app_ctx_id});")
            lines.append(f"#{pds}=PRODUCT_DEFINITION_SHAPE('','',#{pd});")
            lines.append(f"#{sdr}=SHAPE_DEFINITION_REPRESENTATION(#{pds},#{shape_rep});")

        lines.append("ENDSEC;")
        lines.append("END-ISO-10303-21;")

        with open(filename, 'w') as f:
            f.write("\n".join(lines))
            f.write("\n")
