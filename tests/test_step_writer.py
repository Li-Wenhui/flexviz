"""Unit tests for step_writer and step_export modules."""

import pytest
import math
import os
import re
import tempfile
from pathlib import Path

from step_writer import StepWriter, _normalize, _round_pt
from step_export import (
    board_to_step_native, is_step_export_available,
    TaggedEdge, _point_on_arc, _recover_arcs, _recover_circle_holes,
    _sample_arc_2d, _transform_tagged_edges_3d_bend,
)
from geometry import OutlineSegment


class TestStepWriterDedup:
    """Test entity deduplication."""

    def test_cartesian_point_dedup(self):
        w = StepWriter()
        id1 = w.cartesian_point((1.0, 2.0, 3.0))
        id2 = w.cartesian_point((1.0, 2.0, 3.0))
        assert id1 == id2

    def test_cartesian_point_dedup_within_rounding(self):
        w = StepWriter()
        id1 = w.cartesian_point((1.0, 2.0, 3.0))
        id2 = w.cartesian_point((1.0000001, 2.0000001, 3.0000001))
        # Within 6dp rounding
        assert id1 == id2

    def test_cartesian_point_different(self):
        w = StepWriter()
        id1 = w.cartesian_point((1.0, 2.0, 3.0))
        id2 = w.cartesian_point((1.0, 2.0, 4.0))
        assert id1 != id2

    def test_direction_dedup(self):
        w = StepWriter()
        id1 = w.direction((0.0, 0.0, 1.0))
        id2 = w.direction((0.0, 0.0, 1.0))
        assert id1 == id2

    def test_direction_normalized_dedup(self):
        w = StepWriter()
        id1 = w.direction((0.0, 0.0, 1.0))
        id2 = w.direction((0.0, 0.0, 2.0))  # Same direction, different magnitude
        assert id1 == id2

    def test_vertex_point_dedup(self):
        w = StepWriter()
        id1 = w.vertex_point((1.0, 2.0, 3.0))
        id2 = w.vertex_point((1.0, 2.0, 3.0))
        assert id1 == id2


class TestStepWriterEntities:
    """Test individual entity creation."""

    def test_axis2_placement_3d(self):
        w = StepWriter()
        eid = w.axis2_placement_3d((0, 0, 0), (0, 0, 1), (1, 0, 0))
        assert eid > 0

    def test_line(self):
        w = StepWriter()
        eid = w.line((0, 0, 0), (1, 0, 0))
        assert eid > 0

    def test_circle(self):
        w = StepWriter()
        eid = w.circle((0, 0, 0), (0, 0, 1), (1, 0, 0), 5.0)
        assert eid > 0

    def test_plane(self):
        w = StepWriter()
        eid = w.plane((0, 0, 0), (0, 0, 1))
        assert eid > 0

    def test_cylindrical_surface(self):
        w = StepWriter()
        eid = w.cylindrical_surface((0, 0, 0), (0, 0, 1), (1, 0, 0), 5.0)
        assert eid > 0


class TestBuildFlatSolid:
    """Test flat solid construction."""

    def test_simple_box(self):
        """A flat solid from a square should produce a valid closed shell."""
        w = StepWriter()
        outline = [
            (0.0, 0.0, 0.0),
            (10.0, 0.0, 0.0),
            (10.0, 10.0, 0.0),
            (0.0, 10.0, 0.0),
        ]
        normal = (0.0, 0.0, 1.0)
        brep_id = w.build_flat_solid(outline, None, normal, 1.6)
        assert brep_id > 0

        # Check entities contain expected types
        entity_text = " ".join(text for _, text in w._entities)
        assert "PLANE" in entity_text
        assert "CLOSED_SHELL" in entity_text
        assert "MANIFOLD_SOLID_BREP" in entity_text

    def test_box_face_count(self):
        """A box (4 sides + top + bottom) should have 6 ADVANCED_FACE entities."""
        w = StepWriter()
        outline = [
            (0.0, 0.0, 0.0),
            (10.0, 0.0, 0.0),
            (10.0, 5.0, 0.0),
            (0.0, 5.0, 0.0),
        ]
        w.build_flat_solid(outline, None, (0.0, 0.0, 1.0), 1.0)
        face_count = sum(1 for _, text in w._entities if text.startswith("ADVANCED_FACE"))
        assert face_count == 6  # top + bottom + 4 sides

    def test_box_with_hole(self):
        """A box with a hole should have extra side faces for the hole."""
        w = StepWriter()
        outline = [
            (0.0, 0.0, 0.0),
            (20.0, 0.0, 0.0),
            (20.0, 20.0, 0.0),
            (0.0, 20.0, 0.0),
        ]
        hole = [
            (5.0, 5.0, 0.0),
            (15.0, 5.0, 0.0),
            (15.0, 15.0, 0.0),
            (5.0, 15.0, 0.0),
        ]
        brep_id = w.build_flat_solid(outline, [hole], (0.0, 0.0, 1.0), 1.0)
        assert brep_id > 0
        face_count = sum(1 for _, text in w._entities if text.startswith("ADVANCED_FACE"))
        # 2 (top/bottom with hole) + 4 (outer sides) + 4 (hole sides) = 10
        assert face_count == 10


class TestBuildBendSolid:
    """Test cylindrical solid construction."""

    def test_basic_bend(self):
        """Build a bend solid and verify it has CYLINDRICAL_SURFACE entities."""
        w = StepWriter()
        R = 5.0
        t = 1.6
        angle = math.pi / 2  # 90 degree bend

        cyl_origin = (0.0, 0.0, 0.0)
        cyl_axis = (0.0, 1.0, 0.0)
        cyl_ref = (1.0, 0.0, 0.0)

        # Inner corners: arc from 0 to 90 degrees at radius R
        inner_corners = [
            (R, 0.0, 0.0),            # start bottom
            (R, 10.0, 0.0),           # start top
            (0.0, 10.0, R),           # end top (90 deg)
            (0.0, 0.0, R),            # end bottom (90 deg)
        ]

        outer_corners = [
            (R + t, 0.0, 0.0),
            (R + t, 10.0, 0.0),
            (0.0, 10.0, R + t),
            (0.0, 0.0, R + t),
        ]

        brep_id = w.build_bend_solid(
            inner_corners, outer_corners,
            cyl_origin, cyl_axis, cyl_ref,
            R, R + t,
            end_cap_pairs=None
        )
        assert brep_id > 0

        entity_text = " ".join(text for _, text in w._entities)
        assert "CYLINDRICAL_SURFACE" in entity_text
        assert "MANIFOLD_SOLID_BREP" in entity_text
        # Should have 2 cylindrical + 4 planar = 6 faces
        face_count = sum(1 for _, text in w._entities if text.startswith("ADVANCED_FACE"))
        assert face_count == 6


class TestStepFileWrite:
    """Test STEP file output."""

    def test_write_produces_valid_structure(self):
        """Written file should have proper STEP structure."""
        w = StepWriter()
        outline = [
            (0.0, 0.0, 0.0),
            (10.0, 0.0, 0.0),
            (10.0, 10.0, 0.0),
            (0.0, 10.0, 0.0),
        ]
        brep_id = w.build_flat_solid(outline, None, (0.0, 0.0, 1.0), 1.6)
        w.add_body(brep_id, "TEST_BODY")

        with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as f:
            path = f.name

        try:
            w.write(path)
            content = open(path).read()

            assert content.startswith("ISO-10303-21;")
            assert "HEADER;" in content
            assert "ENDSEC;" in content
            assert "DATA;" in content
            assert "END-ISO-10303-21;" in content
            assert "MANIFOLD_SOLID_BREP" in content
            assert "ADVANCED_BREP_SHAPE_REPRESENTATION" in content
            assert "PRODUCT('TEST_BODY'" in content
            assert "APPLICATION_CONTEXT" in content
            assert "SI_UNIT(.MILLI.,.METRE.)" in content
            assert "SI_UNIT($,.RADIAN.)" in content
        finally:
            os.unlink(path)

    def test_entity_references_resolve(self):
        """All #N references in the file should point to defined entities."""
        w = StepWriter()
        outline = [
            (0.0, 0.0, 0.0),
            (10.0, 0.0, 0.0),
            (10.0, 10.0, 0.0),
            (0.0, 10.0, 0.0),
        ]
        brep_id = w.build_flat_solid(outline, None, (0.0, 0.0, 1.0), 1.6)
        w.add_body(brep_id, "TEST")

        with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as f:
            path = f.name

        try:
            w.write(path)
            content = open(path).read()

            # Extract all defined entity IDs
            defined = set()
            for match in re.finditer(r'^#(\d+)=', content, re.MULTILINE):
                defined.add(int(match.group(1)))

            # Extract all referenced entity IDs (skip *= which means unset)
            referenced = set()
            for match in re.finditer(r'#(\d+)', content):
                # Exclude the definition itself (at start of line)
                referenced.add(int(match.group(1)))

            # All referenced IDs should be defined
            undefined = referenced - defined
            assert not undefined, f"Undefined entity references: {undefined}"
        finally:
            os.unlink(path)

    def test_multiple_bodies(self):
        """Multiple bodies should all appear in the output."""
        w = StepWriter()
        outline1 = [(0, 0, 0), (5, 0, 0), (5, 5, 0), (0, 5, 0)]
        outline2 = [(10, 0, 0), (15, 0, 0), (15, 5, 0), (10, 5, 0)]

        b1 = w.build_flat_solid(outline1, None, (0, 0, 1), 1.0)
        b2 = w.build_flat_solid(outline2, None, (0, 0, 1), 1.0)
        w.add_body(b1, "BODY_A")
        w.add_body(b2, "BODY_B")

        with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as f:
            path = f.name

        try:
            w.write(path)
            content = open(path).read()
            assert "BODY_A" in content
            assert "BODY_B" in content
            # Two MANIFOLD_SOLID_BREP
            assert content.count("MANIFOLD_SOLID_BREP") == 2
        finally:
            os.unlink(path)


class TestIsStepExportAvailable:
    """Test that step export is always available."""

    def test_always_available(self):
        assert is_step_export_available() is True


class TestBoardToStepIntegration:
    """Integration tests using test PCB files."""

    def test_flat_board(self, rectangle_pcb_path):
        """Export a flat board (no folds) to STEP."""
        from kicad_parser import KiCadPCB
        from geometry import extract_geometry

        if not rectangle_pcb_path.exists():
            pytest.skip("Test data file not found")
        pcb = KiCadPCB.load(rectangle_pcb_path)
        board_geo = extract_geometry(pcb)

        with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as f:
            path = f.name

        try:
            success = board_to_step_native(board_geo, [], path)
            assert success
            content = open(path).read()
            assert "ISO-10303-21;" in content
            assert "MANIFOLD_SOLID_BREP" in content
        finally:
            os.unlink(path)

    def test_board_with_folds(self, fold_pcb_path):
        """Export a board with fold markers to STEP."""
        from kicad_parser import KiCadPCB
        from geometry import extract_geometry
        from markers import detect_fold_markers

        if not fold_pcb_path.exists():
            pytest.skip("Test data file not found")
        pcb = KiCadPCB.load(fold_pcb_path)
        board_geo = extract_geometry(pcb)
        markers = detect_fold_markers(pcb, layer="User.1")

        with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as f:
            path = f.name

        try:
            success = board_to_step_native(board_geo, markers, path)
            assert success
            content = open(path).read()
            assert "ISO-10303-21;" in content
            assert "MANIFOLD_SOLID_BREP" in content
            # All PCB regions merge into 1 solid; verify many faces exist
            face_count = content.count("ADVANCED_FACE")
            assert face_count >= 10, f"Expected many faces from multiple regions, got {face_count}"
        finally:
            os.unlink(path)


class TestNormalize:
    """Test helper functions."""

    def test_normalize_unit(self):
        result = _normalize((1.0, 0.0, 0.0))
        assert abs(result[0] - 1.0) < 1e-10
        assert abs(result[1]) < 1e-10
        assert abs(result[2]) < 1e-10

    def test_normalize_diagonal(self):
        result = _normalize((1.0, 1.0, 0.0))
        expected = 1.0 / math.sqrt(2.0)
        assert abs(result[0] - expected) < 1e-10
        assert abs(result[1] - expected) < 1e-10

    def test_round_pt(self):
        result = _round_pt((1.0000001, 2.0000002, 3.0000003))
        assert result == (1.0, 2.0, 3.0)


# =====================================================================
# Arc recovery tests
# =====================================================================

def _make_arc_segment(start, end, center, radius, mid=None):
    """Helper to create an OutlineSegment arc."""
    if mid is None:
        # Compute mid point on arc
        cx, cy = center
        a_start = math.atan2(start[1] - cy, start[0] - cx)
        a_end = math.atan2(end[1] - cy, end[0] - cx)
        a_mid = (a_start + a_end) / 2
        mid = (cx + radius * math.cos(a_mid), cy + radius * math.sin(a_mid))
    return OutlineSegment(
        type="arc", start=start, end=end,
        center=center, radius=radius, mid=mid,
    )


class TestArcRecovery:
    """Test arc recovery from linearized outlines."""

    def test_edge_matches_arc_segment(self):
        """An edge whose endpoints lie on an arc should be tagged as arc."""
        R = 5.0
        cx, cy = 10.0, 10.0
        # Arc from 0 to 90 degrees
        start = (cx + R, cy)
        end = (cx, cy + R)
        mid = (cx + R * math.cos(math.pi/4), cy + R * math.sin(math.pi/4))
        arc_seg = _make_arc_segment(start, end, (cx, cy), R, mid)

        # Use two points on the arc as the edge
        a1 = math.pi / 6   # 30 degrees
        a2 = math.pi / 3   # 60 degrees
        p1 = (cx + R * math.cos(a1), cy + R * math.sin(a1))
        p2 = (cx + R * math.cos(a2), cy + R * math.sin(a2))

        region_outline = [p1, p2, (cx, cy)]  # triangle with 2 points on arc
        tagged = _recover_arcs(region_outline, [arc_seg])

        # First edge (p1->p2) should be arc
        assert tagged[0].type == "arc"
        assert abs(tagged[0].radius - R) < 0.001
        assert tagged[0].center == (cx, cy)

    def test_edge_no_match_for_line(self):
        """A straight edge should not match any arc."""
        arc_seg = _make_arc_segment((5, 0), (0, 5), (0, 0), 5.0)
        region_outline = [(0, 0), (10, 0), (10, 10), (0, 10)]
        tagged = _recover_arcs(region_outline, [arc_seg])
        assert all(e.type == "line" for e in tagged)

    def test_partial_arc_after_cut(self):
        """A sub-arc from fold cutting should still match the parent arc."""
        R = 10.0
        cx, cy = 0.0, 0.0
        # Full arc from 0 to 180 degrees
        start = (R, 0)
        end = (-R, 0)
        mid = (0, R)
        arc_seg = _make_arc_segment(start, end, (cx, cy), R, mid)

        # Sub-arc: 45 to 90 degrees (a cut portion)
        a1 = math.pi / 4
        a2 = math.pi / 2
        p1 = (R * math.cos(a1), R * math.sin(a1))
        p2 = (R * math.cos(a2), R * math.sin(a2))

        region_outline = [p1, p2, (0, 0)]
        tagged = _recover_arcs(region_outline, [arc_seg])
        assert tagged[0].type == "arc"

    def test_no_false_positive_different_center(self):
        """Same radius but different center should not match."""
        R = 5.0
        # Arc centered at (0, 0)
        arc_seg = _make_arc_segment((R, 0), (0, R), (0, 0), R)

        # Points on a circle of same radius but centered at (20, 20)
        a1 = math.pi / 6
        a2 = math.pi / 3
        p1 = (20 + R * math.cos(a1), 20 + R * math.sin(a1))
        p2 = (20 + R * math.cos(a2), 20 + R * math.sin(a2))

        region_outline = [p1, p2, (20, 20)]
        tagged = _recover_arcs(region_outline, [arc_seg])
        assert tagged[0].type == "line"


class TestCircleHoleRecovery:
    """Test circle hole detection from polygonized cutouts."""

    def _make_circle_polygon(self, cx, cy, r, n=24):
        """Create a polygon approximation of a circle."""
        return [(cx + r * math.cos(2 * math.pi * i / n),
                 cy + r * math.sin(2 * math.pi * i / n)) for i in range(n)]

    def test_24gon_matches_circle_cutout(self):
        """A polygonized circle should be matched to its CircleCutout source."""
        from kicad_parser import CircleCutout
        cx, cy, r = 10.0, 15.0, 2.5
        cutout = CircleCutout(center_x=cx, center_y=cy, radius=r)
        hole_verts = self._make_circle_polygon(cx, cy, r)

        result = _recover_circle_holes([hole_verts], [cutout], [])
        assert len(result) == 1
        assert result[0][0] == 'circle'
        assert abs(result[0][1][0] - cx) < 0.01
        assert abs(result[0][1][1] - cy) < 0.01
        assert abs(result[0][2] - r) < 0.01

    def test_polygon_hole_not_matched(self):
        """A rectangular hole should stay as polygon."""
        from kicad_parser import CircleCutout
        cutout = CircleCutout(center_x=0, center_y=0, radius=5)
        rect_hole = [(0, 0), (10, 0), (10, 5), (0, 5)]

        result = _recover_circle_holes([rect_hole], [cutout], [])
        assert len(result) == 1
        assert result[0][0] == 'polygon'

    def test_drill_hole_matched(self):
        """A DrillHole should be matched by center+radius."""
        from kicad_parser import DrillHole
        cx, cy, d = 5.0, 5.0, 1.0
        drill = DrillHole(center_x=cx, center_y=cy, diameter=d)
        hole_verts = self._make_circle_polygon(cx, cy, d / 2)

        result = _recover_circle_holes([hole_verts], [], [drill])
        assert len(result) == 1
        assert result[0][0] == 'circle'
        assert abs(result[0][2] - d / 2) < 0.01


class TestMixedEdgeLoop:
    """Test mixed line+arc edge loop construction."""

    def test_all_line_loop(self):
        """An all-line mixed loop should produce the same entity types as _make_line_loop."""
        w = StepWriter()
        edges = [
            {'type': 'line', 'start': (0, 0, 0), 'end': (10, 0, 0)},
            {'type': 'line', 'start': (10, 0, 0), 'end': (10, 10, 0)},
            {'type': 'line', 'start': (10, 10, 0), 'end': (0, 10, 0)},
            {'type': 'line', 'start': (0, 10, 0), 'end': (0, 0, 0)},
        ]
        loop_id = w._make_mixed_loop(edges)
        assert loop_id > 0
        # Should have LINE entities, no CIRCLE
        entity_text = " ".join(text for _, text in w._entities)
        assert "LINE" in entity_text
        assert "CIRCLE" not in entity_text

    def test_arc_edge_produces_circle_entity(self):
        """An arc edge in a mixed loop should produce a CIRCLE entity."""
        w = StepWriter()
        R = 5.0
        edges = [
            {'type': 'arc', 'start': (R, 0, 0), 'end': (0, R, 0),
             'center': (0, 0, 0), 'axis': (0, 0, 1), 'ref_dir': (1, 0, 0), 'radius': R},
            {'type': 'line', 'start': (0, R, 0), 'end': (0, 0, 0)},
            {'type': 'line', 'start': (0, 0, 0), 'end': (R, 0, 0)},
        ]
        loop_id = w._make_mixed_loop(edges)
        assert loop_id > 0
        entity_text = " ".join(text for _, text in w._entities)
        assert "CIRCLE" in entity_text
        assert "LINE" in entity_text

    def test_mixed_line_arc(self):
        """A loop with both LINE and CIRCLE edge curves."""
        w = StepWriter()
        R = 10.0
        a45 = R * math.cos(math.pi / 4)
        edges = [
            {'type': 'line', 'start': (0, 0, 0), 'end': (R, 0, 0)},
            {'type': 'arc', 'start': (R, 0, 0), 'end': (0, R, 0),
             'center': (0, 0, 0), 'axis': (0, 0, 1), 'ref_dir': (1, 0, 0), 'radius': R},
            {'type': 'line', 'start': (0, R, 0), 'end': (0, 0, 0)},
        ]
        loop_id = w._make_mixed_loop(edges)
        assert loop_id > 0
        # Count LINE and CIRCLE entities
        line_count = sum(1 for _, text in w._entities if text.startswith("LINE("))
        circle_count = sum(1 for _, text in w._entities if text.startswith("CIRCLE("))
        assert line_count >= 2
        assert circle_count >= 1


class TestCylindricalSideFaces:
    """Test side face generation for mixed edges."""

    def test_arc_edge_cylindrical_side(self):
        """An arc edge should produce a CYLINDRICAL_SURFACE side face."""
        w = StepWriter()
        R = 5.0
        top_edges = [
            {'type': 'arc', 'start': (R, 0, 0), 'end': (0, R, 0),
             'center': (0, 0, 0), 'axis': (0, 0, 1), 'ref_dir': (1, 0, 0), 'radius': R},
        ]
        bot_edges = [
            {'type': 'arc', 'start': (R, 0, -1.6), 'end': (0, R, -1.6),
             'center': (0, 0, -1.6), 'axis': (0, 0, 1), 'ref_dir': (1, 0, 0), 'radius': R},
        ]
        faces = w._build_side_faces_mixed(top_edges, bot_edges, (0, 0, 1))
        assert len(faces) == 1
        entity_text = " ".join(text for _, text in w._entities)
        assert "CYLINDRICAL_SURFACE" in entity_text

    def test_line_edge_planar_side(self):
        """A line edge should produce a PLANE side face."""
        w = StepWriter()
        top_edges = [
            {'type': 'line', 'start': (0, 0, 0), 'end': (10, 0, 0)},
        ]
        bot_edges = [
            {'type': 'line', 'start': (0, 0, -1.6), 'end': (10, 0, -1.6)},
        ]
        faces = w._build_side_faces_mixed(top_edges, bot_edges, (0, 0, 1))
        assert len(faces) == 1
        entity_text = " ".join(text for _, text in w._entities)
        assert "PLANE" in entity_text
        # Should NOT have CYLINDRICAL_SURFACE
        assert "CYLINDRICAL_SURFACE" not in entity_text


class TestCircleHole:
    """Test circle hole B-Rep construction."""

    def test_circle_hole_produces_cylindrical_wall(self):
        """A circle hole should produce CYLINDRICAL_SURFACE wall faces."""
        w = StepWriter()
        center_top = (10, 10, 0)
        center_bot = (10, 10, -1.6)
        radius = 2.5
        normal = (0, 0, 1)

        top_loop, bot_loop, side_faces = w._build_circle_hole_faces(
            center_top, center_bot, radius, normal
        )
        assert top_loop > 0
        assert bot_loop > 0
        assert len(side_faces) == 2  # Two semicircular patches

        entity_text = " ".join(text for _, text in w._entities)
        assert "CYLINDRICAL_SURFACE" in entity_text
        assert "CIRCLE" in entity_text

    def test_circle_hole_entity_references_resolve(self):
        """All entity references from circle hole construction should be valid."""
        w = StepWriter()
        center_top = (5, 5, 0)
        center_bot = (5, 5, -1.0)
        radius = 1.5

        w._build_circle_hole_faces(center_top, center_bot, radius, (0, 0, 1))

        # Check all #N references in entities resolve
        defined = set()
        referenced = set()
        for eid, text in w._entities:
            defined.add(eid)
            for m in re.finditer(r'#(\d+)', text):
                referenced.add(int(m.group(1)))

        undefined = referenced - defined
        assert not undefined, f"Undefined entity references: {undefined}"


class TestFlatSolidMixed:
    """Test build_flat_solid_mixed high-level method."""

    def test_box_with_one_arc_side(self):
        """An outline with 3 lines + 1 arc should produce both PLANE and CYLINDRICAL_SURFACE."""
        w = StepWriter()
        R = 5.0
        edges = [
            {'type': 'line', 'start': (0, 0, 0), 'end': (10, 0, 0)},
            {'type': 'arc', 'start': (10, 0, 0), 'end': (10, 10, 0),
             'center': (10, 5, 0), 'axis': (0, 0, 1), 'ref_dir': (0, -1, 0), 'radius': R},
            {'type': 'line', 'start': (10, 10, 0), 'end': (0, 10, 0)},
            {'type': 'line', 'start': (0, 10, 0), 'end': (0, 0, 0)},
        ]
        brep_id = w.build_flat_solid_mixed(edges, (0, 0, 1), 1.6)
        assert brep_id > 0

        entity_text = " ".join(text for _, text in w._entities)
        assert "PLANE" in entity_text
        assert "CYLINDRICAL_SURFACE" in entity_text
        assert "MANIFOLD_SOLID_BREP" in entity_text

    def test_box_with_circle_hole(self):
        """A flat solid with a circle hole should have cylindrical wall."""
        w = StepWriter()
        edges = [
            {'type': 'line', 'start': (0, 0, 0), 'end': (20, 0, 0)},
            {'type': 'line', 'start': (20, 0, 0), 'end': (20, 20, 0)},
            {'type': 'line', 'start': (20, 20, 0), 'end': (0, 20, 0)},
            {'type': 'line', 'start': (0, 20, 0), 'end': (0, 0, 0)},
        ]
        hole_data = [
            ('circle', (10, 10, 0), (10, 10, -1.6), 3.0),
        ]
        brep_id = w.build_flat_solid_mixed(edges, (0, 0, 1), 1.6, hole_data=hole_data)
        assert brep_id > 0

        entity_text = " ".join(text for _, text in w._entities)
        assert "CYLINDRICAL_SURFACE" in entity_text

    def test_entity_references_all_resolve(self):
        """All #N references should resolve for a mixed solid."""
        w = StepWriter()
        R = 5.0
        edges = [
            {'type': 'line', 'start': (0, 0, 0), 'end': (10, 0, 0)},
            {'type': 'arc', 'start': (10, 0, 0), 'end': (10, 10, 0),
             'center': (10, 5, 0), 'axis': (0, 0, 1), 'ref_dir': (0, -1, 0), 'radius': R},
            {'type': 'line', 'start': (10, 10, 0), 'end': (0, 10, 0)},
            {'type': 'line', 'start': (0, 10, 0), 'end': (0, 0, 0)},
        ]
        brep_id = w.build_flat_solid_mixed(edges, (0, 0, 1), 1.6)
        w.add_body(brep_id, "MIXED_TEST")

        with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as f:
            path = f.name

        try:
            w.write(path)
            content = open(path).read()

            defined = set()
            for match in re.finditer(r'^#(\d+)=', content, re.MULTILINE):
                defined.add(int(match.group(1)))

            referenced = set()
            for match in re.finditer(r'#(\d+)', content):
                referenced.add(int(match.group(1)))

            undefined = referenced - defined
            assert not undefined, f"Undefined entity references: {undefined}"
        finally:
            os.unlink(path)


class TestStepWriterEdgeCases:
    """Additional edge case tests for StepWriter."""

    def test_point_deduplication(self):
        """Same coordinates return the same entity ID."""
        w = StepWriter()
        id1 = w.cartesian_point((5.0, 10.0, 15.0))
        id2 = w.cartesian_point((5.0, 10.0, 15.0))
        assert id1 == id2

    def test_direction_normalization(self):
        """(0,0,2) and (0,0,1) normalize to the same direction entity."""
        w = StepWriter()
        id1 = w.direction((0.0, 0.0, 1.0))
        id2 = w.direction((0.0, 0.0, 2.0))
        assert id1 == id2

    def test_opposite_directions_different(self):
        """(0,0,1) and (0,0,-1) are different direction entities."""
        w = StepWriter()
        id1 = w.direction((0.0, 0.0, 1.0))
        id2 = w.direction((0.0, 0.0, -1.0))
        assert id1 != id2

    def test_point_near_zero_rounding(self):
        """Very small coords like (1e-12, 0, 0) round to (0,0,0) for dedup."""
        w = StepWriter()
        id1 = w.cartesian_point((0.0, 0.0, 0.0))
        id2 = w.cartesian_point((1e-12, 0.0, 0.0))
        # Both should round to (0,0,0) at 6 decimal places
        assert id1 == id2

    def test_entity_ids_monotonic(self):
        """Each new unique entity gets a strictly incrementing ID."""
        w = StepWriter()
        id1 = w.cartesian_point((0.0, 0.0, 0.0))
        id2 = w.cartesian_point((1.0, 0.0, 0.0))
        id3 = w.cartesian_point((2.0, 0.0, 0.0))
        assert id1 < id2 < id3

    def test_write_creates_file(self):
        """Write a simple solid; verify file exists and starts with ISO-10303-21."""
        w = StepWriter()
        outline = [
            (0.0, 0.0, 0.0),
            (10.0, 0.0, 0.0),
            (10.0, 10.0, 0.0),
            (0.0, 10.0, 0.0),
        ]
        brep_id = w.build_flat_solid(outline, None, (0.0, 0.0, 1.0), 1.6)
        w.add_body(brep_id, "TEST_SOLID")

        with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as f:
            path = f.name

        try:
            w.write(path)
            assert os.path.exists(path)
            content = open(path).read()
            assert content.startswith("ISO-10303-21;")
        finally:
            os.unlink(path)

    def test_empty_writer_writes_header(self):
        """StepWriter with no entities still writes valid STEP header/footer."""
        w = StepWriter()
        with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as f:
            path = f.name

        try:
            w.write(path)
            content = open(path).read()
            assert content.startswith("ISO-10303-21;")
            assert "HEADER;" in content
            assert "ENDSEC;" in content
            assert "DATA;" in content
            assert "END-ISO-10303-21;" in content
        finally:
            os.unlink(path)

    def test_manifold_solid_brep_box(self):
        """Create a simple box using the API; verify reasonable entity count."""
        w = StepWriter()
        outline = [
            (0.0, 0.0, 0.0),
            (10.0, 0.0, 0.0),
            (10.0, 10.0, 0.0),
            (0.0, 10.0, 0.0),
        ]
        brep_id = w.build_flat_solid(outline, None, (0.0, 0.0, 1.0), 1.6)
        assert brep_id > 0

        entity_text = " ".join(text for _, text in w._entities)
        assert "MANIFOLD_SOLID_BREP" in entity_text
        # A box should have many entities (points, directions, edges, faces, etc.)
        assert len(w._entities) > 20


# =====================================================================
# B-spline curve tests (Phase 7)
# =====================================================================


class TestBSplineCurveEntity:
    """Test B_SPLINE_CURVE_WITH_KNOTS entity creation."""

    def test_b_spline_curve_entity_format(self):
        """Verify the STEP entity text is correctly formatted."""
        w = StepWriter()
        cp1 = w.cartesian_point((0.0, 0.0, 0.0))
        cp2 = w.cartesian_point((5.0, 1.0, 0.0))
        cp3 = w.cartesian_point((10.0, 0.0, 0.0))

        bsp_id = w.b_spline_curve(
            degree=1,
            control_point_ids=[cp1, cp2, cp3],
            knot_multiplicities=[2, 1, 2],
            knots=[0.0, 0.5, 1.0],
            form=".POLYLINE_FORM."
        )
        assert bsp_id > 0

        # Find the entity text
        bsp_text = None
        for eid, text in w._entities:
            if eid == bsp_id:
                bsp_text = text
                break
        assert bsp_text is not None
        assert "B_SPLINE_CURVE_WITH_KNOTS" in bsp_text
        assert f"#{cp1}" in bsp_text
        assert f"#{cp2}" in bsp_text
        assert f"#{cp3}" in bsp_text
        assert ".POLYLINE_FORM." in bsp_text

    def test_b_spline_degree_1_polyline(self):
        """Degree 1 B-spline with N points: verify knot structure."""
        w = StepWriter()
        # 5 control points for a degree-1 polyline
        pts = [(i * 2.0, math.sin(i * 0.5), 0.0) for i in range(5)]
        cp_ids = [w.cartesian_point(p) for p in pts]

        knots = [0.0, 0.25, 0.5, 0.75, 1.0]
        mults = [2, 1, 1, 1, 2]  # clamped ends

        bsp_id = w.b_spline_curve(1, cp_ids, mults, knots)
        assert bsp_id > 0

        bsp_text = None
        for eid, text in w._entities:
            if eid == bsp_id:
                bsp_text = text
                break

        # Degree should be 1
        assert bsp_text.startswith("B_SPLINE_CURVE_WITH_KNOTS('',1,")
        # All 5 control point refs should be present
        for cp_id in cp_ids:
            assert f"#{cp_id}" in bsp_text

    def test_b_spline_edge_creation(self):
        """B-spline edge via _get_or_create_bspline_edge."""
        w = StepWriter()
        pts = [(0.0, 0.0, 0.0), (3.0, 1.0, 0.0), (6.0, 0.5, 0.0),
               (9.0, 1.5, 0.0), (12.0, 0.0, 0.0)]

        ec_id, same_dir = w._get_or_create_bspline_edge(pts[0], pts[-1], pts)
        assert ec_id > 0
        assert same_dir is True

        entity_text = " ".join(text for _, text in w._entities)
        assert "B_SPLINE_CURVE_WITH_KNOTS" in entity_text
        assert "EDGE_CURVE" in entity_text

    def test_b_spline_edge_dedup(self):
        """Same start/end points should return cached edge."""
        w = StepWriter()
        pts = [(0.0, 0.0, 0.0), (5.0, 1.0, 0.0), (10.0, 0.0, 0.0)]

        ec_id1, _ = w._get_or_create_bspline_edge(pts[0], pts[-1], pts)
        ec_id2, _ = w._get_or_create_bspline_edge(pts[0], pts[-1], pts)
        assert ec_id1 == ec_id2

    def test_b_spline_in_mixed_loop(self):
        """B-spline edges should be usable in mixed loops."""
        w = StepWriter()
        pts = [(0.0, 0.0, 0.0), (3.0, 2.0, 0.0), (6.0, 1.0, 0.0),
               (10.0, 0.0, 0.0)]
        edges = [
            {'type': 'bspline', 'start': pts[0], 'end': pts[-1],
             'sample_points': pts},
            {'type': 'line', 'start': pts[-1], 'end': (10.0, -5.0, 0.0)},
            {'type': 'line', 'start': (10.0, -5.0, 0.0), 'end': pts[0]},
        ]
        loop_id = w._make_mixed_loop(edges)
        assert loop_id > 0

        entity_text = " ".join(text for _, text in w._entities)
        assert "B_SPLINE_CURVE_WITH_KNOTS" in entity_text
        assert "LINE" in entity_text

    def test_b_spline_write_to_file(self):
        """B-spline curve entities should appear correctly in written STEP file."""
        w = StepWriter()
        pts_3d = [(i * 2.0, math.sin(i * 0.5), 0.0) for i in range(8)]
        edges = [
            {'type': 'bspline', 'start': pts_3d[0], 'end': pts_3d[-1],
             'sample_points': pts_3d},
            {'type': 'line', 'start': pts_3d[-1], 'end': (14.0, -5.0, 0.0)},
            {'type': 'line', 'start': (14.0, -5.0, 0.0), 'end': pts_3d[0]},
        ]
        # Build a flat solid with mixed edges including B-spline
        brep_id = w.build_flat_solid_mixed(edges, (0, 0, 1), 1.6)
        w.add_body(brep_id, "BSPLINE_TEST")

        with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as f:
            path = f.name

        try:
            w.write(path)
            content = open(path).read()
            assert "B_SPLINE_CURVE_WITH_KNOTS" in content
            assert "MANIFOLD_SOLID_BREP" in content

            # Verify all references resolve
            defined = set()
            for match in re.finditer(r'^#(\d+)=', content, re.MULTILINE):
                defined.add(int(match.group(1)))
            referenced = set()
            for match in re.finditer(r'#(\d+)', content):
                referenced.add(int(match.group(1)))
            undefined = referenced - defined
            assert not undefined, f"Undefined entity references: {undefined}"
        finally:
            os.unlink(path)


class TestSampleArc2d:
    """Test 2D arc sampling."""

    def test_quarter_circle_ccw(self):
        """Sample a quarter circle CCW from (R,0) to (0,R)."""
        R = 10.0
        pts = _sample_arc_2d((R, 0), (0, R), (0, 0), R, ccw=True, n_samples=5)
        assert len(pts) == 5
        # First point should be (R, 0)
        assert abs(pts[0][0] - R) < 1e-6
        assert abs(pts[0][1]) < 1e-6
        # Last point should be (0, R)
        assert abs(pts[-1][0]) < 1e-6
        assert abs(pts[-1][1] - R) < 1e-6
        # All points should be on the circle
        for px, py in pts:
            dist = math.sqrt(px**2 + py**2)
            assert abs(dist - R) < 1e-6

    def test_quarter_circle_cw(self):
        """Sample a quarter circle CW from (R,0) to (0,R)."""
        R = 5.0
        pts = _sample_arc_2d((R, 0), (0, R), (0, 0), R, ccw=False, n_samples=5)
        assert len(pts) == 5
        # CW from (R,0) to (0,R) goes the long way around (270 degrees)
        # Check first and last
        assert abs(pts[0][0] - R) < 1e-6
        assert abs(pts[-1][1] - R) < 1e-6

    def test_many_samples_uniform(self):
        """32 samples should produce points with roughly equal spacing."""
        R = 10.0
        pts = _sample_arc_2d((R, 0), (0, R), (0, 0), R, ccw=True, n_samples=32)
        assert len(pts) == 32
        # All should be on the circle
        for px, py in pts:
            dist = math.sqrt(px**2 + py**2)
            assert abs(dist - R) < 1e-6


class TestTransformTaggedEdges3dBend:
    """Test bend zone arc-to-bspline transformation."""

    def test_line_edge_stays_line(self):
        """A line edge in a bend zone should still produce a 'line' type."""
        from bend_transform import FoldDefinition
        edge = TaggedEdge(type="line", start=(0, 0), end=(10, 0))
        # Empty recipe = identity transform
        result = _transform_tagged_edges_3d_bend([edge], [])
        assert len(result) == 1
        assert result[0]['type'] == 'line'

    def test_arc_edge_becomes_bspline(self):
        """An arc edge in a bend zone should become a 'bspline' type."""
        edge = TaggedEdge(
            type="arc", start=(5, 0), end=(0, 5),
            center=(0, 0), radius=5.0, ccw=True
        )
        # Empty recipe = identity transform (arc still becomes bspline
        # because it's in a bend zone context)
        result = _transform_tagged_edges_3d_bend([edge], [], n_arc_samples=8)
        assert len(result) == 1
        assert result[0]['type'] == 'bspline'
        assert 'sample_points' in result[0]
        assert len(result[0]['sample_points']) == 8

    def test_bspline_sample_points_on_arc(self):
        """B-spline sample points should lie on the original arc (no fold)."""
        R = 10.0
        edge = TaggedEdge(
            type="arc", start=(R, 0), end=(0, R),
            center=(0, 0), radius=R, ccw=True
        )
        result = _transform_tagged_edges_3d_bend([edge], [], n_arc_samples=16)
        pts = result[0]['sample_points']
        for p in pts:
            # With identity transform, z=0 and distance from origin = R
            dist = math.sqrt(p[0]**2 + p[1]**2)
            assert abs(dist - R) < 0.01
            assert abs(p[2]) < 1e-6


class TestArcInBendZoneIntegration:
    """Integration test: arc crossing fold produces B-spline in STEP output."""

    def test_arc_in_bend_zone_produces_bspline(self):
        """When a board outline has an arc that crosses a fold zone,
        the STEP export should contain B_SPLINE_CURVE_WITH_KNOTS entities."""
        from geometry import Polygon, OutlineSegment, BoardGeometry
        from markers import FoldMarker

        # Create a rounded rectangle outline that crosses a fold zone.
        # Board: 0..30 in X, 0..20 in Y, with fillet radius 3mm at corners.
        # Fold at x=25, so the right-side fillet arcs (at x~27) cross the fold zone.
        R = 3.0
        # Build outline vertices for a rounded rectangle
        # We'll create a simplified version with explicit arc segments
        # Bottom edge: (R, 0) to (30-R, 0) line
        # Bottom-right corner: arc center (30-R, R), from 270 to 360 deg
        # Right edge: (30, R) to (30, 20-R) line
        # Top-right corner: arc center (30-R, 20-R), from 0 to 90 deg
        # Top edge: (30-R, 20) to (R, 20) line
        # Top-left corner: arc center (R, 20-R), from 90 to 180 deg
        # Left edge: (0, 20-R) to (0, R) line
        # Bottom-left corner: arc center (R, R), from 180 to 270 deg

        def arc_pts(cx, cy, r, sa_deg, ea_deg, n=16):
            """Generate arc vertices."""
            pts = []
            for i in range(n + 1):
                t = i / n
                a = math.radians(sa_deg + t * (ea_deg - sa_deg))
                pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
            return pts

        # Build full outline
        verts = []
        verts.extend([(R, 0)])
        verts.extend([(30 - R, 0)])
        verts.extend(arc_pts(30 - R, R, R, 270, 360, 8)[1:])
        verts.extend([(30, 20 - R)])
        verts.extend(arc_pts(30 - R, 20 - R, R, 0, 90, 8)[1:])
        verts.extend([(R, 20)])
        verts.extend(arc_pts(R, 20 - R, R, 90, 180, 8)[1:])
        verts.extend([(0, R)])
        verts.extend(arc_pts(R, R, R, 180, 270, 8)[1:])

        # Build outline segments (for arc recovery)
        def make_seg(stype, s, e, cx=None, cy=None, r=0, mid=None):
            return OutlineSegment(type=stype, start=s, end=e,
                                 center=(cx, cy) if cx is not None else None,
                                 radius=r, mid=mid)

        segments = [
            make_seg("line", (R, 0), (30 - R, 0)),
            make_seg("arc", (30 - R, 0), (30, R), cx=30 - R, cy=R, r=R,
                     mid=(30 - R + R * math.cos(math.radians(315)),
                          R + R * math.sin(math.radians(315)))),
            make_seg("line", (30, R), (30, 20 - R)),
            make_seg("arc", (30, 20 - R), (30 - R, 20), cx=30 - R, cy=20 - R, r=R,
                     mid=(30 - R + R * math.cos(math.radians(45)),
                          20 - R + R * math.sin(math.radians(45)))),
            make_seg("line", (30 - R, 20), (R, 20)),
            make_seg("arc", (R, 20), (0, 20 - R), cx=R, cy=20 - R, r=R,
                     mid=(R + R * math.cos(math.radians(135)),
                          20 - R + R * math.sin(math.radians(135)))),
            make_seg("line", (0, 20 - R), (0, R)),
            make_seg("arc", (0, R), (R, 0), cx=R, cy=R, r=R,
                     mid=(R + R * math.cos(math.radians(225)),
                          R + R * math.sin(math.radians(225)))),
        ]

        outline = Polygon(vertices=verts, segments=segments)

        # Fold at x=28, so zone [27..29] crosses the right fillets (centers at x=27)
        angle_deg = 90.0
        zone_width = 2.0
        angle_rad = math.radians(angle_deg)
        radius = zone_width / angle_rad
        fold_center = (28.0, 10.0)
        fold_axis = (0.0, 1.0)
        perp = (1.0, 0.0)
        hw = zone_width / 2

        marker = FoldMarker(
            line_a_start=(fold_center[0] - hw, 0),
            line_a_end=(fold_center[0] - hw, 20),
            line_b_start=(fold_center[0] + hw, 0),
            line_b_end=(fold_center[0] + hw, 20),
            angle_degrees=angle_deg,
            zone_width=zone_width,
            radius=radius,
            axis=fold_axis,
            center=fold_center,
        )

        board_geo = BoardGeometry(
            outline=outline,
            thickness=1.6,
        )

        with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as f:
            path = f.name

        try:
            success = board_to_step_native(board_geo, [marker], path)
            assert success
            content = open(path).read()
            assert "ISO-10303-21;" in content
            assert "MANIFOLD_SOLID_BREP" in content
            # The right-side fillet arcs cross the fold zone, so B-spline
            # curves should be present in the output
            assert "B_SPLINE_CURVE_WITH_KNOTS" in content, \
                "Expected B-spline curves for arcs crossing bend zone"
        finally:
            os.unlink(path)
