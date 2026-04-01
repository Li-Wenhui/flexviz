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
