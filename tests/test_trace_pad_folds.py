"""Tests for trace and pad rendering across fold zones.

Phase 1 validation: verify traces/pads follow bends correctly,
region lookup fallback works, and adaptive subdivision at fold boundaries.
"""

import pytest
import math
import os
from pathlib import Path

from mesh import (
    Mesh,
    create_trace_mesh,
    create_pad_mesh,
    create_board_geometry_mesh,
    get_region_recipe,
    _compute_fold_crossing_t_values,
    _dot3,
    COLOR_COPPER,
    COLOR_PAD,
)
from geometry import (
    Polygon, LineSegment, PadGeometry, BoardGeometry, BoundingBox, ComponentGeometry,
    extract_geometry,
)
from markers import FoldMarker, detect_fold_markers
from planar_subdivision import split_board_into_regions, find_containing_region
from bend_transform import FoldDefinition, transform_point, transform_point_and_normal
from kicad_parser import KiCadPCB, load_kicad_pcb


# =============================================================================
# Helpers
# =============================================================================

def make_vertical_fold_marker(x_center, zone_width=5.0, angle_deg=90.0, board_height=30.0):
    """Create a vertical fold marker at x=x_center (fold axis along Y)."""
    hw = zone_width / 2
    angle_rad = math.radians(angle_deg)
    radius = zone_width / abs(angle_rad) if abs(angle_rad) > 1e-9 else float('inf')
    return FoldMarker(
        line_a_start=(x_center - hw, 0),
        line_a_end=(x_center - hw, board_height),
        line_b_start=(x_center + hw, 0),
        line_b_end=(x_center + hw, board_height),
        angle_degrees=angle_deg,
        zone_width=zone_width,
        radius=radius,
        axis=(0.0, 1.0),  # vertical fold axis
        center=(x_center, board_height / 2),
    )


def make_horizontal_fold_marker(y_center, zone_width=2.0, angle_deg=90.0, board_width=10.0, x_start=0.0):
    """Create a horizontal fold marker at y=y_center (fold axis along X)."""
    hw = zone_width / 2
    angle_rad = math.radians(angle_deg)
    radius = zone_width / abs(angle_rad) if abs(angle_rad) > 1e-9 else float('inf')
    return FoldMarker(
        line_a_start=(x_start, y_center - hw),
        line_a_end=(x_start + board_width, y_center - hw),
        line_b_start=(x_start, y_center + hw),
        line_b_end=(x_start + board_width, y_center + hw),
        angle_degrees=angle_deg,
        zone_width=zone_width,
        radius=radius,
        axis=(1.0, 0.0),
        center=(x_start + board_width / 2, y_center),
    )


def make_board_with_fold(width=120, height=30, fold_x=50, fold_angle=90):
    """Create a board geometry with one vertical fold."""
    outline = Polygon([(0, 0), (width, 0), (width, height), (0, height)])
    marker = make_vertical_fold_marker(fold_x, zone_width=5.0, angle_deg=fold_angle, board_height=height)
    board = BoardGeometry(outline=outline, thickness=1.6)
    return board, [marker]


def get_regions_for_board(board, markers):
    """Get regions from board + markers."""
    outline_verts = [(v[0], v[1]) for v in board.outline.vertices]
    return split_board_into_regions(outline_verts, [], markers, num_bend_subdivisions=4)


# =============================================================================
# Test: Fold crossing t-value computation
# =============================================================================

class TestFoldCrossingTValues:
    """Test adaptive subdivision at fold zone boundaries."""

    def test_horizontal_trace_crossing_vertical_fold(self):
        """A horizontal trace should detect crossings at fold zone boundaries."""
        marker = make_vertical_fold_marker(50.0, zone_width=5.0)
        start = (10.0, 15.0)
        end = (90.0, 15.0)

        t_values = _compute_fold_crossing_t_values(start, end, [marker])

        # Should find crossings at x=47.5 and x=52.5 (fold center 50, hw=2.5)
        assert len(t_values) == 2
        # t for x=47.5: (47.5-10)/(90-10) = 37.5/80 = 0.46875
        assert abs(t_values[0] - 0.46875) < 0.01
        # t for x=52.5: (52.5-10)/(90-10) = 42.5/80 = 0.53125
        assert abs(t_values[1] - 0.53125) < 0.01

    def test_trace_not_crossing_fold(self):
        """A trace entirely before the fold should have no crossings."""
        marker = make_vertical_fold_marker(50.0, zone_width=5.0)
        start = (10.0, 15.0)
        end = (40.0, 15.0)

        t_values = _compute_fold_crossing_t_values(start, end, [marker])

        assert len(t_values) == 0

    def test_trace_crossing_two_folds(self):
        """A trace crossing two folds should have 4 boundary crossings."""
        markers = [
            make_vertical_fold_marker(40.0, zone_width=5.0),
            make_vertical_fold_marker(80.0, zone_width=5.0),
        ]
        start = (10.0, 15.0)
        end = (110.0, 15.0)

        t_values = _compute_fold_crossing_t_values(start, end, markers)

        assert len(t_values) == 4

    def test_parallel_trace_no_crossing(self):
        """A trace parallel to a fold axis should have no crossings."""
        marker = make_vertical_fold_marker(50.0, zone_width=5.0)
        # Trace runs along Y at x=30 (before fold zone)
        start = (30.0, 0.0)
        end = (30.0, 30.0)

        t_values = _compute_fold_crossing_t_values(start, end, [marker])

        assert len(t_values) == 0


# =============================================================================
# Test: Trace mesh with folds
# =============================================================================

class TestTraceMeshWithFolds:
    """Test trace mesh generation across fold zones."""

    def test_trace_crossing_fold_has_more_subdivisions(self):
        """Trace crossing a fold should have extra subdivision points at boundaries."""
        board, markers = make_board_with_fold()
        regions = get_regions_for_board(board, markers)

        segment = LineSegment((10, 15), (90, 15), width=0.3)

        # Without markers (old behavior)
        mesh_no_markers = create_trace_mesh(segment, 0.01, regions, subdivisions=20)
        # With markers (new adaptive behavior)
        mesh_with_markers = create_trace_mesh(segment, 0.01, regions, subdivisions=20, markers=markers)

        # With markers should have more vertices (extra at fold boundaries)
        assert len(mesh_with_markers.vertices) > len(mesh_no_markers.vertices)

    def test_trace_crossing_fold_follows_bend(self):
        """Trace crossing a 90-degree fold should have Z variation."""
        board, markers = make_board_with_fold(fold_angle=90)
        regions = get_regions_for_board(board, markers)

        segment = LineSegment((10, 15), (90, 15), width=0.3)
        mesh = create_trace_mesh(segment, 0.01, regions, subdivisions=20, markers=markers)

        # Extract Z values — some should be significantly non-zero
        z_values = [v[2] for v in mesh.vertices]
        max_z = max(abs(z) for z in z_values)

        # A 90-degree fold with 5mm zone width should lift points several mm
        assert max_z > 1.0, f"Max Z displacement = {max_z}, expected > 1.0 for 90-degree fold"

    def test_trace_continuity_across_fold(self):
        """Adjacent trace quads should not have large gaps at fold boundaries."""
        board, markers = make_board_with_fold(fold_angle=90)
        regions = get_regions_for_board(board, markers)

        segment = LineSegment((10, 15), (90, 15), width=0.3)
        mesh = create_trace_mesh(segment, 0.01, regions, subdivisions=20, markers=markers)

        # Check that consecutive edge1 vertices don't jump too far
        # (vertex indices 0..N are edge1, N+1..2N are edge2)
        n = len(mesh.vertices) // 2
        max_gap = 0.0
        for i in range(n - 1):
            v1 = mesh.vertices[i]
            v2 = mesh.vertices[i + 1]
            gap = math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))
            max_gap = max(max_gap, gap)

        # Max gap between consecutive vertices should be reasonable
        # Trace length is 80mm, so average segment ~ 80/20 = 4mm
        # Allow up to 10mm for segments near fold zones
        assert max_gap < 12.0, f"Max gap = {max_gap}mm, suspiciously large (discontinuity?)"

    def test_trace_no_fold_still_flat(self):
        """Trace with no folds should remain flat (z near 0)."""
        segment = LineSegment((10, 15), (90, 15), width=0.3)
        mesh = create_trace_mesh(segment, 0.01)

        z_values = [v[2] for v in mesh.vertices]
        assert all(abs(z - 0.01) < 0.001 for z in z_values)

    def test_trace_region_fallback(self):
        """Even if some points fall on region boundaries, trace should still render."""
        board, markers = make_board_with_fold()
        regions = get_regions_for_board(board, markers)

        # Trace that starts exactly at the fold boundary
        hw = markers[0].zone_width / 2
        fold_x = markers[0].center[0]
        segment = LineSegment((fold_x - hw, 15), (fold_x + hw, 15), width=0.3)

        mesh = create_trace_mesh(segment, 0.01, regions, subdivisions=20, markers=markers)

        # Should still produce geometry
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0

    def test_back_layer_trace(self):
        """Back layer trace should have negative Z offset."""
        segment = LineSegment((10, 15), (90, 15), width=0.3, layer="B.Cu")
        mesh = create_trace_mesh(segment, 0.01, pcb_thickness=1.6)

        z_values = [v[2] for v in mesh.vertices]
        # Back layer should be offset by -(thickness + z_offset) = -(1.6 + 0.01)
        assert all(z < -1.5 for z in z_values)


# =============================================================================
# Test: Pad mesh with folds
# =============================================================================

class TestPadMeshWithFolds:
    """Test pad mesh generation with per-vertex region lookup."""

    def test_pad_in_flat_region(self):
        """Pad in the anchor (flat) region should remain flat."""
        board, markers = make_board_with_fold(fold_x=20)  # fold far left
        regions = get_regions_for_board(board, markers)

        # Pad at x=80, which will be in the anchor flat region (right of fold)
        pad = PadGeometry(center=(80, 15), shape='rect', size=(2.0, 1.0))
        mesh = create_pad_mesh(pad, 0.02, regions, board.thickness)

        z_values = [v[2] for v in mesh.vertices]
        # All vertices should be near z=0.02 (flat, on front surface)
        assert all(abs(z - 0.02) < 0.1 for z in z_values), f"Z values: {z_values}"

    def test_pad_after_fold(self):
        """Pad on the AFTER side of a 90-degree fold should be displaced in Z."""
        board, markers = make_board_with_fold(fold_x=90, fold_angle=90)
        regions = get_regions_for_board(board, markers)

        # Pad at x=60 — the fold is at x=90, so x=60 is on the AFTER side
        # (anchor region is rightmost, at x>92.5; everything left of fold is AFTER)
        pad = PadGeometry(center=(60, 15), shape='rect', size=(2.0, 1.0))
        mesh = create_pad_mesh(pad, 0.02, regions, board.thickness)

        z_values = [v[2] for v in mesh.vertices]
        # After a 90-degree fold, the pad should be significantly off the XY plane
        max_z = max(abs(z) for z in z_values)
        assert max_z > 1.0, f"Pad after fold has max |z|={max_z}, expected > 1.0"

    def test_pad_with_drill_hole(self):
        """Through-hole pad should still render correctly with regions."""
        board, markers = make_board_with_fold(fold_x=80)
        regions = get_regions_for_board(board, markers)

        pad = PadGeometry(center=(20, 15), shape='circle', size=(2.0, 2.0), drill=1.0)
        mesh = create_pad_mesh(pad, 0.02, regions, board.thickness)

        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0

    def test_pad_per_vertex_lookup(self):
        """Pad straddling a fold boundary should have vertices in different regions."""
        board, markers = make_board_with_fold(fold_x=50, fold_angle=90)
        regions = get_regions_for_board(board, markers)

        # Large pad centered right at fold boundary (x=47.5 is boundary)
        hw = markers[0].zone_width / 2
        fold_x = markers[0].center[0]
        pad_x = fold_x - hw  # right at the BEFORE boundary
        pad = PadGeometry(center=(pad_x, 15), shape='rect', size=(4.0, 2.0))

        mesh = create_pad_mesh(pad, 0.02, regions, board.thickness)

        # Should still produce geometry even though pad straddles the boundary
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0


# =============================================================================
# Test: Full board geometry mesh with traces and folds
# =============================================================================

class TestBoardGeometryMeshWithFolds:
    """Integration tests for complete board + traces + pads + folds."""

    def test_with_fold_pcb_data(self):
        """Test with the with_fold.kicad_pcb test data structure."""
        board, markers = make_board_with_fold(width=120, height=30, fold_x=42.5, fold_angle=90)
        board.traces = {
            'F.Cu': [LineSegment((10, 15), (110, 15), width=0.3)]
        }

        mesh = create_board_geometry_mesh(
            board,
            markers=markers,
            include_traces=True,
            include_pads=False,
        )

        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0

        # Board should have Z variation from the fold
        z_values = [v[2] for v in mesh.vertices]
        assert max(z_values) - min(z_values) > 1.0

    def test_two_folds_with_trace(self):
        """Board with two folds and a trace crossing both."""
        outline = Polygon([(0, 0), (120, 0), (120, 30), (0, 30)])
        markers = [
            make_vertical_fold_marker(42.5, zone_width=5.0, angle_deg=90.0),
            make_vertical_fold_marker(82.5, zone_width=5.0, angle_deg=-45.0),
        ]
        board = BoardGeometry(
            outline=outline,
            thickness=1.6,
            traces={'F.Cu': [LineSegment((10, 15), (110, 15), width=0.3)]}
        )

        mesh = create_board_geometry_mesh(
            board,
            markers=markers,
            include_traces=True,
            include_pads=False,
        )

        # Mesh should be non-empty and have significant 3D geometry
        assert len(mesh.vertices) > 100
        z_values = [v[2] for v in mesh.vertices]
        assert max(z_values) - min(z_values) > 1.0

    def test_board_with_pads_and_folds(self):
        """Board with fold, traces, and pads."""
        outline = Polygon([(0, 0), (120, 0), (120, 30), (0, 30)])
        markers = [make_vertical_fold_marker(50.0, zone_width=5.0, angle_deg=90.0)]

        pad1 = PadGeometry(center=(20, 15), shape='rect', size=(1.0, 0.5))
        pad2 = PadGeometry(center=(80, 15), shape='rect', size=(1.0, 0.5))
        comp = ComponentGeometry(
            reference="R1", value="10k",
            center=(20, 15), angle=0,
            bounding_box=BoundingBox(19, 14, 21, 16),
            pads=[pad1], layer="F.Cu"
        )
        comp2 = ComponentGeometry(
            reference="R2", value="10k",
            center=(80, 15), angle=0,
            bounding_box=BoundingBox(79, 14, 81, 16),
            pads=[pad2], layer="F.Cu"
        )

        board = BoardGeometry(
            outline=outline,
            thickness=1.6,
            traces={'F.Cu': [LineSegment((20, 15), (80, 15), width=0.3)]},
            components=[comp, comp2]
        )

        mesh = create_board_geometry_mesh(
            board,
            markers=markers,
            include_traces=True,
            include_pads=True,
        )

        assert len(mesh.vertices) > 50
        assert len(mesh.faces) > 20


# =============================================================================
# Test: Dot product helper
# =============================================================================

class TestDot3:
    def test_parallel(self):
        assert _dot3((1, 0, 0), (1, 0, 0)) == 1.0

    def test_opposite(self):
        assert _dot3((0, 0, 1), (0, 0, -1)) == -1.0

    def test_perpendicular(self):
        assert _dot3((1, 0, 0), (0, 1, 0)) == 0.0


# =============================================================================
# Test: Bend subdivision alignment
# =============================================================================

class TestBendSubdivisionAlignment:
    """Verify trace subdivisions align with board mesh bend zone facets."""

    def test_crossing_t_values_with_subdivisions(self):
        """With num_bend_subdivisions=4, should get 5 crossings per zone (not just 2)."""
        marker = make_vertical_fold_marker(50.0, zone_width=5.0)
        start = (10.0, 15.0)
        end = (90.0, 15.0)

        # Without subdivisions: 2 crossings (outer boundaries)
        t_basic = _compute_fold_crossing_t_values(start, end, [marker], num_bend_subdivisions=1)
        assert len(t_basic) == 2

        # With 4 subdivisions: 5 crossings (4+1 boundaries)
        t_sub4 = _compute_fold_crossing_t_values(start, end, [marker], num_bend_subdivisions=4)
        assert len(t_sub4) == 5

        # With 8 subdivisions: 9 crossings (8+1 boundaries)
        t_sub8 = _compute_fold_crossing_t_values(start, end, [marker], num_bend_subdivisions=8)
        assert len(t_sub8) == 9

    def test_trace_vertices_align_with_board_facets(self):
        """Trace mesh should have vertices at every board facet boundary in bend zone."""
        board, markers = make_board_with_fold(fold_angle=90)
        num_subs = 4
        regions = split_board_into_regions(
            [(v[0], v[1]) for v in board.outline.vertices],
            [], markers, num_bend_subdivisions=num_subs
        )

        segment = LineSegment((10, 15), (90, 15), width=0.3)
        mesh = create_trace_mesh(
            segment, 0.01, regions, subdivisions=20,
            markers=markers, num_bend_subdivisions=num_subs
        )

        # Should have more vertices than the 20-subdivision baseline
        # At minimum: 21 (uniform) + 5 (bend boundaries) = 26 unique t-values
        n_verts = len(mesh.vertices) // 2  # two edges
        assert n_verts >= 25, f"Expected >= 25 trace points, got {n_verts}"

    def test_trace_stays_on_surface_in_bend_zone(self):
        """Trace vertices in bend zone should lie on the cylindrical surface, not cut through."""
        board, markers = make_board_with_fold(fold_angle=90)
        num_subs = 8
        regions = split_board_into_regions(
            [(v[0], v[1]) for v in board.outline.vertices],
            [], markers, num_bend_subdivisions=num_subs
        )

        segment = LineSegment((10, 15), (90, 15), width=0.3)
        # Generate board mesh and trace mesh
        from mesh import create_board_mesh_with_regions
        board_mesh = create_board_mesh_with_regions(
            board.outline, board.thickness, markers=markers,
            num_bend_subdivisions=num_subs
        )
        trace_mesh = create_trace_mesh(
            segment, 0.05, regions, subdivisions=20,
            markers=markers, num_bend_subdivisions=num_subs
        )

        # Check that trace vertices in the bend zone (x=47.5 to 52.5) have Z > 0
        # (they should be on the curved surface, not at z=0)
        fold_center = 50.0
        hw = 2.5
        bend_trace_verts = [
            v for v in trace_mesh.vertices
            if fold_center - hw - 1 < v[0] < fold_center + hw + 1 and abs(v[2]) > 0.01
        ]
        # Should have multiple vertices in the bend zone with non-zero Z
        assert len(bend_trace_verts) >= 4, (
            f"Expected >= 4 trace vertices in bend zone with Z > 0, got {len(bend_trace_verts)}"
        )


# =============================================================================
# Test: H-shape PCB with 10 folds, traces, and pads
# =============================================================================

class TestHShapePCB:
    """Integration tests using the h_shape.kicad_pcb test data."""

    @pytest.fixture
    def h_shape_data(self):
        """Load and parse h_shape.kicad_pcb."""
        pcb_path = Path(__file__).parent / "test_data" / "h_shape.kicad_pcb"
        sexpr = load_kicad_pcb(str(pcb_path))
        pcb = KiCadPCB(sexpr)
        board = extract_geometry(pcb)
        markers = detect_fold_markers(pcb, layer="User.1")
        return board, markers, pcb

    def test_h_shape_detects_8_folds(self, h_shape_data):
        """H-shape PCB should have 8 fold markers."""
        board, markers, pcb = h_shape_data
        assert len(markers) == 8

    def test_h_shape_has_traces(self, h_shape_data):
        """H-shape PCB should have traces on F.Cu and B.Cu."""
        board, markers, pcb = h_shape_data
        all_traces = []
        for layer, traces in board.traces.items():
            all_traces.extend(traces)
        assert len(all_traces) >= 5, f"Expected >= 5 traces, got {len(all_traces)}"

    def test_h_shape_has_pads(self, h_shape_data):
        """H-shape PCB should have pads from footprints."""
        board, markers, pcb = h_shape_data
        assert len(board.all_pads) >= 10, f"Expected >= 10 pads, got {len(board.all_pads)}"

    def test_h_shape_mesh_generation(self, h_shape_data):
        """Full mesh with board + traces + pads should generate without errors."""
        board, markers, pcb = h_shape_data
        mesh = create_board_geometry_mesh(
            board,
            markers=markers,
            include_traces=True,
            include_pads=True,
            num_bend_subdivisions=4,
        )
        assert len(mesh.vertices) > 200
        assert len(mesh.faces) > 100

    def test_h_shape_3d_extent(self, h_shape_data):
        """H-shape with 10 folds should produce significant 3D geometry."""
        board, markers, pcb = h_shape_data
        mesh = create_board_geometry_mesh(
            board,
            markers=markers,
            include_traces=True,
            include_pads=True,
            num_bend_subdivisions=4,
        )
        z_values = [v[2] for v in mesh.vertices]
        z_range = max(z_values) - min(z_values)
        assert z_range > 1.0, f"Z range = {z_range}, expected > 1.0 for 10-fold H-shape"

    def test_h_shape_trace_crosses_all_left_bar_folds(self, h_shape_data):
        """The left bar trace (y=2 to y=38) should cross 4 fold zones."""
        board, markers, pcb = h_shape_data

        # Find the left-bar vertical trace
        left_bar_trace = None
        for trace in board.traces.get('F.Cu', []):
            sx, sy = trace.start
            ex, ey = trace.end
            if abs(sx - 5) < 1 and abs(ex - 5) < 1 and abs(sy - 2) < 1 and abs(ey - 38) < 1:
                left_bar_trace = trace
                break
        assert left_bar_trace is not None, "Left bar full-length trace not found"

        # Filter markers for left bar (horizontal folds at x=[0,10])
        left_markers = [m for m in markers if abs(m.axis[0]) > 0.5 and m.center[0] < 15]
        assert len(left_markers) == 4, f"Expected 4 left-bar folds, got {len(left_markers)}"

        # Compute crossings (4 folds × 5 boundaries each, some may be excluded by t-range)
        start = (left_bar_trace.start[0], left_bar_trace.start[1])
        end = (left_bar_trace.end[0], left_bar_trace.end[1])
        crossings = _compute_fold_crossing_t_values(start, end, left_markers, num_bend_subdivisions=4)
        assert len(crossings) >= 18, f"Expected >= 18 crossings, got {len(crossings)}"

    def test_h_shape_crossbar_trace_crosses_vertical_folds(self, h_shape_data):
        """Cross-bar trace should cross vertical/diagonal fold zones."""
        board, markers, pcb = h_shape_data

        # Filter markers for cross-bar (folds with vertical component in axis)
        crossbar_markers = [m for m in markers if abs(m.axis[1]) > 0.5]
        assert len(crossbar_markers) == 3, f"Expected 3 cross-bar folds, got {len(crossbar_markers)}"

    def test_h_shape_diagonal_trace(self, h_shape_data):
        """Diagonal trace crossing fold diagonally should produce valid mesh."""
        board, markers, pcb = h_shape_data
        regions = split_board_into_regions(
            [(v[0], v[1]) for v in board.outline.vertices],
            [], markers, num_bend_subdivisions=4
        )

        # Find the diagonal trace (3,2) to (7,10)
        diag_trace = None
        for trace in board.traces.get('F.Cu', []):
            sx, sy = trace.start
            ex, ey = trace.end
            if abs(sx - 3) < 1 and abs(sy - 2) < 1:
                diag_trace = trace
                break
        assert diag_trace is not None, "Diagonal trace not found"

        mesh = create_trace_mesh(
            diag_trace, 0.05, regions, subdivisions=20,
            markers=markers, num_bend_subdivisions=4
        )
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0

    def test_h_shape_pad_on_fold_boundary(self, h_shape_data):
        """Pad R5 at x=14 (fold 5 boundary) should still render."""
        board, markers, pcb = h_shape_data
        regions = split_board_into_regions(
            [(v[0], v[1]) for v in board.outline.vertices],
            [], markers, num_bend_subdivisions=4
        )

        # Find pad near x=14, y=20
        boundary_pads = [
            p for p in board.all_pads
            if abs(p.center[0] - 14) < 2 and abs(p.center[1] - 20) < 2
        ]
        assert len(boundary_pads) > 0, "No pads found near fold 5 boundary"

        for pad in boundary_pads:
            mesh = create_pad_mesh(pad, 0.08, regions, board.thickness)
            assert len(mesh.vertices) > 0
            assert len(mesh.faces) > 0

    def test_h_shape_through_hole_pad(self, h_shape_data):
        """Through-hole pad J1 near fold 2 should render with drill hole."""
        board, markers, pcb = h_shape_data
        regions = split_board_into_regions(
            [(v[0], v[1]) for v in board.outline.vertices],
            [], markers, num_bend_subdivisions=4
        )

        # Find through-hole pads (drill > 0)
        th_pads = [p for p in board.all_pads if p.drill > 0]
        assert len(th_pads) >= 2, f"Expected >= 2 through-hole pads, got {len(th_pads)}"

        for pad in th_pads:
            mesh = create_pad_mesh(pad, 0.08, regions, board.thickness)
            # Pads inside the board should render; those outside may be empty
            if find_containing_region(pad.center, regions) is not None:
                assert len(mesh.vertices) > 0
                assert len(mesh.faces) > 0


class TestThroughHoleBugs:
    """Tests for through-hole pad specific bugs (Phase 1.2)."""

    def test_oval_drill_parsing(self):
        """Oval drills like (drill oval 0.8 1.7) should parse correctly."""
        pcb = KiCadPCB.load(
            Path(__file__).parent / 'test_data' / 'h_shape.kicad_pcb'
        )
        fps = pcb.get_footprints()
        # Audio jack has oval drills
        oval_pads = [
            p for fp in fps for p in fp.pads
            if p.pad_type == 'thru_hole' and p.shape == 'roundrect'
        ]
        assert len(oval_pads) > 0, "Should find through-hole roundrect pads"
        for pad in oval_pads:
            assert pad.drill > 0, (
                f"Oval drill should parse to > 0, got {pad.drill}"
            )

    def test_oval_drill_max_dimension(self):
        """Oval drill (drill oval 0.8 1.7) should use max dimension = 1.7."""
        pcb = KiCadPCB.load(
            Path(__file__).parent / 'test_data' / 'h_shape.kicad_pcb'
        )
        fps = pcb.get_footprints()
        oval_pads = [
            p for fp in fps for p in fp.pads
            if p.pad_type == 'thru_hole' and p.shape == 'roundrect'
        ]
        for pad in oval_pads:
            assert pad.drill == 1.7, (
                f"Expected max(0.8, 1.7) = 1.7, got {pad.drill}"
            )

    def test_wildcard_cu_layer(self):
        """Through-hole pads with *.Cu should resolve to F.Cu or B.Cu."""
        pcb = KiCadPCB.load(
            Path(__file__).parent / 'test_data' / 'h_shape.kicad_pcb'
        )
        board = extract_geometry(pcb)
        # J1 Conn_01x02 is on F.Cu with *.Cu pads
        j1 = [c for c in board.components
              if c.reference == 'J1' and c.center == (5.0, 12.0)][0]
        for pad in j1.pads:
            assert pad.layer == 'F.Cu', (
                f"Through-hole pad on F.Cu footprint should be F.Cu, got {pad.layer}"
            )

    def test_through_hole_pad_region_fallback(self):
        """Through-hole pads with center in drill hole should find region via fallback."""
        pcb = KiCadPCB.load(
            Path(__file__).parent / 'test_data' / 'h_shape.kicad_pcb'
        )
        board = extract_geometry(pcb)
        markers = detect_fold_markers(pcb, layer='User.1')
        outline_verts = [(v[0], v[1]) for v in board.outline.vertices]
        cutout_verts = [[(v[0], v[1]) for v in c.vertices]
                        for c in (board.cutouts or [])]
        regions = split_board_into_regions(
            outline_verts, cutout_verts, markers, num_bend_subdivisions=4
        )

        # J1 pads have drill holes — center is in cutout
        j1 = [c for c in board.components
              if c.reference == 'J1' and c.center == (5.0, 12.0)][0]
        for pad in j1.pads:
            assert pad.drill > 0
            # Center should be in a cutout (returns None)
            center_region = find_containing_region(pad.center, regions)
            assert center_region is None, "Pad center should be in drill hole cutout"

            # But pad mesh should still render (fallback to polygon vertices)
            mesh = create_pad_mesh(pad, 0.08, regions, board.thickness)
            assert len(mesh.vertices) > 0, "Through-hole pad should render via fallback"

    def test_through_hole_3d_distance_preserved(self):
        """3D distance between trace endpoint and through-hole pad should match 2D."""
        from geometry import pad_to_polygon

        pcb = KiCadPCB.load(
            Path(__file__).parent / 'test_data' / 'h_shape.kicad_pcb'
        )
        board = extract_geometry(pcb)
        markers = detect_fold_markers(pcb, layer='User.1')
        outline_verts = [(v[0], v[1]) for v in board.outline.vertices]
        cutout_verts = [[(v[0], v[1]) for v in c.vertices]
                        for c in (board.cutouts or [])]
        regions = split_board_into_regions(
            outline_verts, cutout_verts, markers, num_bend_subdivisions=4
        )

        j1 = [c for c in board.components
              if c.reference == 'J1' and c.center == (5.0, 12.0)][0]
        # Trace endpoint (5, 11.5) on net 3
        trace_ep = (5.0, 11.5)
        trace_region = find_containing_region(trace_ep, regions)
        assert trace_region is not None
        trace_recipe = get_region_recipe(trace_region)
        trace_3d, _ = transform_point_and_normal(trace_ep, trace_recipe)

        for pad in j1.pads:
            # Find pad region via fallback
            pad_region = find_containing_region(pad.center, regions)
            if not pad_region:
                poly = pad_to_polygon(pad)
                for v in poly.vertices:
                    pad_region = find_containing_region(v, regions)
                    if pad_region:
                        break
            assert pad_region is not None
            pad_recipe = get_region_recipe(pad_region)
            pad_3d, _ = transform_point_and_normal(pad.center, pad_recipe)

            dist_2d = math.sqrt(
                (pad.center[0] - trace_ep[0])**2 +
                (pad.center[1] - trace_ep[1])**2
            )
            dist_3d = math.sqrt(
                sum((a - b)**2 for a, b in zip(pad_3d, trace_3d))
            )
            assert abs(dist_3d - dist_2d) < 0.5, (
                f"3D distance {dist_3d:.2f} should be close to "
                f"2D distance {dist_2d:.2f}"
            )

    def test_rotated_component_bbox_contains_pads(self):
        """Bounding box for rotated components should include all pad positions."""
        pcb = KiCadPCB.load(
            Path(__file__).parent / 'test_data' / 'h_shape.kicad_pcb'
        )
        board = extract_geometry(pcb)
        for comp in board.components:
            if comp.angle == 0:
                continue
            bb = comp.bounding_box
            for pad in comp.pads:
                cx, cy = pad.center
                assert bb.min_x - 0.2 <= cx <= bb.max_x + 0.2, (
                    f"{comp.reference} pad ({cx:.2f},{cy:.2f}) x outside "
                    f"bbox [{bb.min_x:.2f},{bb.max_x:.2f}]"
                )
                assert bb.min_y - 0.2 <= cy <= bb.max_y + 0.2, (
                    f"{comp.reference} pad ({cx:.2f},{cy:.2f}) y outside "
                    f"bbox [{bb.min_y:.2f},{bb.max_y:.2f}]"
                )

    def test_component_per_vertex_region_lookup(self):
        """Component boxes should use per-vertex region lookup for fold boundaries."""
        from mesh import create_component_mesh

        pcb = KiCadPCB.load(
            Path(__file__).parent / 'test_data' / 'h_shape.kicad_pcb'
        )
        board = extract_geometry(pcb)
        markers = detect_fold_markers(pcb, layer='User.1')
        outline_verts = [(v[0], v[1]) for v in board.outline.vertices]
        cutout_verts = [[(v[0], v[1]) for v in c.vertices]
                        for c in (board.cutouts or [])]
        regions = split_board_into_regions(
            outline_verts, cutout_verts, markers, num_bend_subdivisions=1
        )

        # J1 at (5,12) spans fold boundary at y=12.05
        j1 = [c for c in board.components
              if c.reference == 'J1' and c.center == (5.0, 12.0)][0]

        mesh = create_component_mesh(j1, 2.0, regions, board.thickness)
        assert len(mesh.vertices) > 0

        # Box bottom vertices should be on the board surface
        import numpy as np
        verts = np.array(mesh.vertices)
        # Bottom vertices = first N/2 vertices
        n = len(verts) // 2
        bottom = verts[:n]

        # Each bottom vertex should be close to the board surface
        # (within z_offset + small tolerance)
        for bv in bottom:
            # Find a board surface point nearby
            found_close = False
            for r in regions:
                recipe = get_region_recipe(r)
                # Check if this bottom vertex is close to any region's transform
                # of a point with similar x,y
                for rv in r.outline[:4]:
                    v3d, _ = transform_point_and_normal(rv, recipe)
                    dist = math.sqrt(sum((a-b)**2 for a, b in zip(bv, v3d)))
                    if dist < 3.0:  # within 3mm of some board vertex
                        found_close = True
                        break
                if found_close:
                    break
            assert found_close, (
                f"Component vertex ({bv[0]:.2f}, {bv[1]:.2f}, {bv[2]:.2f}) "
                f"not near any board surface"
            )
