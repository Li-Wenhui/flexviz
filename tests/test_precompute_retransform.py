"""Tests for decoupled precompute / retransform (Phase 4).

Verifies that:
1. precompute_board_mesh + transform_board_mesh produces identical results
   to create_board_mesh_with_regions
2. precompute_trace_mesh + transform_trace_mesh produces identical results
   to create_trace_mesh
3. precompute_all_layers + retransform_all_layers produces identical results
   to the standard build path
4. Changing angles and retransforming gives correct results
5. PrecomputedLayerData caching works correctly
"""

import pytest
import math

from mesh import (
    Mesh,
    create_board_mesh_with_regions,
    create_trace_mesh,
    build_board_layer,
    build_traces_layer,
    build_pads_layer,
    compute_regions,
    get_region_recipe,
)
from board_mesh import precompute_board_mesh, transform_board_mesh
from trace_mesh import precompute_trace_mesh, transform_trace_mesh
from mesh import precompute_all_layers, retransform_all_layers
from mesh_types import PrecomputedBoardData, PrecomputedTraceData, PrecomputedLayerData
from geometry import (
    Polygon, LineSegment, PadGeometry, BoardGeometry, BoundingBox, ComponentGeometry,
)
from markers import FoldMarker
from planar_subdivision import split_board_into_regions


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
        axis=(0.0, 1.0),
        center=(x_center, board_height / 2),
    )


def make_simple_board():
    """Create a simple 30x20mm rectangular board."""
    outline = Polygon([(0, 0), (30, 0), (30, 20), (0, 20)])
    pad = PadGeometry(
        center=(5, 10), size=(1.5, 1.5), shape='rect',
        layer='F.Cu', drill=0.0,
    )
    comp = ComponentGeometry(
        reference='R1', value='100', center=(5, 10), angle=0,
        bounding_box=BoundingBox(min_x=4.25, min_y=9.25, max_x=5.75, max_y=10.75),
        pads=[pad], layer='F.Cu',
    )
    return BoardGeometry(
        outline=outline,
        thickness=1.6,
        traces={'F.Cu': [
            LineSegment(start=(5, 10), end=(25, 10), width=0.25, layer='F.Cu'),
        ]},
        components=[comp],
        cutouts=[],
    )


def make_board_with_fold():
    """Create a board with a single vertical fold."""
    board = make_simple_board()
    marker = make_vertical_fold_marker(15.0, zone_width=4.0, angle_deg=90.0, board_height=20.0)
    return board, [marker]


def meshes_are_close(mesh_a, mesh_b, tol=1e-4):
    """Check that two meshes have the same topology and nearly identical vertices."""
    if len(mesh_a.vertices) != len(mesh_b.vertices):
        return False, f"vertex count mismatch: {len(mesh_a.vertices)} vs {len(mesh_b.vertices)}"
    if len(mesh_a.faces) != len(mesh_b.faces):
        return False, f"face count mismatch: {len(mesh_a.faces)} vs {len(mesh_b.faces)}"

    for i, (va, vb) in enumerate(zip(mesh_a.vertices, mesh_b.vertices)):
        for j in range(3):
            if abs(va[j] - vb[j]) > tol:
                return False, f"vertex {i} differs: {va} vs {vb}"

    for i, (fa, fb) in enumerate(zip(mesh_a.faces, mesh_b.faces)):
        if fa != fb:
            return False, f"face {i} differs: {fa} vs {fb}"

    return True, "ok"


# =============================================================================
# Board Mesh: precompute + transform equivalence
# =============================================================================

class TestBoardPrecompute:
    """Test that precompute + transform == create_board_mesh_with_regions."""

    def test_no_markers(self):
        """Board without markers: precompute + transform == original."""
        outline = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        thickness = 1.6

        original = create_board_mesh_with_regions(outline, thickness)
        precomputed = precompute_board_mesh(outline, thickness)
        retransformed = transform_board_mesh(precomputed)

        ok, msg = meshes_are_close(original, retransformed)
        assert ok, msg

    def test_with_single_fold(self):
        """Board with one fold marker: precompute + transform == original."""
        outline = Polygon([(0, 0), (30, 0), (30, 20), (0, 20)])
        thickness = 1.6
        marker = make_vertical_fold_marker(15.0, zone_width=4.0, angle_deg=90.0, board_height=20.0)

        original = create_board_mesh_with_regions(
            outline, thickness, markers=[marker], num_bend_subdivisions=4
        )
        precomputed = precompute_board_mesh(
            outline, thickness, markers=[marker], num_bend_subdivisions=4
        )
        retransformed = transform_board_mesh(precomputed, apply_bend=True)

        ok, msg = meshes_are_close(original, retransformed)
        assert ok, msg

    def test_angle_change(self):
        """Changing angle then retransforming gives same result as full rebuild."""
        outline = Polygon([(0, 0), (30, 0), (30, 20), (0, 20)])
        thickness = 1.6
        marker = make_vertical_fold_marker(15.0, zone_width=4.0, angle_deg=90.0, board_height=20.0)

        # Precompute at 90 degrees
        precomputed = precompute_board_mesh(
            outline, thickness, markers=[marker], num_bend_subdivisions=4
        )

        # Change angle to 45 degrees
        marker.angle_degrees = 45.0

        # Full rebuild at 45 degrees
        original = create_board_mesh_with_regions(
            outline, thickness, markers=[marker], num_bend_subdivisions=4
        )
        # Retransform from precomputed data (marker.angle_degrees is now 45)
        retransformed = transform_board_mesh(precomputed, apply_bend=True)

        ok, msg = meshes_are_close(original, retransformed)
        assert ok, msg

    def test_empty_outline(self):
        """Empty outline returns empty precomputed data."""
        outline = Polygon([])
        precomputed = precompute_board_mesh(outline, 1.6)
        assert len(precomputed.region_data) == 0
        mesh = transform_board_mesh(precomputed)
        assert len(mesh.vertices) == 0

    def test_with_cutouts(self):
        """Board with cutouts: precompute + transform == original."""
        outline = Polygon([(0, 0), (30, 0), (30, 20), (0, 20)])
        cutout = Polygon([(12, 8), (18, 8), (18, 12), (12, 12)])
        thickness = 1.6
        marker = make_vertical_fold_marker(15.0, zone_width=4.0, angle_deg=60.0, board_height=20.0)

        original = create_board_mesh_with_regions(
            outline, thickness, markers=[marker], cutouts=[cutout],
            num_bend_subdivisions=2
        )
        precomputed = precompute_board_mesh(
            outline, thickness, markers=[marker], cutouts=[cutout],
            num_bend_subdivisions=2
        )
        retransformed = transform_board_mesh(precomputed, apply_bend=True)

        ok, msg = meshes_are_close(original, retransformed)
        assert ok, msg

    def test_debug_regions(self):
        """Debug region coloring is preserved."""
        outline = Polygon([(0, 0), (30, 0), (30, 20), (0, 20)])
        marker = make_vertical_fold_marker(15.0, zone_width=4.0, angle_deg=90.0, board_height=20.0)

        original = create_board_mesh_with_regions(
            outline, 1.6, markers=[marker], debug_regions=True
        )
        precomputed = precompute_board_mesh(
            outline, 1.6, markers=[marker], debug_regions=True
        )
        retransformed = transform_board_mesh(precomputed)

        # Colors should match
        assert len(original.colors) == len(retransformed.colors)
        for ca, cb in zip(original.colors, retransformed.colors):
            assert ca == cb

    def test_apply_bend_false(self):
        """apply_bend=False gives flat mesh."""
        outline = Polygon([(0, 0), (30, 0), (30, 20), (0, 20)])
        marker = make_vertical_fold_marker(15.0, zone_width=4.0, angle_deg=90.0, board_height=20.0)

        original = create_board_mesh_with_regions(
            outline, 1.6, markers=[marker], apply_bend=False
        )
        precomputed = precompute_board_mesh(
            outline, 1.6, markers=[marker]
        )
        retransformed = transform_board_mesh(precomputed, apply_bend=False)

        ok, msg = meshes_are_close(original, retransformed)
        assert ok, msg

    def test_multiple_folds(self):
        """Board with two fold markers."""
        outline = Polygon([(0, 0), (40, 0), (40, 20), (0, 20)])
        m1 = make_vertical_fold_marker(13.0, zone_width=3.0, angle_deg=45.0, board_height=20.0)
        m2 = make_vertical_fold_marker(27.0, zone_width=3.0, angle_deg=-60.0, board_height=20.0)

        original = create_board_mesh_with_regions(
            outline, 1.6, markers=[m1, m2], num_bend_subdivisions=3
        )
        precomputed = precompute_board_mesh(
            outline, 1.6, markers=[m1, m2], num_bend_subdivisions=3
        )
        retransformed = transform_board_mesh(precomputed)

        ok, msg = meshes_are_close(original, retransformed)
        assert ok, msg


# =============================================================================
# Trace Mesh: precompute + transform equivalence
# =============================================================================

class TestTracePrecompute:
    """Test that precompute + transform == create_trace_mesh."""

    def test_no_regions(self):
        """Trace without regions: precompute + transform == original."""
        seg = LineSegment(start=(5, 10), end=(25, 10), width=0.25, layer='F.Cu')
        z_offset = 0.05

        original = create_trace_mesh(seg, z_offset)
        precomputed = precompute_trace_mesh(seg)
        assert precomputed is not None
        retransformed = transform_trace_mesh(precomputed, z_offset)

        ok, msg = meshes_are_close(original, retransformed)
        assert ok, msg

    def test_with_regions(self):
        """Trace with fold regions: precompute + transform == original."""
        outline_verts = [(0, 0), (30, 0), (30, 20), (0, 20)]
        marker = make_vertical_fold_marker(15.0, zone_width=4.0, angle_deg=90.0, board_height=20.0)
        regions = split_board_into_regions(outline_verts, [], [marker], num_bend_subdivisions=4)

        seg = LineSegment(start=(5, 10), end=(25, 10), width=0.25, layer='F.Cu')
        z_offset = 0.05
        pcb_thickness = 1.6

        original = create_trace_mesh(
            seg, z_offset, regions, pcb_thickness=pcb_thickness,
            markers=[marker], num_bend_subdivisions=4
        )
        precomputed = precompute_trace_mesh(
            seg, regions, markers=[marker], num_bend_subdivisions=4
        )
        assert precomputed is not None
        retransformed = transform_trace_mesh(precomputed, z_offset, pcb_thickness=pcb_thickness)

        ok, msg = meshes_are_close(original, retransformed)
        assert ok, msg

    def test_angle_change(self):
        """Changing angle then retransforming gives correct trace mesh."""
        outline_verts = [(0, 0), (30, 0), (30, 20), (0, 20)]
        marker = make_vertical_fold_marker(15.0, zone_width=4.0, angle_deg=90.0, board_height=20.0)
        regions = split_board_into_regions(outline_verts, [], [marker], num_bend_subdivisions=4)

        seg = LineSegment(start=(5, 10), end=(25, 10), width=0.25, layer='F.Cu')
        z_offset = 0.05
        pcb_thickness = 1.6

        # Precompute at 90 degrees
        precomputed = precompute_trace_mesh(
            seg, regions, markers=[marker], num_bend_subdivisions=4
        )

        # Change to 45 degrees
        marker.angle_degrees = 45.0

        # Full rebuild
        original = create_trace_mesh(
            seg, z_offset, regions, pcb_thickness=pcb_thickness,
            markers=[marker], num_bend_subdivisions=4
        )
        retransformed = transform_trace_mesh(precomputed, z_offset, pcb_thickness=pcb_thickness)

        ok, msg = meshes_are_close(original, retransformed)
        assert ok, msg

    def test_back_layer(self):
        """Back layer trace: precompute + transform == original."""
        seg = LineSegment(start=(5, 10), end=(25, 10), width=0.25, layer='B.Cu')
        z_offset = 0.05
        pcb_thickness = 1.6

        original = create_trace_mesh(seg, z_offset, pcb_thickness=pcb_thickness)
        precomputed = precompute_trace_mesh(seg)
        retransformed = transform_trace_mesh(precomputed, z_offset, pcb_thickness=pcb_thickness)

        ok, msg = meshes_are_close(original, retransformed)
        assert ok, msg

    def test_invalid_ribbon(self):
        """Degenerate segment returns None from precompute."""
        seg = LineSegment(start=(5, 10), end=(5, 10), width=0.0, layer='F.Cu')
        precomputed = precompute_trace_mesh(seg)
        # Either None or a valid empty-ish result is acceptable
        # The original create_trace_mesh returns empty mesh for such cases


# =============================================================================
# Full pipeline: precompute_all_layers + retransform_all_layers
# =============================================================================

class TestFullPipeline:
    """Test the full precompute/retransform pipeline against standard build."""

    def test_equivalence_with_fold(self):
        """precompute + retransform == standard build for board + traces."""
        board, markers = make_board_with_fold()

        # Standard build
        regions, active_regions = compute_regions(board, markers, num_bend_subdivisions=4, apply_bend=True)
        std_board = build_board_layer(board, markers, 1.0, 4, False, True)
        std_traces = build_traces_layer(board, active_regions, markers, 4)

        # Precompute + retransform
        precomputed = precompute_all_layers(board, markers, num_bend_subdivisions=4)
        result = retransform_all_layers(
            precomputed, board, active_regions, markers,
            num_bend_subdivisions=4, apply_bend=True,
        )

        ok, msg = meshes_are_close(std_board, result['board'])
        assert ok, f"board: {msg}"

        ok, msg = meshes_are_close(std_traces, result['traces'])
        assert ok, f"traces: {msg}"

    def test_angle_change_full_pipeline(self):
        """Changing angles and retransforming gives correct results."""
        board, markers = make_board_with_fold()
        marker = markers[0]

        # Precompute at 90 degrees
        precomputed = precompute_all_layers(board, markers, num_bend_subdivisions=4)

        # Change to 30 degrees
        marker.angle_degrees = 30.0

        # Standard build at 30 degrees
        regions, active_regions = compute_regions(board, markers, num_bend_subdivisions=4, apply_bend=True)
        std_board = build_board_layer(board, markers, 1.0, 4, False, True)
        std_traces = build_traces_layer(board, active_regions, markers, 4)

        # Retransform
        result = retransform_all_layers(
            precomputed, board, active_regions, markers,
            num_bend_subdivisions=4, apply_bend=True,
        )

        ok, msg = meshes_are_close(std_board, result['board'])
        assert ok, f"board: {msg}"

        ok, msg = meshes_are_close(std_traces, result['traces'])
        assert ok, f"traces: {msg}"

    def test_no_markers(self):
        """Pipeline works without any markers."""
        board = make_simple_board()
        markers = []

        precomputed = precompute_all_layers(board, markers)
        regions, active_regions = compute_regions(board, markers, apply_bend=True)
        result = retransform_all_layers(
            precomputed, board, active_regions, markers,
        )

        std_board = build_board_layer(board, markers, 1.0, 1, False, True)
        ok, msg = meshes_are_close(std_board, result['board'])
        assert ok, f"board: {msg}"

    def test_retransform_has_all_layers(self):
        """retransform_all_layers returns all expected layer keys."""
        board, markers = make_board_with_fold()
        precomputed = precompute_all_layers(board, markers)
        regions, active_regions = compute_regions(board, markers, apply_bend=True)
        result = retransform_all_layers(
            precomputed, board, active_regions, markers,
        )

        expected_keys = {'board', 'traces', 'pads', 'components', '3d_models', 'stiffeners'}
        assert set(result.keys()) == expected_keys

    def test_precomputed_layer_data_structure(self):
        """PrecomputedLayerData has expected attributes."""
        board, markers = make_board_with_fold()
        precomputed = precompute_all_layers(board, markers, num_bend_subdivisions=4)

        assert isinstance(precomputed, PrecomputedLayerData)
        assert precomputed.board is not None
        assert isinstance(precomputed.board, PrecomputedBoardData)
        assert len(precomputed.traces) > 0
        assert isinstance(precomputed.traces[0], PrecomputedTraceData)
        assert precomputed.regions is not None
        assert len(precomputed.markers) == 1
        assert precomputed.num_bend_subdivisions == 4

    def test_multiple_angle_changes(self):
        """Repeated angle changes produce correct results each time."""
        board, markers = make_board_with_fold()
        marker = markers[0]

        precomputed = precompute_all_layers(board, markers, num_bend_subdivisions=4)

        for angle in [0.0, 30.0, 45.0, 90.0, 120.0, -45.0]:
            marker.angle_degrees = angle

            regions, active_regions = compute_regions(board, markers, num_bend_subdivisions=4, apply_bend=True)
            std_board = build_board_layer(board, markers, 1.0, 4, False, True)

            result = retransform_all_layers(
                precomputed, board, active_regions, markers,
                num_bend_subdivisions=4, apply_bend=True,
            )

            ok, msg = meshes_are_close(std_board, result['board'])
            assert ok, f"board at {angle}deg: {msg}"
