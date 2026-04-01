"""Unit tests for bend_transform module."""

import pytest
import math
from pathlib import Path

from bend_transform import (
    FoldDefinition,
    transform_point,
    compute_normal,
    transform_point_and_normal,
    create_fold_definitions,
)
from markers import FoldMarker


class TestFoldDefinition:
    """Tests for FoldDefinition class."""

    def test_create(self):
        """Test creating a fold definition."""
        fold = FoldDefinition(
            center=(50, 15),
            axis=(0, 1),
            zone_width=5.0,
            angle=math.pi / 2
        )
        assert fold.center == (50, 15)
        assert fold.angle == math.pi / 2

    def test_radius(self):
        """Test radius calculation."""
        fold = FoldDefinition(
            center=(50, 15),
            axis=(0, 1),
            zone_width=5.0,
            angle=math.pi / 2  # 90 degrees
        )
        # radius = zone_width / angle = 5 / (pi/2) ≈ 3.18
        expected = 5.0 / (math.pi / 2)
        assert abs(fold.radius - expected) < 0.01

    def test_radius_zero_angle(self):
        """Test radius with zero angle."""
        fold = FoldDefinition(
            center=(50, 15),
            axis=(0, 1),
            zone_width=5.0,
            angle=0.0
        )
        assert fold.radius == float('inf')

    def test_perp(self):
        """Test perpendicular vector calculation."""
        # Vertical fold axis
        fold = FoldDefinition(
            center=(50, 15),
            axis=(0, 1),
            zone_width=5.0,
            angle=math.pi / 2
        )
        perp = fold.perp
        assert abs(perp[0] - (-1)) < 0.01
        assert abs(perp[1]) < 0.01

        # Horizontal fold axis
        fold = FoldDefinition(
            center=(50, 15),
            axis=(1, 0),
            zone_width=5.0,
            angle=math.pi / 2
        )
        perp = fold.perp
        assert abs(perp[0]) < 0.01
        assert abs(perp[1] - 1) < 0.01

    def test_from_marker(self):
        """Test creating from FoldMarker."""
        marker = FoldMarker(
            line_a_start=(40, 0),
            line_a_end=(40, 30),
            line_b_start=(45, 0),
            line_b_end=(45, 30),
            angle_degrees=90.0,
            zone_width=5.0,
            radius=3.18,
            axis=(0, 1),
            center=(42.5, 15)
        )

        fold = FoldDefinition.from_marker(marker)
        assert fold.center == (42.5, 15)
        assert fold.axis == (0, 1)
        assert fold.zone_width == 5.0
        assert abs(fold.angle - math.pi / 2) < 0.01


class TestTransformPoint:
    """Tests for recipe-based point transformation."""

    def test_empty_recipe(self):
        """Test with empty recipe returns flat point."""
        result = transform_point((50, 15), [])
        assert result == (50, 15, 0.0)

    def test_in_zone_transformation(self):
        """Test point in fold zone gets curved."""
        fold = FoldDefinition(
            center=(50, 15),
            axis=(0, 1),  # Vertical fold
            zone_width=10.0,
            angle=math.pi / 2  # 90 degrees
        )

        # Recipe: point is IN_ZONE
        recipe = [(fold, "IN_ZONE", False)]
        result = transform_point((50, 15), recipe)

        # Point at center of zone should have some z displacement
        # For 90-degree fold at midpoint (45 degrees), z > 0
        assert result[2] > 0

    def test_after_transformation(self):
        """Test point after fold zone gets rotated."""
        fold = FoldDefinition(
            center=(50, 15),
            axis=(0, 1),  # Vertical fold
            zone_width=10.0,
            angle=math.pi / 2  # 90 degrees
        )

        # Recipe: point is AFTER (past the fold zone)
        recipe = [(fold, "AFTER", False)]
        result = transform_point((60, 15), recipe)

        # After 90-degree fold, point should have significant z displacement
        assert abs(result[2]) > 1

    def test_negative_angle_opposite_direction(self):
        """Test negative angle bends in opposite direction."""
        fold_pos = FoldDefinition(
            center=(50, 15),
            axis=(0, 1),
            zone_width=10.0,
            angle=math.pi / 2
        )
        fold_neg = FoldDefinition(
            center=(50, 15),
            axis=(0, 1),
            zone_width=10.0,
            angle=-math.pi / 2
        )

        recipe_pos = [(fold_pos, "AFTER", False)]
        recipe_neg = [(fold_neg, "AFTER", False)]

        result_pos = transform_point((60, 15), recipe_pos)
        result_neg = transform_point((60, 15), recipe_neg)

        # Opposite angles should produce opposite z directions
        assert result_pos[2] * result_neg[2] < 0

    def test_back_entry_mirroring(self):
        """Test back entry mirrors the transformation."""
        fold = FoldDefinition(
            center=(50, 15),
            axis=(0, 1),
            zone_width=10.0,
            angle=math.pi / 2
        )

        # Same position, different entry direction
        recipe_front = [(fold, "AFTER", False)]
        recipe_back = [(fold, "AFTER", True)]

        result_front = transform_point((60, 15), recipe_front)
        result_back = transform_point((40, 15), recipe_back)

        # Back entry should produce mirrored perpendicular position
        # Both should have z displacement (same sign due to mirroring)
        assert abs(result_front[2]) > 1
        assert abs(result_back[2]) > 1


class TestComputeNormal:
    """Tests for normal computation."""

    def test_flat_normal(self):
        """Test normal with no folds is (0, 0, 1)."""
        normal = compute_normal((50, 15), [])
        assert abs(normal[0]) < 0.01
        assert abs(normal[1]) < 0.01
        assert abs(normal[2] - 1.0) < 0.01

    def test_after_fold_normal_rotated(self):
        """Test normal after fold is rotated."""
        fold = FoldDefinition(
            center=(50, 15),
            axis=(0, 1),  # Vertical fold
            zone_width=10.0,
            angle=math.pi / 2  # 90 degrees
        )

        recipe = [(fold, "AFTER", False)]
        normal = compute_normal((60, 15), recipe)

        # After 90-degree fold, normal should be significantly rotated from (0,0,1)
        # The exact direction depends on fold axis orientation
        assert abs(normal[0]) > 0.9 or abs(normal[2]) < 0.1  # Normal is rotated


class TestCreateFoldDefinitions:
    """Tests for creating fold definitions from markers."""

    def test_create_from_markers(self):
        """Test creating fold definitions from markers."""
        markers = [
            FoldMarker(
                line_a_start=(40, 0),
                line_a_end=(40, 30),
                line_b_start=(45, 0),
                line_b_end=(45, 30),
                angle_degrees=90.0,
                zone_width=5.0,
                radius=3.18,
                axis=(0, 1),
                center=(42.5, 15)
            ),
            FoldMarker(
                line_a_start=(80, 0),
                line_a_end=(80, 30),
                line_b_start=(85, 0),
                line_b_end=(85, 30),
                angle_degrees=-45.0,
                zone_width=5.0,
                radius=6.37,
                axis=(0, 1),
                center=(82.5, 15)
            )
        ]

        folds = create_fold_definitions(markers)

        assert len(folds) == 2
        assert abs(folds[0].angle - math.pi / 2) < 0.01
        assert abs(folds[1].angle - (-math.pi / 4)) < 0.01

    def test_create_empty(self):
        """Test with no markers."""
        folds = create_fold_definitions([])
        assert folds == []


# =============================================================================
# Edge Case Tests
# =============================================================================

from bend_transform import _rotation_matrix_around_axis, _multiply_matrices, _apply_rotation


class TestTransformPointEdgeCases:
    """Edge case tests for transform_point."""

    def test_zero_angle_fold_in_zone(self):
        """Zero-angle fold IN_ZONE should leave the point flat."""
        fold = FoldDefinition(
            center=(50, 15),
            axis=(0, 1),
            zone_width=10.0,
            angle=0.0,
        )
        # Point at center: perp_dist = 0, along = 0
        recipe = [(fold, "IN_ZONE", False)]
        result = transform_point((50, 15), recipe)
        # With angle=0, local_perp = dist_into_zone - hw = 5 - 5 = 0
        # local_up = 0, so position = fold_center + 0*perp + 0*up
        # perp direction is (-1, 0), so x = 50 + 0*(-1) + 0*0 = 50
        assert result[0] == pytest.approx(50, abs=1e-6)
        assert result[1] == pytest.approx(15, abs=1e-6)
        assert result[2] == pytest.approx(0.0, abs=1e-6)

    def test_very_small_angle(self):
        """Very small angle fold AFTER should produce near-flat result."""
        fold = FoldDefinition(
            center=(50, 15),
            axis=(0, 1),
            zone_width=10.0,
            angle=0.001,
        )
        # Point 20 units past the fold center (10 past zone end)
        recipe = [(fold, "AFTER", False)]
        result = transform_point((70, 15), recipe)
        # z displacement should be very small for a near-zero angle
        assert abs(result[2]) < 0.5

    def test_180_degree_fold_after(self):
        """180-degree fold AFTER: zone_end should be at z = 2R."""
        angle = math.pi
        zone_width = 10.0
        R = zone_width / abs(angle)
        fold = FoldDefinition(
            center=(50, 15),
            axis=(0, 1),
            zone_width=zone_width,
            angle=angle,
        )
        # Point just at the zone boundary (perp_dist = hw = 5, excess = 0)
        recipe = [(fold, "AFTER", False)]
        result = transform_point((55, 15), recipe)
        # zone_end_up = R * (1 - cos(pi)) = R * 2 = 2R
        expected_z = 2 * R
        assert result[2] == pytest.approx(expected_z, abs=1e-6)
        # zone_end_perp = R * sin(pi) - hw ≈ 0 - 5 = -5
        # sin(pi) ≈ 0, so local_perp ≈ -hw = -5
        # perp direction is (-1, 0), so x offset = -5 * (-1) = 5
        # x = 50 + 5 = 55... but let's just verify z is correct
        assert result[2] > 0

    def test_negative_180_fold(self):
        """Negative 180-degree fold should produce z opposite to positive."""
        zone_width = 10.0
        fold_pos = FoldDefinition(
            center=(50, 15), axis=(0, 1), zone_width=zone_width, angle=math.pi
        )
        fold_neg = FoldDefinition(
            center=(50, 15), axis=(0, 1), zone_width=zone_width, angle=-math.pi
        )
        recipe_pos = [(fold_pos, "AFTER", False)]
        recipe_neg = [(fold_neg, "AFTER", False)]
        result_pos = transform_point((55, 15), recipe_pos)
        result_neg = transform_point((55, 15), recipe_neg)
        # z should be opposite sign
        assert result_pos[2] == pytest.approx(-result_neg[2], abs=1e-6)

    def test_270_degree_fold(self):
        """270-degree fold should not crash and produce z > 0."""
        fold = FoldDefinition(
            center=(50, 15),
            axis=(0, 1),
            zone_width=10.0,
            angle=3 * math.pi / 2,
        )
        recipe = [(fold, "AFTER", False)]
        result = transform_point((60, 15), recipe)
        # Should not crash; z > 0 for positive angle
        assert result[2] > 0

    def test_in_zone_at_before_boundary(self):
        """Point exactly at the BEFORE boundary (perp_dist = -hw) should have z ≈ 0."""
        fold = FoldDefinition(
            center=(50, 15),
            axis=(0, 1),
            zone_width=10.0,
            angle=math.pi / 2,
        )
        # perp = (-1, 0). Point at x=55 → perp_dist = (55-50)*(-1) = -5 = -hw
        recipe = [(fold, "IN_ZONE", False)]
        result = transform_point((55, 15), recipe)
        # At before boundary: theta = 0, local_perp = -hw, local_up = 0
        assert result[2] == pytest.approx(0.0, abs=1e-6)

    def test_in_zone_at_after_boundary(self):
        """Point at AFTER boundary (perp_dist = +hw) via IN_ZONE should match AFTER at excess=0."""
        fold = FoldDefinition(
            center=(50, 15),
            axis=(0, 1),
            zone_width=10.0,
            angle=math.pi / 2,
        )
        # perp = (-1, 0). perp_dist = +hw = 5 means x = 50 + 5*(-1) = 45
        recipe_in = [(fold, "IN_ZONE", False)]
        result_in = transform_point((45, 15), recipe_in)

        recipe_after = [(fold, "AFTER", False)]
        result_after = transform_point((45, 15), recipe_after)

        assert result_in[0] == pytest.approx(result_after[0], abs=1e-6)
        assert result_in[1] == pytest.approx(result_after[1], abs=1e-6)
        assert result_in[2] == pytest.approx(result_after[2], abs=1e-6)

    def test_in_zone_at_center(self):
        """Point at fold center (perp_dist = 0) should be at half arc fraction."""
        fold = FoldDefinition(
            center=(50, 15),
            axis=(0, 1),
            zone_width=10.0,
            angle=math.pi / 2,
        )
        R = fold.radius
        hw = fold.zone_width / 2
        # perp_dist = 0 → dist_into_zone = hw = 5, arc_fraction = 0.5
        # theta = 0.5 * pi/2 = pi/4
        recipe = [(fold, "IN_ZONE", False)]
        result = transform_point((50, 15), recipe)

        theta = math.pi / 4
        expected_up = R * (1 - math.cos(theta))
        assert result[2] == pytest.approx(expected_up, abs=1e-6)

    def test_continuity_in_zone_to_after(self):
        """IN_ZONE at perp_dist=+hw and AFTER at excess=0 should produce the same point."""
        fold = FoldDefinition(
            center=(50, 15),
            axis=(0, 1),
            zone_width=10.0,
            angle=math.pi / 3,
        )
        # perp = (-1, 0). perp_dist = +hw = 5 → x = 50 + 5*(-1) = 45
        result_in = transform_point((45, 15), [(fold, "IN_ZONE", False)])
        result_after = transform_point((45, 15), [(fold, "AFTER", False)])

        assert result_in[0] == pytest.approx(result_after[0], abs=1e-6)
        assert result_in[1] == pytest.approx(result_after[1], abs=1e-6)
        assert result_in[2] == pytest.approx(result_after[2], abs=1e-6)

    def test_continuity_in_zone_to_after_negative_angle(self):
        """IN_ZONE/AFTER continuity also holds for negative angle."""
        fold = FoldDefinition(
            center=(50, 15),
            axis=(0, 1),
            zone_width=10.0,
            angle=-math.pi / 4,
        )
        # perp = (-1, 0). perp_dist = +hw = 5 → x = 45
        result_in = transform_point((45, 15), [(fold, "IN_ZONE", False)])
        result_after = transform_point((45, 15), [(fold, "AFTER", False)])

        assert result_in[0] == pytest.approx(result_after[0], abs=1e-6)
        assert result_in[1] == pytest.approx(result_after[1], abs=1e-6)
        assert result_in[2] == pytest.approx(result_after[2], abs=1e-6)


class TestMultiFoldChains:
    """Tests for multi-fold recipe chains."""

    def _make_fold(self, center_x, angle):
        """Helper to create a fold with axis=(0,1) at given x position."""
        return FoldDefinition(
            center=(center_x, 15),
            axis=(0, 1),
            zone_width=10.0,
            angle=angle,
        )

    def test_two_sequential_after_folds(self):
        """Two 90-degree AFTER folds produce non-trivial 3D displacement."""
        fold1 = self._make_fold(50, math.pi / 2)
        fold2 = self._make_fold(70, math.pi / 2)
        # Point well past both folds
        recipe = [
            (fold1, "AFTER", False),
            (fold2, "AFTER", False),
        ]
        result = transform_point((90, 15), recipe)
        # After two 90-degree folds (total 180 degrees), the point should have
        # significant displacement in both z and the perpendicular direction
        assert result[2] != pytest.approx(0.0, abs=0.1)

    def test_accordion_fold(self):
        """Accordion folds (+90, -90, +90) should partially cancel."""
        fold1 = self._make_fold(30, math.pi / 2)
        fold2 = self._make_fold(50, -math.pi / 2)
        fold3 = self._make_fold(70, math.pi / 2)
        recipe = [
            (fold1, "AFTER", False),
            (fold2, "AFTER", False),
            (fold3, "AFTER", False),
        ]
        result = transform_point((90, 15), recipe)
        # The accordion should produce a finite result without crashing
        # Net rotation is +90 (90 - 90 + 90 = 90), so z should be non-zero
        assert math.isfinite(result[0])
        assert math.isfinite(result[1])
        assert math.isfinite(result[2])
        assert abs(result[2]) > 0.1

    def test_after_then_in_zone_terminates(self):
        """IN_ZONE is terminal; recipe stops processing at that point."""
        fold1 = self._make_fold(30, math.pi / 2)
        fold2 = self._make_fold(50, math.pi / 2)
        recipe = [
            (fold1, "AFTER", False),
            (fold2, "IN_ZONE", False),
        ]
        # Point at fold2 center: perp_dist = 0 relative to fold2
        result = transform_point((50, 15), recipe)
        # After fold1 AFTER rotation, then fold2 IN_ZONE arc, z should be non-zero
        assert abs(result[2]) > 0.01

    def test_three_90_degree_folds_spiral(self):
        """Three 90-degree folds produce 270 degrees total rotation."""
        fold1 = self._make_fold(30, math.pi / 2)
        fold2 = self._make_fold(50, math.pi / 2)
        fold3 = self._make_fold(70, math.pi / 2)
        recipe = [
            (fold1, "AFTER", False),
            (fold2, "AFTER", False),
            (fold3, "AFTER", False),
        ]
        result = transform_point((90, 15), recipe)
        # After 270 degrees of rotation, the result should be well-defined
        assert math.isfinite(result[0])
        assert math.isfinite(result[1])
        assert math.isfinite(result[2])

    def test_multi_fold_preserves_along_coordinate(self):
        """The along-axis coordinate should be preserved through AFTER folds."""
        fold1 = self._make_fold(50, math.pi / 2)
        fold2 = self._make_fold(70, math.pi / 4)
        # Point with along = 10 (y = 25 since axis=(0,1), center_y=15)
        recipe = [
            (fold1, "AFTER", False),
            (fold2, "AFTER", False),
        ]
        result_y0 = transform_point((90, 15), recipe)
        result_y10 = transform_point((90, 25), recipe)
        # The difference should be purely along the (original) fold axis direction.
        # The axis=(0,1) gets rotated through cumulative transforms, but the
        # along-axis position should only shift the result in the axis direction.
        # Verify the two results differ (along != 0 matters).
        dist = math.sqrt(
            (result_y10[0] - result_y0[0]) ** 2
            + (result_y10[1] - result_y0[1]) ** 2
            + (result_y10[2] - result_y0[2]) ** 2
        )
        assert dist == pytest.approx(10.0, abs=1e-4)


class TestBackEntryEdgeCases:
    """Tests for back-entry edge cases."""

    def test_back_entry_in_zone_at_center(self):
        """Back entry IN_ZONE at fold center should have same |z| as normal entry."""
        fold = FoldDefinition(
            center=(50, 15),
            axis=(0, 1),
            zone_width=10.0,
            angle=math.pi / 2,
        )
        result_front = transform_point((50, 15), [(fold, "IN_ZONE", False)])
        result_back = transform_point((50, 15), [(fold, "IN_ZONE", True)])
        # z magnitude should be the same
        assert abs(result_front[2]) == pytest.approx(abs(result_back[2]), abs=1e-6)

    def test_back_entry_after_z_displacement(self):
        """Back entry AFTER should produce non-zero z displacement."""
        fold = FoldDefinition(
            center=(50, 15),
            axis=(0, 1),
            zone_width=10.0,
            angle=math.pi / 2,
        )
        result = transform_point((40, 15), [(fold, "AFTER", True)])
        assert abs(result[2]) > 0.1

    def test_back_entry_normal_opposite_rotation(self):
        """Back entry AFTER normal should rotate opposite to normal entry."""
        fold = FoldDefinition(
            center=(50, 15),
            axis=(0, 1),
            zone_width=10.0,
            angle=math.pi / 2,
        )
        normal_front = compute_normal((60, 15), [(fold, "AFTER", False)])
        normal_back = compute_normal((40, 15), [(fold, "AFTER", True)])
        # Back entry uses -angle for the rotation, so the normals should
        # be rotated in opposite directions. They should NOT be identical.
        dot = (
            normal_front[0] * normal_back[0]
            + normal_front[1] * normal_back[1]
            + normal_front[2] * normal_back[2]
        )
        # For 90-degree fold, front normal ~ (perp, 0, 0) rotated +90,
        # back normal rotated -90. Dot product should not be 1.
        assert dot != pytest.approx(1.0, abs=0.01)

    def test_normal_flat_region(self):
        """Normal with empty recipe should be (0, 0, 1)."""
        normal = compute_normal((50, 15), [])
        assert normal[0] == pytest.approx(0.0, abs=1e-6)
        assert normal[1] == pytest.approx(0.0, abs=1e-6)
        assert normal[2] == pytest.approx(1.0, abs=1e-6)

    def test_normal_after_90_fold(self):
        """Normal after 90-degree fold around (0,1) should be roughly (+-1, 0, 0)."""
        fold = FoldDefinition(
            center=(50, 15),
            axis=(0, 1),
            zone_width=10.0,
            angle=math.pi / 2,
        )
        normal = compute_normal((60, 15), [(fold, "AFTER", False)])
        # After 90-degree rotation around (0,1,0), (0,0,1) -> (1,0,0) or (-1,0,0)
        # depending on rotation sign convention. The important thing: z should be ~0
        # and one of x or y should dominate.
        assert abs(normal[2]) < 0.1
        assert abs(normal[0]) > 0.9 or abs(normal[1]) > 0.9


class TestRodriguesRotation:
    """Tests for the Rodrigues rotation helper functions."""

    def test_rotation_identity_zero_angle(self):
        """Rotation by 0 around (0,0,1) should be identity."""
        R = _rotation_matrix_around_axis((0, 0, 1), 0)
        identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        for i in range(3):
            for j in range(3):
                assert R[i][j] == pytest.approx(identity[i][j], abs=1e-6)

    def test_rotation_180_around_z(self):
        """180-degree rotation around z should negate x and y, keep z."""
        R = _rotation_matrix_around_axis((0, 0, 1), math.pi)
        v = _apply_rotation(R, (1.0, 2.0, 3.0))
        assert v[0] == pytest.approx(-1.0, abs=1e-6)
        assert v[1] == pytest.approx(-2.0, abs=1e-6)
        assert v[2] == pytest.approx(3.0, abs=1e-6)

    def test_rotation_preserves_length(self):
        """Rotation should preserve vector magnitude."""
        R = _rotation_matrix_around_axis((0, 0, 1), math.pi / 4)
        v_in = (1.0, 2.0, 3.0)
        v_out = _apply_rotation(R, v_in)
        len_in = math.sqrt(sum(c ** 2 for c in v_in))
        len_out = math.sqrt(sum(c ** 2 for c in v_out))
        assert len_out == pytest.approx(len_in, abs=1e-6)

    def test_rotation_orthogonal(self):
        """R^T @ R should be identity (orthogonal matrix)."""
        R = _rotation_matrix_around_axis((1, 1, 0), math.pi / 3)
        # Compute R^T @ R
        RT = [[R[j][i] for j in range(3)] for i in range(3)]
        product = _multiply_matrices(RT, R)
        identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        for i in range(3):
            for j in range(3):
                assert product[i][j] == pytest.approx(identity[i][j], abs=1e-6)

    def test_rotation_zero_axis(self):
        """Zero-length axis should return identity (degenerate case)."""
        R = _rotation_matrix_around_axis((0, 0, 0), math.pi)
        identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        for i in range(3):
            for j in range(3):
                assert R[i][j] == pytest.approx(identity[i][j], abs=1e-6)
