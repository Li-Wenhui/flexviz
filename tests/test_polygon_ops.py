"""Tests for polygon_ops module — shared polygon utilities."""

import pytest
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "plugins" / "com_github_aightech_flexviz"))

from polygon_ops import (
    signed_area, ensure_ccw, ensure_cw, point_in_polygon,
    cross_product_2d, is_convex_vertex, is_reflex_vertex_pts,
    point_in_triangle, points_equal,
)


class TestSignedArea:
    def test_ccw_square(self):
        square = [(0, 0), (1, 0), (1, 1), (0, 1)]
        assert signed_area(square) == pytest.approx(1.0)

    def test_cw_square(self):
        square = [(0, 0), (0, 1), (1, 1), (1, 0)]
        assert signed_area(square) == pytest.approx(-1.0)

    def test_empty_polygon(self):
        assert signed_area([]) == 0.0

    def test_two_points(self):
        assert signed_area([(0, 0), (1, 1)]) == 0.0

    def test_triangle(self):
        tri = [(0, 0), (4, 0), (0, 3)]
        assert signed_area(tri) == pytest.approx(6.0)


class TestEnsureWinding:
    def test_ensure_ccw_already_ccw(self):
        ccw = [(0, 0), (1, 0), (1, 1), (0, 1)]
        result = ensure_ccw(ccw)
        assert signed_area(result) > 0

    def test_ensure_ccw_from_cw(self):
        cw = [(0, 0), (0, 1), (1, 1), (1, 0)]
        result = ensure_ccw(cw)
        assert signed_area(result) > 0

    def test_ensure_cw_already_cw(self):
        cw = [(0, 0), (0, 1), (1, 1), (1, 0)]
        result = ensure_cw(cw)
        assert signed_area(result) < 0

    def test_ensure_cw_from_ccw(self):
        ccw = [(0, 0), (1, 0), (1, 1), (0, 1)]
        result = ensure_cw(ccw)
        assert signed_area(result) < 0


class TestPointInPolygon:
    def test_inside_square(self):
        square = [(0, 0), (10, 0), (10, 10), (0, 10)]
        assert point_in_polygon((5, 5), square) is True

    def test_outside_square(self):
        square = [(0, 0), (10, 0), (10, 10), (0, 10)]
        assert point_in_polygon((15, 5), square) is False

    def test_inside_triangle(self):
        tri = [(0, 0), (10, 0), (5, 10)]
        assert point_in_polygon((5, 3), tri) is True

    def test_outside_triangle(self):
        tri = [(0, 0), (10, 0), (5, 10)]
        assert point_in_polygon((0, 10), tri) is False


class TestCrossProduct:
    def test_ccw_turn(self):
        assert cross_product_2d((0, 0), (1, 0), (0, 1)) > 0

    def test_cw_turn(self):
        assert cross_product_2d((0, 0), (1, 0), (0, -1)) < 0

    def test_collinear(self):
        assert cross_product_2d((0, 0), (1, 0), (2, 0)) == pytest.approx(0)


class TestConvexReflex:
    def test_convex_vertex(self):
        assert is_convex_vertex((0, 0), (1, 0), (1, 1)) is True

    def test_reflex_vertex(self):
        assert is_reflex_vertex_pts((0, 0), (1, 0), (1, -1)) is True


class TestPointInTriangle:
    def test_inside(self):
        assert point_in_triangle((1, 1), (0, 0), (3, 0), (0, 3)) is True

    def test_outside(self):
        assert point_in_triangle((3, 3), (0, 0), (3, 0), (0, 3)) is False


class TestPointsEqual:
    def test_equal(self):
        assert points_equal((1.0, 2.0), (1.0, 2.0)) is True

    def test_near_equal(self):
        assert points_equal((1.0, 2.0), (1.0 + 1e-10, 2.0)) is True

    def test_not_equal(self):
        assert points_equal((1.0, 2.0), (1.1, 2.0)) is False
