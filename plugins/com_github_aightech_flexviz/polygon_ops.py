"""
Shared polygon utility functions.

Consolidated from mesh.py and planar_subdivision.py to eliminate duplication.
"""

import math


# Type alias matching planar_subdivision convention
Point = tuple[float, float]
Polygon = list[Point]


def signed_area(polygon: Polygon) -> float:
    """
    Calculate signed area of polygon using shoelace formula.

    Returns:
        Positive for CCW winding, negative for CW winding.
    """
    n = len(polygon)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    return area / 2.0


def ensure_ccw(polygon: Polygon) -> Polygon:
    """Ensure polygon has counter-clockwise winding order."""
    if signed_area(polygon) < 0:
        return list(reversed(polygon))
    return list(polygon)


def ensure_cw(polygon: Polygon) -> Polygon:
    """Ensure polygon has clockwise winding order."""
    if signed_area(polygon) > 0:
        return list(reversed(polygon))
    return list(polygon)


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """
    Check if point is inside polygon using ray casting algorithm.

    Args:
        point: (x, y) coordinates
        polygon: List of (x, y) vertices

    Returns:
        True if point is inside polygon.
    """
    x, y = point
    n = len(polygon)
    inside = False

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i

    return inside


def cross_product_2d(o: Point, a: Point, b: Point) -> float:
    """
    Cross product of vectors OA and OB.
    Positive = B is to the left of OA (CCW turn)
    Negative = B is to the right of OA (CW turn)
    """
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def is_convex_vertex(prev_v: Point, curr_v: Point, next_v: Point) -> bool:
    """
    Check if curr_v is a convex vertex (interior angle < 180 degrees).
    For CCW polygon, convex means cross product > 0.
    """
    return cross_product_2d(prev_v, curr_v, next_v) > 0


def is_reflex_vertex_pts(prev_v: Point, curr_v: Point, next_v: Point) -> bool:
    """
    Check if curr_v is a reflex vertex (interior angle > 180 degrees).
    For CCW polygon, reflex means cross product < 0.
    """
    return cross_product_2d(prev_v, curr_v, next_v) < 0


def point_in_triangle(p: Point, a: Point, b: Point, c: Point) -> bool:
    """Check if point p is inside triangle abc."""
    d1 = cross_product_2d(a, b, p)
    d2 = cross_product_2d(b, c, p)
    d3 = cross_product_2d(c, a, p)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)


def points_equal(p1: Point, p2: Point, eps: float = 1e-9) -> bool:
    """Check if two points are equal within tolerance."""
    return abs(p1[0] - p2[0]) < eps and abs(p1[1] - p2[1]) < eps
