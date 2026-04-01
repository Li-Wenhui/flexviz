"""
Polygon triangulation routines.

Provides ear-clipping triangulation for simple polygons and polygons with holes,
based on Eberly's "Triangulation by Ear Clipping" algorithm.
"""

try:
    from .polygon_ops import (
        signed_area, ensure_ccw, ensure_cw, cross_product_2d,
        is_convex_vertex, is_reflex_vertex_pts, point_in_triangle, point_in_polygon
    )
except ImportError:
    from polygon_ops import (
        signed_area, ensure_ccw, ensure_cw, cross_product_2d,
        is_convex_vertex, is_reflex_vertex_pts, point_in_triangle, point_in_polygon
    )


def find_mutually_visible_vertex(M: tuple[float, float],
                                  polygon: list[tuple[float, float]]) -> int:
    """
    Find a vertex in polygon that is mutually visible with point M.

    Implements the algorithm from Eberly's "Triangulation by Ear Clipping":
    1. Cast ray from M in +X direction
    2. Find closest intersection point I on polygon edge
    3. If I is a vertex, return that vertex
    4. Otherwise, P = endpoint of edge with max x-value
    5. If reflex vertices exist inside triangle <M, I, P>, return the one
       with minimum angle to ray direction
    6. Otherwise return P

    Args:
        M: The point (typically rightmost vertex of a hole)
        polygon: The outer polygon (CCW ordered)

    Returns:
        Index of the mutually visible vertex in polygon
    """
    mx, my = M
    n = len(polygon)

    # Step 1-2: Cast ray M + t*(1,0) and find closest intersection
    min_t = float('inf')
    hit_edge_idx = -1
    intersection_point = None

    for i in range(n):
        j = (i + 1) % n
        vi, vj = polygon[i], polygon[j]

        # Only consider edges where vi is below (or on) ray and vj is above (or on)
        if not ((vi[1] <= my < vj[1]) or (vj[1] <= my < vi[1])):
            continue

        # Skip horizontal edges
        if abs(vj[1] - vi[1]) < 1e-10:
            continue

        # Calculate intersection x
        t_edge = (my - vi[1]) / (vj[1] - vi[1])
        x_int = vi[0] + t_edge * (vj[0] - vi[0])

        # Must be to the right of M
        if x_int <= mx:
            continue

        t = x_int - mx  # Distance along ray

        if t < min_t:
            min_t = t
            hit_edge_idx = i
            intersection_point = (x_int, my)

    if hit_edge_idx == -1:
        # Fallback: find closest vertex to the right
        best_idx = 0
        best_dist = float('inf')
        for i, v in enumerate(polygon):
            if v[0] > mx:
                dist = (v[0] - mx) ** 2 + (v[1] - my) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
        return best_idx

    I = intersection_point
    vi = polygon[hit_edge_idx]
    vj = polygon[(hit_edge_idx + 1) % n]

    # Step 3: Check if I is (very close to) a vertex
    eps = 1e-9
    if abs(I[0] - vi[0]) < eps and abs(I[1] - vi[1]) < eps:
        return hit_edge_idx
    if abs(I[0] - vj[0]) < eps and abs(I[1] - vj[1]) < eps:
        return (hit_edge_idx + 1) % n

    # Step 4: P = endpoint with maximum x-value
    if vi[0] > vj[0]:
        P_idx = hit_edge_idx
    else:
        P_idx = (hit_edge_idx + 1) % n
    P = polygon[P_idx]

    # Step 5: Find reflex vertices inside triangle <M, I, P>
    reflex_in_triangle = []
    for i in range(n):
        if i == P_idx:
            continue
        prev_v = polygon[(i - 1) % n]
        curr_v = polygon[i]
        next_v = polygon[(i + 1) % n]

        if is_reflex_vertex_pts(prev_v, curr_v, next_v):
            if point_in_triangle(curr_v, M, I, P):
                reflex_in_triangle.append(i)

    # Step 6: If no reflex vertices in triangle, P is visible
    if not reflex_in_triangle:
        return P_idx

    # Step 7: Find reflex vertex R that minimizes angle between <M,I> and <M,R>
    best_idx = reflex_in_triangle[0]
    best_angle = float('inf')

    for idx in reflex_in_triangle:
        R = polygon[idx]
        dx = R[0] - mx
        dy = abs(R[1] - my)
        if dx > eps:
            angle = dy / dx  # tan(angle) - smaller is better
            if angle < best_angle:
                best_angle = angle
                best_idx = idx

    return best_idx


def merge_hole_into_polygon(
    outer: list[tuple[float, float]],
    hole: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """
    Merge a hole into the outer polygon to create a pseudosimple polygon.

    From Section 3 of Eberly's paper:
    - Find M = vertex with max x-value in hole
    - Find V = mutually visible vertex in outer
    - Create bridge with edges <V,M> and <M,V>

    The merged polygon format (from equation 2, page 8):
    {outer_before_V, V, hole_from_M, M, V, outer_after_V}

    Args:
        outer: Outer polygon vertices (CCW)
        hole: Hole polygon vertices (CW)

    Returns:
        Merged pseudosimple polygon
    """
    # Find M = vertex with maximum x-value in hole
    m_idx = max(range(len(hole)), key=lambda i: hole[i][0])
    M = hole[m_idx]

    # Find V = mutually visible vertex in outer polygon
    v_idx = find_mutually_visible_vertex(M, outer)

    # Reorder hole to start from M
    hole_from_M = hole[m_idx:] + hole[:m_idx]

    # Build merged polygon:
    # {outer[0..v_idx], hole_from_M, M, V, outer[v_idx+1..end]}
    merged = (
        outer[:v_idx + 1] +      # outer up to and including V
        hole_from_M +             # hole starting from M
        [M] +                     # M again (bridge return)
        [outer[v_idx]] +          # V again (bridge return)
        outer[v_idx + 1:]         # rest of outer
    )

    return merged


def triangulate_with_holes(
    outer: list[tuple[float, float]],
    holes: list[list[tuple[float, float]]]
) -> tuple[list[tuple[int, int, int]], list[tuple[float, float]]]:
    """
    Triangulate a polygon with holes using ear clipping.

    Algorithm from Eberly's "Triangulation by Ear Clipping" Section 5:
    1. Ensure outer is CCW, holes are CW
    2. Sort holes by maximum x-value (rightmost first)
    3. Merge each hole into the outer polygon
    4. Triangulate the resulting pseudosimple polygon

    Args:
        outer: Outer polygon vertices
        holes: List of hole polygons

    Returns:
        Tuple of (triangles, merged_polygon)
        triangles: List of (i, j, k) indices into merged_polygon
    """
    if not holes:
        outer_ccw = ensure_ccw(outer)
        return triangulate_polygon(outer_ccw), outer_ccw

    # Step 1: Ensure correct winding
    outer_ccw = ensure_ccw(outer)
    holes_cw = [ensure_cw(h) for h in holes]

    # Step 2: Sort holes by maximum x-value (process rightmost first)
    hole_data = []
    for hole in holes_cw:
        max_x = max(v[0] for v in hole)
        hole_data.append((max_x, hole))
    hole_data.sort(key=lambda x: x[0], reverse=True)

    # Step 3: Merge each hole
    result = outer_ccw
    for _, hole in hole_data:
        result = merge_hole_into_polygon(result, hole)

    # Step 4: Triangulate
    triangles = triangulate_polygon(result)

    return triangles, result


def find_reflex_vertices(polygon: list[tuple[float, float]]) -> list[int]:
    """Find all reflex vertex indices in a CCW polygon."""
    n = len(polygon)
    reflex = []
    for i in range(n):
        prev_v = polygon[(i - 1) % n]
        curr_v = polygon[i]
        next_v = polygon[(i + 1) % n]
        if is_reflex_vertex_pts(prev_v, curr_v, next_v):
            reflex.append(i)
    return reflex


def triangulate_polygon(vertices: list[tuple[float, float]]) -> list[tuple[int, int, int]]:
    """
    Triangulate a simple polygon using ear clipping.

    Based on Eberly's "Triangulation by Ear Clipping" algorithm.

    Args:
        vertices: List of (x, y) vertices in order

    Returns:
        List of triangles, each as (i, j, k) indices into vertices
    """
    polygon = ensure_ccw(vertices)
    n = len(polygon)

    if n < 3:
        return []
    if n == 3:
        return [(0, 1, 2)]

    # Work with indices into the original polygon
    indices = list(range(n))
    triangles = []

    # Find initial reflex vertices
    reflex_set = set(find_reflex_vertices(polygon))

    max_iterations = n * n
    iteration = 0

    while len(indices) > 3 and iteration < max_iterations:
        iteration += 1
        ear_found = False

        for i in range(len(indices)):
            idx = indices[i]
            prev_idx = indices[(i - 1) % len(indices)]
            next_idx = indices[(i + 1) % len(indices)]

            prev_v = polygon[prev_idx]
            curr_v = polygon[idx]
            next_v = polygon[next_idx]

            # Check if convex
            if not is_convex_vertex(prev_v, curr_v, next_v):
                continue

            # Check no reflex vertex inside triangle
            is_valid_ear = True
            for r_idx in reflex_set:
                if r_idx in (prev_idx, idx, next_idx):
                    continue
                if r_idx not in indices:
                    continue
                if point_in_triangle(polygon[r_idx], prev_v, curr_v, next_v):
                    is_valid_ear = False
                    break

            if is_valid_ear:
                # Found an ear - clip it
                triangles.append((prev_idx, idx, next_idx))
                indices.remove(idx)

                # Update reflex status of adjacent vertices
                if len(indices) >= 3:
                    # Find new positions of prev and next in remaining indices
                    new_prev_pos = indices.index(prev_idx)
                    new_next_pos = indices.index(next_idx)

                    # Check if prev vertex changed from reflex to convex
                    pp = indices[(new_prev_pos - 1) % len(indices)]
                    pn = indices[(new_prev_pos + 1) % len(indices)]
                    if is_reflex_vertex_pts(polygon[pp], polygon[prev_idx], polygon[pn]):
                        reflex_set.add(prev_idx)
                    else:
                        reflex_set.discard(prev_idx)

                    # Check if next vertex changed from reflex to convex
                    np_ = indices[(new_next_pos - 1) % len(indices)]
                    nn = indices[(new_next_pos + 1) % len(indices)]
                    if is_reflex_vertex_pts(polygon[np_], polygon[next_idx], polygon[nn]):
                        reflex_set.add(next_idx)
                    else:
                        reflex_set.discard(next_idx)

                ear_found = True
                break

        if not ear_found:
            # Fallback: just clip any vertex
            if len(indices) > 3:
                i = 0
                prev_idx = indices[(i - 1) % len(indices)]
                idx = indices[i]
                next_idx = indices[(i + 1) % len(indices)]
                triangles.append((prev_idx, idx, next_idx))
                indices.remove(idx)

    # Final triangle
    if len(indices) == 3:
        triangles.append((indices[0], indices[1], indices[2]))

    return triangles
