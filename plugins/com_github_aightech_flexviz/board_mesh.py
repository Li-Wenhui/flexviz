"""
Board outline mesh generation.

Creates 3D meshes for the PCB board outline with fold region splitting,
cutout handling, and bend transformations.
"""

import math

try:
    from .geometry import Polygon, BoardGeometry, subdivide_polygon
    from .bend_transform import FoldDefinition, FoldRecipe, transform_point, transform_point_and_normal, recipe_from_region
    from .markers import FoldMarker
    from .planar_subdivision import split_board_into_regions, Region, find_containing_region
    from .triangulation import triangulate_polygon, triangulate_with_holes
    from .mesh_types import (
        Mesh, COLOR_BOARD, COLOR_CUTOUT, get_region_recipe,
        transform_vertices_with_thickness, get_debug_color, snap_to_plane,
        PrecomputedBoardData, PrecomputedRegionData,
    )
except ImportError:
    from geometry import Polygon, BoardGeometry, subdivide_polygon
    from bend_transform import FoldDefinition, FoldRecipe, transform_point, transform_point_and_normal, recipe_from_region
    from markers import FoldMarker
    from planar_subdivision import split_board_into_regions, Region, find_containing_region
    from triangulation import triangulate_polygon, triangulate_with_holes
    from mesh_types import (
        Mesh, COLOR_BOARD, COLOR_CUTOUT, get_region_recipe,
        transform_vertices_with_thickness, get_debug_color, snap_to_plane,
        PrecomputedBoardData, PrecomputedRegionData,
    )


def precompute_board_mesh(
    outline: Polygon,
    thickness: float,
    markers: list[FoldMarker] = None,
    subdivide_length: float = 1.0,
    cutouts: list[Polygon] = None,
    num_bend_subdivisions: int = 1,
    debug_regions: bool = False,
) -> PrecomputedBoardData:
    """
    Precompute angle-independent board mesh data: region splitting, subdivision,
    triangulation, and 2D vertex positions.

    Args:
        outline: Board outline polygon
        thickness: Board thickness in mm
        markers: List of fold markers (for region splitting)
        subdivide_length: Maximum edge length for subdivision
        cutouts: List of cutout polygons (holes in the board)
        num_bend_subdivisions: Number of strips in bend zone
        debug_regions: If True, color each region differently for debugging

    Returns:
        PrecomputedBoardData with all angle-independent data
    """
    if not outline.vertices:
        return PrecomputedBoardData(region_data=[], thickness=thickness, debug_regions=debug_regions)

    cutouts = cutouts or []

    # Convert outline and cutouts to lists of tuples
    outline_verts = [(v[0], v[1]) for v in outline.vertices]
    cutout_verts = [[(v[0], v[1]) for v in c.vertices] for c in cutouts]

    # Split into regions along fold lines (or single region if no markers)
    if markers:
        regions = split_board_into_regions(outline_verts, cutout_verts, markers, num_bend_subdivisions)
    else:
        single_region = Region(
            index=0,
            outline=outline_verts,
            holes=cutout_verts,
            fold_recipe=[]
        )
        regions = [single_region]

    region_data_list = []
    for region in regions:
        # Choose color based on debug mode
        if debug_regions:
            region_color = get_debug_color(region.index)
        else:
            region_color = COLOR_BOARD

        # Subdivide the region outline
        region_poly = Polygon(region.outline)
        subdivided = subdivide_polygon(region_poly, subdivide_length)
        subdivided_verts = [(v[0], v[1]) for v in subdivided.vertices]

        # Subdivide holes in this region
        region_holes_2d = []
        for hole in region.holes:
            hole_poly = Polygon(hole)
            sub_hole = subdivide_polygon(hole_poly, subdivide_length)
            region_holes_2d.append([(v[0], v[1]) for v in sub_hole.vertices])

        # Triangulate this region (angle-independent)
        has_holes = bool(region_holes_2d)
        if has_holes:
            triangles, merged = triangulate_with_holes(subdivided_verts, region_holes_2d)
            merged_2d = list(merged)
        else:
            triangles = triangulate_polygon(subdivided_verts)
            merged_2d = None

        region_data_list.append(PrecomputedRegionData(
            region=region,
            subdivided_verts=subdivided_verts,
            region_holes_2d=region_holes_2d,
            triangles=triangles,
            merged_2d=merged_2d,
            has_holes=has_holes,
            color=region_color,
        ))

    return PrecomputedBoardData(
        region_data=region_data_list,
        thickness=thickness,
        debug_regions=debug_regions,
    )


def transform_board_mesh(precomputed: PrecomputedBoardData, apply_bend: bool = True) -> Mesh:
    """
    Transform precomputed 2D board vertices to 3D using current fold angles.

    This is the fast path: it skips region splitting, subdivision, and triangulation,
    and only performs the 2D-to-3D transformation using current fold recipes.

    Args:
        precomputed: PrecomputedBoardData from precompute_board_mesh()
        apply_bend: If False, show flat board with regions but no bending

    Returns:
        Mesh representing the board
    """
    mesh = Mesh()
    thickness = precomputed.thickness

    for rd in precomputed.region_data:
        region = rd.region
        region_color = rd.color
        subdivided_verts = rd.subdivided_verts
        region_holes_2d = rd.region_holes_2d

        # Get fold recipe (angle-dependent via marker.angle_degrees)
        region_recipe = get_region_recipe(region) if apply_bend else []

        # Transform vertices to 3D with thickness offset
        top_vertices, bottom_vertices, vertex_normals = transform_vertices_with_thickness(
            subdivided_verts, region_recipe, thickness
        )

        # Snap flat regions to ensure coplanarity
        top_vertices = snap_to_plane(top_vertices, vertex_normals)

        # Bottom vertices: offset along negative normal direction
        bottom_vertices = []
        for v3d, normal in zip(top_vertices, vertex_normals):
            bottom_v = (
                v3d[0] - normal[0] * thickness,
                v3d[1] - normal[1] * thickness,
                v3d[2] - normal[2] * thickness
            )
            bottom_vertices.append(bottom_v)

        # Snap bottom vertices too
        bottom_vertices = snap_to_plane(bottom_vertices, vertex_normals)

        # Add outline vertices to mesh
        n = len(top_vertices)
        top_indices = [mesh.add_vertex(v) for v in top_vertices]
        bottom_indices = [mesh.add_vertex(v) for v in bottom_vertices]

        # Process holes for this region
        hole_top_indices = []
        hole_bottom_indices = []
        hole_2d_lists = []

        for hole_2d in region_holes_2d:
            hole_top_verts = []
            hole_bottom_verts = []

            for v in hole_2d:
                v3d, normal = transform_point_and_normal(v, region_recipe)
                hole_top_verts.append(v3d)
                hole_bottom_verts.append((
                    v3d[0] - normal[0] * thickness,
                    v3d[1] - normal[1] * thickness,
                    v3d[2] - normal[2] * thickness
                ))

            ht_indices = [mesh.add_vertex(v) for v in hole_top_verts]
            hb_indices = [mesh.add_vertex(v) for v in hole_bottom_verts]

            hole_top_indices.append(ht_indices)
            hole_bottom_indices.append(hb_indices)
            hole_2d_lists.append(hole_2d)

        # Use precomputed triangulation
        if rd.has_holes:
            triangles = rd.triangles
            merged_2d = rd.merged_2d

            # Build vertex mapping
            all_2d = list(subdivided_verts)
            all_top = list(top_indices)
            all_bottom = list(bottom_indices)

            for h2d, ht_idx, hb_idx in zip(hole_2d_lists, hole_top_indices, hole_bottom_indices):
                all_2d.extend(h2d)
                all_top.extend(ht_idx)
                all_bottom.extend(hb_idx)

            # Map merged vertices to mesh indices
            merged_top = []
            merged_bottom = []
            for v2d in merged_2d:
                found = False
                for i, av in enumerate(all_2d):
                    if abs(av[0] - v2d[0]) < 0.001 and abs(av[1] - v2d[1]) < 0.001:
                        merged_top.append(all_top[i])
                        merged_bottom.append(all_bottom[i])
                        found = True
                        break
                if not found:
                    v3d, normal = transform_point_and_normal(v2d, region_recipe)
                    ti = mesh.add_vertex(v3d)
                    bi = mesh.add_vertex((
                        v3d[0] - normal[0] * thickness,
                        v3d[1] - normal[1] * thickness,
                        v3d[2] - normal[2] * thickness
                    ))
                    merged_top.append(ti)
                    merged_bottom.append(bi)

            # Add triangles
            for tri in triangles:
                mesh.add_triangle(merged_top[tri[0]], merged_top[tri[1]], merged_top[tri[2]], region_color)
            for tri in triangles:
                mesh.add_triangle(merged_bottom[tri[0]], merged_bottom[tri[2]], merged_bottom[tri[1]], region_color)
        else:
            # No holes in this region — use precomputed triangles
            triangles = rd.triangles
            for tri in triangles:
                mesh.add_triangle(top_indices[tri[0]], top_indices[tri[1]], top_indices[tri[2]], region_color)
            for tri in triangles:
                mesh.add_triangle(bottom_indices[tri[0]], bottom_indices[tri[2]], bottom_indices[tri[1]], region_color)

        # Side faces for region outline
        for i in range(n):
            j = (i + 1) % n
            mesh.add_quad(top_indices[i], top_indices[j], bottom_indices[j], bottom_indices[i], region_color)

        # Side faces for holes
        for ht_idx, hb_idx in zip(hole_top_indices, hole_bottom_indices):
            nh = len(ht_idx)
            for i in range(nh):
                j = (i + 1) % nh
                mesh.add_quad(ht_idx[j], ht_idx[i], hb_idx[i], hb_idx[j], COLOR_CUTOUT)

    return mesh


def create_board_mesh_with_regions(
    outline: Polygon,
    thickness: float,
    markers: list[FoldMarker] = None,
    subdivide_length: float = 1.0,
    cutouts: list[Polygon] = None,
    num_bend_subdivisions: int = 1,
    debug_regions: bool = False,
    apply_bend: bool = True
) -> Mesh:
    """
    Create a 3D mesh for the board, split by fold regions.

    This is a convenience wrapper that calls precompute_board_mesh() then
    transform_board_mesh(). For repeated angle changes, call them separately.

    Args:
        outline: Board outline polygon
        thickness: Board thickness in mm
        markers: List of fold markers (for region splitting and 3D transform)
        subdivide_length: Maximum edge length for subdivision
        cutouts: List of cutout polygons (holes in the board)
        num_bend_subdivisions: Number of strips in bend zone
        debug_regions: If True, color each region differently for debugging
        apply_bend: If False, show flat board with regions but no bending

    Returns:
        Mesh representing the board
    """
    precomputed = precompute_board_mesh(
        outline, thickness, markers, subdivide_length, cutouts,
        num_bend_subdivisions, debug_regions,
    )
    mesh = transform_board_mesh(precomputed, apply_bend)
    return mesh
