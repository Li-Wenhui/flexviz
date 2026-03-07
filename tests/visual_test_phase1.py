"""
Visual validation for Phase 1: trace/pad rendering across fold zones.

Generates images to tests/visual/phase1_traces/ for manual inspection.
Run with: python tests/visual_test_phase1.py

Edge cases covered:
- Horizontal traces crossing folds
- Diagonal traces crossing folds at oblique angles
- Traces starting/ending inside fold zones
- Back-layer traces (B.Cu)
- Pads on fold boundaries
- Through-hole pads with drill holes across folds
- Multiple folds with different angles
- Dense parallel routing
- Real PCB file (with_fold.kicad_pcb)
"""

import sys
import math
from pathlib import Path

# Add plugin directory to path
PLUGIN_DIR = Path(__file__).parent.parent / "plugins" / "com_github_aightech_flexviz"
sys.path.insert(0, str(PLUGIN_DIR))

import pyvista as pv
import numpy as np

from mesh import (
    Mesh, create_trace_mesh, create_pad_mesh, create_board_geometry_mesh
)
from geometry import (
    Polygon, LineSegment, PadGeometry, BoardGeometry, BoundingBox, ComponentGeometry
)
from markers import FoldMarker
from planar_subdivision import split_board_into_regions


OUTPUT_DIR = Path(__file__).parent / "visual" / "phase1_traces"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def make_vertical_fold_marker(x_center, zone_width=5.0, angle_deg=90.0, board_height=30.0):
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


def mesh_to_pyvista(mesh: Mesh):
    """Convert our Mesh to PyVista PolyData."""
    if not mesh.vertices or not mesh.faces:
        return pv.PolyData()

    vertices = np.array(mesh.vertices, dtype=np.float64)
    faces = []
    colors = []
    for i, face in enumerate(mesh.faces):
        faces.append(len(face))
        faces.extend(face)
        if i < len(mesh.colors):
            colors.append(mesh.colors[i])
        else:
            colors.append((50, 150, 50))

    pd = pv.PolyData(vertices, np.array(faces))

    if colors:
        color_array = np.array(colors, dtype=np.uint8)
        pd.cell_data['colors'] = color_array

    return pd


def render_scene(meshes, filename, camera_position=None, title=""):
    """Render multiple meshes to a PNG file."""
    plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
    plotter.set_background('white')

    for mesh_data, kwargs in meshes:
        pd = mesh_to_pyvista(mesh_data)
        if pd.n_points > 0:
            plotter.add_mesh(pd, **kwargs)

    if camera_position:
        plotter.camera_position = camera_position
    else:
        plotter.camera.azimuth = 30
        plotter.camera.elevation = 30
        plotter.reset_camera()

    if title:
        plotter.add_text(title, position='upper_left', font_size=14, color='black')

    filepath = OUTPUT_DIR / filename
    plotter.screenshot(str(filepath))
    plotter.close()
    print(f"  Saved: {filepath}")


# =============================================================================
# Scene 1: Diagonal traces crossing a fold at oblique angles
# =============================================================================
def scene_diagonal_traces():
    print("Scene 1: Diagonal traces crossing fold at various angles")

    outline = Polygon([(0, 0), (120, 0), (120, 40), (0, 40)])
    markers = [make_vertical_fold_marker(60.0, zone_width=5.0, angle_deg=90.0, board_height=40)]

    traces = [
        # Horizontal (0 degrees)
        LineSegment((10, 20), (110, 20), width=0.4),
        # Slight diagonal (~15 degrees)
        LineSegment((10, 8), (110, 32), width=0.4),
        # Steep diagonal (~30 degrees, other direction)
        LineSegment((10, 35), (110, 5), width=0.4),
        # Short trace that starts inside fold zone
        LineSegment((58, 20), (75, 20), width=0.3),
        # Short trace that ends inside fold zone
        LineSegment((45, 10), (61, 10), width=0.3),
    ]

    board = BoardGeometry(
        outline=outline, thickness=1.6,
        traces={'F.Cu': traces}
    )

    mesh = create_board_geometry_mesh(
        board, markers=markers,
        include_traces=True, include_pads=False,
        num_bend_subdivisions=4,
    )

    render_scene(
        [(mesh, {'scalars': 'colors', 'rgb': True, 'show_edges': False})],
        "01_diagonal_traces.png",
        title="Diagonal + horizontal traces crossing 90-deg fold"
    )


# =============================================================================
# Scene 2: Two folds with complex routing
# =============================================================================
def scene_two_folds_complex():
    print("Scene 2: Two folds with complex routing")

    outline = Polygon([(0, 0), (140, 0), (140, 40), (0, 40)])
    markers = [
        make_vertical_fold_marker(45.0, zone_width=5.0, angle_deg=90.0, board_height=40),
        make_vertical_fold_marker(95.0, zone_width=5.0, angle_deg=-60.0, board_height=40),
    ]

    traces = [
        # Full-length horizontal traces at various Y
        LineSegment((5, 8), (135, 8), width=0.3),
        LineSegment((5, 16), (135, 16), width=0.3),
        LineSegment((5, 24), (135, 24), width=0.3),
        LineSegment((5, 32), (135, 32), width=0.3),
        # Diagonal crossing both folds
        LineSegment((10, 5), (130, 35), width=0.5),
        LineSegment((10, 35), (130, 5), width=0.5),
        # Short traces between the two folds
        LineSegment((50, 20), (90, 20), width=0.6),
        # Trace only crossing first fold
        LineSegment((20, 15), (70, 15), width=0.3),
        # Trace only crossing second fold
        LineSegment((70, 25), (130, 25), width=0.3),
    ]

    board = BoardGeometry(
        outline=outline, thickness=1.6,
        traces={'F.Cu': traces}
    )

    mesh = create_board_geometry_mesh(
        board, markers=markers,
        include_traces=True, include_pads=False,
        num_bend_subdivisions=4,
    )

    render_scene(
        [(mesh, {'scalars': 'colors', 'rgb': True, 'show_edges': False})],
        "02_two_folds_complex.png",
        title="Two folds (90, -60) with 9 traces including diagonals"
    )


# =============================================================================
# Scene 3: Pads on and near fold boundaries
# =============================================================================
def scene_pads_on_boundaries():
    print("Scene 3: Pads on and near fold boundaries")

    outline = Polygon([(0, 0), (120, 0), (120, 40), (0, 40)])
    markers = [make_vertical_fold_marker(60.0, zone_width=5.0, angle_deg=90.0, board_height=40)]
    hw = 2.5

    comps = []
    # Pad well before fold
    for x, y, label in [
        (20, 20, "R1"),     # well before fold
        (57.5, 20, "R2"),   # right at BEFORE boundary
        (60, 20, "R3"),     # centered in fold zone
        (62.5, 20, "R4"),   # right at AFTER boundary
        (80, 20, "R5"),     # well after fold (anchor side)
        # Row at different Y
        (20, 10, "R6"),
        (57.5, 10, "R7"),
        (62.5, 10, "R8"),
        (80, 10, "R9"),
        # Through-hole pads (with drill)
        (20, 30, "TH1"),
        (60, 30, "TH2"),    # TH pad in fold zone
        (80, 30, "TH3"),
    ]:
        is_th = label.startswith("TH")
        sz = (2.5, 2.5) if is_th else (2.0, 1.0)
        shp = 'circle' if is_th else 'rect'
        drill = 1.0 if is_th else 0.0
        pad = PadGeometry(center=(x, y), shape=shp, size=sz, drill=drill)
        comp = ComponentGeometry(
            reference=label, value="10k",
            center=(x, y), angle=0,
            bounding_box=BoundingBox(x-1.5, y-1.5, x+1.5, y+1.5),
            pads=[pad], layer="F.Cu"
        )
        comps.append(comp)

    # Traces connecting the pads
    traces = [
        LineSegment((20, 20), (80, 20), width=0.3),
        LineSegment((20, 10), (80, 10), width=0.3),
        LineSegment((20, 30), (80, 30), width=0.3),
    ]

    board = BoardGeometry(
        outline=outline, thickness=1.6,
        traces={'F.Cu': traces},
        components=comps
    )

    mesh = create_board_geometry_mesh(
        board, markers=markers,
        include_traces=True, include_pads=True,
        num_bend_subdivisions=4,
    )

    render_scene(
        [(mesh, {'scalars': 'colors', 'rgb': True, 'show_edges': False})],
        "03_pads_on_boundaries.png",
        title="Pads at fold boundaries + TH pads + traces"
    )


# =============================================================================
# Scene 4: Front and back layer traces
# =============================================================================
def scene_front_back_traces():
    print("Scene 4: Front and back layer traces")

    outline = Polygon([(0, 0), (120, 0), (120, 30), (0, 30)])
    markers = [make_vertical_fold_marker(60.0, zone_width=5.0, angle_deg=90.0)]

    board = BoardGeometry(
        outline=outline, thickness=1.6,
        traces={
            'F.Cu': [
                LineSegment((10, 10), (110, 10), width=0.5),
                LineSegment((10, 20), (110, 20), width=0.5),
            ],
            'B.Cu': [
                LineSegment((10, 15), (110, 15), width=0.5, layer='B.Cu'),
                LineSegment((10, 25), (110, 25), width=0.5, layer='B.Cu'),
            ],
        }
    )

    mesh = create_board_geometry_mesh(
        board, markers=markers,
        include_traces=True, include_pads=False,
        num_bend_subdivisions=4,
    )

    render_scene(
        [(mesh, {'scalars': 'colors', 'rgb': True, 'show_edges': False})],
        "04_front_back_traces.png",
        title="Front (y=10,20) + Back (y=15,25) layer traces"
    )


# =============================================================================
# Scene 5: Dense parallel routing (stress test)
# =============================================================================
def scene_dense_routing():
    print("Scene 5: Dense parallel routing (20 traces)")

    outline = Polygon([(0, 0), (120, 0), (120, 40), (0, 40)])
    markers = [make_vertical_fold_marker(60.0, zone_width=5.0, angle_deg=90.0, board_height=40)]

    traces = []
    for y_idx in range(20):
        y = 2 + y_idx * 1.8
        traces.append(LineSegment((5, y), (115, y), width=0.25))

    board = BoardGeometry(
        outline=outline, thickness=1.6,
        traces={'F.Cu': traces}
    )

    mesh = create_board_geometry_mesh(
        board, markers=markers,
        include_traces=True, include_pads=False,
        num_bend_subdivisions=4,
    )

    render_scene(
        [(mesh, {'scalars': 'colors', 'rgb': True, 'show_edges': False})],
        "05_dense_routing.png",
        title="20 dense parallel traces across 90-deg fold"
    )


# =============================================================================
# Scene 6: Three folds (accordion) with routing
# =============================================================================
def scene_three_folds_accordion():
    print("Scene 6: Three folds (accordion) with traces")

    outline = Polygon([(0, 0), (160, 0), (160, 30), (0, 30)])
    markers = [
        make_vertical_fold_marker(40.0, zone_width=4.0, angle_deg=120.0),
        make_vertical_fold_marker(80.0, zone_width=4.0, angle_deg=-120.0),
        make_vertical_fold_marker(120.0, zone_width=4.0, angle_deg=120.0),
    ]

    traces = [
        LineSegment((5, 8), (155, 8), width=0.4),
        LineSegment((5, 15), (155, 15), width=0.5),
        LineSegment((5, 22), (155, 22), width=0.4),
        # Diagonal
        LineSegment((5, 5), (155, 25), width=0.3),
    ]

    board = BoardGeometry(
        outline=outline, thickness=1.6,
        traces={'F.Cu': traces}
    )

    mesh = create_board_geometry_mesh(
        board, markers=markers,
        include_traces=True, include_pads=False,
        num_bend_subdivisions=4,
    )

    render_scene(
        [(mesh, {'scalars': 'colors', 'rgb': True, 'show_edges': False})],
        "06_three_folds_accordion.png",
        title="3 alternating folds (120, -120, 120) with 4 traces"
    )


# =============================================================================
# Scene 7: Transparent board to verify trace follows surface
# =============================================================================
def scene_transparent_verify():
    print("Scene 7: Transparent board verifying trace surface attachment")

    outline = Polygon([(0, 0), (120, 0), (120, 30), (0, 30)])
    markers = [make_vertical_fold_marker(60.0, zone_width=5.0, angle_deg=90.0)]

    board = BoardGeometry(outline=outline, thickness=1.6)

    board_mesh = create_board_geometry_mesh(
        board, markers=markers,
        include_traces=False, include_pads=False,
        num_bend_subdivisions=4,
    )

    outline_verts = [(v[0], v[1]) for v in board.outline.vertices]
    regions = split_board_into_regions(outline_verts, [], markers, num_bend_subdivisions=4)

    num_subs = 4

    # Multiple traces
    trace_meshes = Mesh()
    for y in [8, 15, 22]:
        tm = create_trace_mesh(
            LineSegment((10, y), (110, y), width=0.8),
            z_offset=0.15, regions=regions, subdivisions=40,
            pcb_thickness=board.thickness, markers=markers,
            num_bend_subdivisions=num_subs
        )
        trace_meshes.merge(tm)

    # Diagonal
    tm = create_trace_mesh(
        LineSegment((10, 5), (110, 25), width=0.6),
        z_offset=0.15, regions=regions, subdivisions=40,
        pcb_thickness=board.thickness, markers=markers,
        num_bend_subdivisions=num_subs
    )
    trace_meshes.merge(tm)

    render_scene(
        [
            (board_mesh, {'color': 'green', 'opacity': 0.3, 'show_edges': True}),
            (trace_meshes, {'color': 'orange', 'show_edges': True}),
        ],
        "07_transparent_verify.png",
        title="Transparent board: traces hugging surface across fold"
    )


# =============================================================================
# Scene 8: Real PCB file (with_fold.kicad_pcb)
# =============================================================================
def scene_real_pcb():
    print("Scene 8: Real PCB (with_fold.kicad_pcb)")

    test_data = Path(__file__).parent / "test_data" / "with_fold.kicad_pcb"
    if not test_data.exists():
        print("  SKIPPED: with_fold.kicad_pcb not found")
        return

    from kicad_parser import KiCadPCB, load_kicad_pcb
    from geometry import extract_geometry
    from markers import detect_fold_markers

    pcb = KiCadPCB(load_kicad_pcb(str(test_data)))
    board = extract_geometry(pcb)
    fold_markers = detect_fold_markers(pcb)

    if not fold_markers:
        print("  SKIPPED: no fold markers detected")
        return

    mesh = create_board_geometry_mesh(
        board, markers=fold_markers,
        include_traces=True, include_pads=True,
    )

    render_scene(
        [(mesh, {'scalars': 'colors', 'rgb': True, 'show_edges': False})],
        "08_real_pcb_with_fold.png",
        title=f"with_fold.kicad_pcb: {len(fold_markers)} folds, traces + pads"
    )


# =============================================================================
# Scene 9: H-shape PCB with 10 folds, traces, and pads
# =============================================================================
def scene_h_shape_pcb():
    print("Scene 9: H-shape PCB (10 folds, traces + pads)")

    test_data = Path(__file__).parent / "test_data" / "h_shape.kicad_pcb"
    if not test_data.exists():
        print("  SKIPPED: h_shape.kicad_pcb not found")
        return

    from kicad_parser import KiCadPCB, load_kicad_pcb
    from geometry import extract_geometry
    from markers import detect_fold_markers

    pcb = KiCadPCB(load_kicad_pcb(str(test_data)))
    board = extract_geometry(pcb)
    fold_markers = detect_fold_markers(pcb)

    print(f"  Detected {len(fold_markers)} folds, {sum(len(t) for t in board.traces.values())} traces, {len(board.all_pads)} pads")

    mesh = create_board_geometry_mesh(
        board, markers=fold_markers,
        include_traces=True, include_pads=True,
        num_bend_subdivisions=4,
    )

    render_scene(
        [(mesh, {'scalars': 'colors', 'rgb': True, 'show_edges': False})],
        "09_h_shape_pcb.png",
        title=f"H-shape PCB: {len(fold_markers)} folds, traces + pads (bend_subs=4)"
    )


# =============================================================================
# Scene 10: H-shape transparent with trace surface alignment
# =============================================================================
def scene_h_shape_transparent():
    print("Scene 10: H-shape transparent (verify trace surface alignment)")

    test_data = Path(__file__).parent / "test_data" / "h_shape.kicad_pcb"
    if not test_data.exists():
        print("  SKIPPED: h_shape.kicad_pcb not found")
        return

    from kicad_parser import KiCadPCB, load_kicad_pcb
    from geometry import extract_geometry
    from markers import detect_fold_markers

    pcb = KiCadPCB(load_kicad_pcb(str(test_data)))
    board = extract_geometry(pcb)
    fold_markers = detect_fold_markers(pcb)

    # Board only (no traces/pads)
    board_mesh = create_board_geometry_mesh(
        board, markers=fold_markers,
        include_traces=False, include_pads=False,
        num_bend_subdivisions=4,
    )

    # Traces + pads only
    traces_board = BoardGeometry(
        outline=board.outline, thickness=board.thickness,
        traces=board.traces, components=board.components,
        cutouts=board.cutouts,
    )
    traces_mesh = create_board_geometry_mesh(
        traces_board, markers=fold_markers,
        include_traces=True, include_pads=True,
        num_bend_subdivisions=4,
    )
    # Remove board verts from traces_mesh (we just want to overlay traces)
    # Easier: generate board separately and overlay

    render_scene(
        [
            (board_mesh, {'color': 'green', 'opacity': 0.3, 'show_edges': True}),
            (traces_mesh, {'scalars': 'colors', 'rgb': True, 'show_edges': True, 'opacity': 0.9}),
        ],
        "10_h_shape_transparent.png",
        title="H-shape: transparent board with traces (verify surface alignment)"
    )


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    scene_diagonal_traces()
    scene_two_folds_complex()
    scene_pads_on_boundaries()
    scene_front_back_traces()
    scene_dense_routing()
    scene_three_folds_accordion()
    scene_transparent_verify()
    scene_real_pcb()
    scene_h_shape_pcb()
    scene_h_shape_transparent()
    print()
    print(f"All images saved to: {OUTPUT_DIR}")
