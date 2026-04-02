#!/usr/bin/env python3
"""
01_board_import.py -- Board import documentation figures.

Loads the h_shape test board and generates annotated figures showing:
  1. Board outline polygon
  2. KiCad layer stack (conceptual diagram)
  3. Traces and pads on the flat board
  4. Anatomy of a single fold marker
  5. All 8 fold markers overlaid on the board
  6. Key fold-marker formulas

Usage:
    python docs/math/01_board_import.py
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup -- allow importing plugin modules without installation
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "plugins" / "com_github_aightech_flexviz"))

import math
import matplotlib
matplotlib.use("Agg")  # headless rendering -- no display required
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Circle
import numpy as np

from kicad_parser import KiCadPCB
from geometry import extract_geometry
from markers import detect_fold_markers

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
FIGURES_DIR = ROOT / "docs" / "math" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300

# ---------------------------------------------------------------------------
# Load the h_shape board
# ---------------------------------------------------------------------------
PCB_PATH = ROOT / "tests" / "test_data" / "h_shape.kicad_pcb"
pcb = KiCadPCB.load(str(PCB_PATH))
geo = extract_geometry(pcb)
markers = detect_fold_markers(pcb, "User.1")

print(f"Loaded board: {PCB_PATH.name}")
print(f"  Outline vertices : {len(geo.outline.vertices)}")
print(f"  Thickness        : {geo.thickness} mm")
print(f"  Trace layers     : {list(geo.traces.keys())}")
print(f"  Components       : {len(geo.components)}")
print(f"  Pads (total)     : {len(geo.all_pads)}")
print(f"  Cutouts          : {len(geo.cutouts)}")
print(f"  Fold markers     : {len(markers)}")


# ===================================================================
# Helper: draw the board outline as a filled polygon
# ===================================================================
def draw_board_outline(ax, *, fill_color="#e8f0e8", edge_color="#336633",
                       linewidth=1.5, alpha=0.6, label=None):
    """Draw the board outline polygon on *ax*, filled with cutout holes."""
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import PathPatch

    verts = geo.outline.vertices
    if not verts:
        return

    # Build compound path: outer boundary + cutout holes
    def ring_codes(n):
        return [MplPath.MOVETO] + [MplPath.LINETO] * (n - 1) + [MplPath.CLOSEPOLY]

    path_verts = list(verts) + [verts[0]]
    path_codes = ring_codes(len(verts))

    for cutout in geo.cutouts:
        cverts = cutout.vertices
        path_verts += list(cverts) + [cverts[0]]
        path_codes += ring_codes(len(cverts))

    path = MplPath(path_verts, path_codes)
    patch = PathPatch(path, facecolor=fill_color, edgecolor=edge_color,
                      linewidth=linewidth, alpha=alpha, zorder=0, label=label)
    ax.add_patch(patch)

    # Also draw cutout edges explicitly for visibility
    for cutout in geo.cutouts:
        cverts = cutout.vertices
        cxs = [v[0] for v in cverts] + [cverts[0][0]]
        cys = [v[1] for v in cverts] + [cverts[0][1]]
        ax.plot(cxs, cys, color=edge_color, linewidth=linewidth, zorder=1)


# ===================================================================
# Figure 1: Board outline
# ===================================================================
def fig_board_outline():
    """Plot the h_shape board outline polygon with vertex labels."""
    fig, ax = plt.subplots(figsize=(8, 8))

    draw_board_outline(ax, fill_color="#d9eed9", edge_color="#226622",
                       linewidth=2.0)

    verts = geo.outline.vertices
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]

    # Plot vertex dots
    ax.scatter(xs, ys, s=18, color="#226622", zorder=3)

    # Plot cutout vertex dots
    for cutout in geo.cutouts:
        cverts = cutout.vertices
        cxs = [v[0] for v in cverts]
        cys = [v[1] for v in cverts]
        ax.scatter(cxs, cys, s=18, color="#993333", zorder=3)

    # Label a selection of vertices (corners of the H shape)
    # Pick every 4th vertex so labels stay readable
    step = max(1, len(verts) // 8)
    for i in range(0, len(verts), step):
        x, y = verts[i]
        ax.annotate(
            f"({x:.1f}, {y:.1f})",
            xy=(x, y),
            xytext=(8, -8),
            textcoords="offset points",
            fontsize=6,
            color="#333333",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="gray",
                      alpha=0.8),
            zorder=4,
        )

    ax.set_title("H-Shape Board Outline (with cutouts)", fontsize=14, fontweight="bold")
    ax.set_xlabel("x  (mm)")
    ax.set_ylabel("y  (mm)")
    ax.set_aspect("equal")
    ax.invert_yaxis()  # KiCad uses Y-down
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.tight_layout()

    out = FIGURES_DIR / "01_board_outline.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"\nFigure 1 saved: {out}")
    print("  Shows the board outline polygon with vertex coordinates and cutout holes.")
    print(f"  {len(verts)} vertices define the H shape (including arc approximations).")
    print(f"  {len(geo.cutouts)} cutout(s) shown with red vertex dots.")


# ===================================================================
# Figure 2: Layer stack (conceptual diagram)
# ===================================================================
def fig_layer_stack():
    """Conceptual diagram of the KiCad layer stack relevant to flex viewing."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Layer definitions: (label, color, role_text)
    layers = [
        ("User.1\n(fold markers)", "#ffd966", "Fold marker lines + dimension angles"),
        ("F.Cu\n(front copper)",   "#cc4444", "Front copper traces and pads"),
        ("Substrate\n(board)",     "#c2d9a8", "FR4 / polyimide base  (thickness = "
                                              f"{geo.thickness} mm)"),
        ("B.Cu\n(back copper)",    "#4488cc", "Back copper traces and pads"),
        ("Edge.Cuts\n(outline)",   "#226622", "Board outline polygon"),
    ]

    y0 = 0.0
    layer_h = 0.8
    gap = 0.25
    rect_x = 1.5
    rect_w = 4.0

    for i, (label, color, role) in enumerate(layers):
        y = y0 + i * (layer_h + gap)
        rect = Rectangle((rect_x, y), rect_w, layer_h,
                          facecolor=color, edgecolor="#333333",
                          linewidth=1.5, alpha=0.85, zorder=2)
        ax.add_patch(rect)

        # Layer name on the left
        ax.text(rect_x - 0.15, y + layer_h / 2, label,
                ha="right", va="center", fontsize=9, fontweight="bold",
                color="#222222")

        # Role description on the right
        ax.text(rect_x + rect_w + 0.15, y + layer_h / 2, role,
                ha="left", va="center", fontsize=8, color="#444444")

    # Draw arrows showing data flow into pipeline
    total_h = len(layers) * (layer_h + gap) - gap
    arrow_x = rect_x + rect_w + 4.5
    ax.annotate(
        "Pipeline",
        xy=(arrow_x, y0 + total_h),
        xytext=(arrow_x, y0),
        fontsize=10, ha="center", va="bottom",
        arrowprops=dict(arrowstyle="->", lw=1.8, color="#555555"),
        color="#555555", fontweight="bold",
    )
    ax.text(arrow_x, y0 + total_h / 2,
            "   .kicad_pcb\n   parser\n   extracts\n   all layers",
            fontsize=7, color="#555555", va="center")

    ax.set_xlim(-2.5, arrow_x + 2)
    ax.set_ylim(-0.5, y0 + total_h + 1.0)
    ax.invert_yaxis()
    ax.set_title("KiCad Layer Stack (Flex Viewer Perspective)",
                 fontsize=13, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()

    out = FIGURES_DIR / "01_layer_stack.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"\nFigure 2 saved: {out}")
    print("  Conceptual diagram of the KiCad layers used by the flex viewer.")


# ===================================================================
# Figure 3: Traces and pads
# ===================================================================
def fig_traces_and_pads():
    """Plot copper traces and pads on the flat board outline."""
    fig, ax = plt.subplots(figsize=(10, 9))

    # Board outline (light background)
    draw_board_outline(ax, fill_color="#f0f0f0", edge_color="#aaaaaa",
                       linewidth=1.0)

    layer_colors = {
        "F.Cu": "#cc3333",
        "B.Cu": "#3366cc",
    }

    # --- Traces ---
    for layer_name, traces in geo.traces.items():
        color = layer_colors.get(layer_name, "#888888")
        for tr in traces:
            ax.plot(
                [tr.start[0], tr.end[0]],
                [tr.start[1], tr.end[1]],
                color=color,
                linewidth=max(0.8, tr.width * 2),  # exaggerate for visibility
                solid_capstyle="round",
                zorder=2,
            )

    # Proxy artists for the legend
    for layer_name, color in layer_colors.items():
        if layer_name in geo.traces:
            ax.plot([], [], color=color, linewidth=2, label=f"Traces ({layer_name})")

    # --- Pads ---
    smd_pads = [p for p in geo.all_pads if p.drill == 0]
    th_pads = [p for p in geo.all_pads if p.drill > 0]

    # SMD pads as small filled markers
    if smd_pads:
        ax.scatter(
            [p.center[0] for p in smd_pads],
            [p.center[1] for p in smd_pads],
            s=12, color="#22aa44", zorder=3, label="SMD pads",
        )

    # Through-hole pads: filled dot + open circle for drill
    if th_pads:
        ax.scatter(
            [p.center[0] for p in th_pads],
            [p.center[1] for p in th_pads],
            s=30, color="#22aa44", zorder=3, label="TH pads",
        )
        for p in th_pads:
            drill_circle = Circle(
                p.center, p.drill / 2,
                fill=False, edgecolor="#444444", linewidth=0.6, zorder=4,
            )
            ax.add_patch(drill_circle)
        # Proxy for legend
        ax.plot([], [], "o", mfc="none", mec="#444444", ms=5,
                label="Drill holes")

    ax.set_title("Traces and Pads on H-Shape Board", fontsize=14,
                 fontweight="bold")
    ax.set_xlabel("x  (mm)")
    ax.set_ylabel("y  (mm)")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()

    out = FIGURES_DIR / "01_traces_and_pads.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"\nFigure 3 saved: {out}")
    print(f"  {sum(len(t) for t in geo.traces.values())} traces across "
          f"{len(geo.traces)} layers.")
    print(f"  {len(smd_pads)} SMD pads, {len(th_pads)} through-hole pads.")


# ===================================================================
# Figure 4: Fold marker anatomy (zoom into one marker)
# ===================================================================
def fig_fold_marker_anatomy():
    """Zoom into one fold marker and annotate all its parameters."""
    # Pick a marker with a clear, interesting angle.
    # Use the 45-degree marker at index 0 (horizontal axis).
    m = markers[0]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Compute perpendicular direction
    perp = (-m.axis[1], m.axis[0])
    cx, cy = m.center

    # Arrow length -- proportional to zone width but big enough to read
    arrow_len = max(m.zone_width * 1.8, 3.0)

    # ------------------------------------------------------------------
    # Set explicit view limits so arrows + annotations are fully visible.
    # Collect all points that must be on screen, then pad generously.
    # ------------------------------------------------------------------
    all_pts_x = [
        m.line_a_start[0], m.line_a_end[0],
        m.line_b_start[0], m.line_b_end[0],
        cx, cx + m.axis[0] * arrow_len * 1.3,
        cx + perp[0] * arrow_len * 1.3,
    ]
    all_pts_y = [
        m.line_a_start[1], m.line_a_end[1],
        m.line_b_start[1], m.line_b_end[1],
        cy, cy + m.axis[1] * arrow_len * 1.3,
        cy + perp[1] * arrow_len * 1.3,
    ]
    pad = 3.0  # mm padding around all elements
    x_lo, x_hi = min(all_pts_x) - pad, max(all_pts_x) + pad + 5  # extra for labels
    y_lo, y_hi = min(all_pts_y) - pad, max(all_pts_y) + pad

    # Draw the two parallel boundary lines (dashed)
    ax.plot(
        [m.line_a_start[0], m.line_a_end[0]],
        [m.line_a_start[1], m.line_a_end[1]],
        color="#cc4444", linewidth=2.5, linestyle="--", zorder=3,
        label="Line A (zone boundary)",
    )
    ax.plot(
        [m.line_b_start[0], m.line_b_end[0]],
        [m.line_b_start[1], m.line_b_end[1]],
        color="#3366cc", linewidth=2.5, linestyle="--", zorder=3,
        label="Line B (zone boundary)",
    )

    # Shade the zone between the two lines
    zone_poly_x = [
        m.line_a_start[0], m.line_a_end[0],
        m.line_b_end[0], m.line_b_start[0], m.line_a_start[0],
    ]
    zone_poly_y = [
        m.line_a_start[1], m.line_a_end[1],
        m.line_b_end[1], m.line_b_start[1], m.line_a_start[1],
    ]
    ax.fill(zone_poly_x, zone_poly_y, color="#ffffcc", alpha=0.5, zorder=1,
            label="Bend zone")

    # Center dot
    ax.plot(cx, cy, "ko", ms=7, zorder=5)
    ax.annotate(
        f"center\n({cx:.2f}, {cy:.2f})",
        xy=(cx, cy), xytext=(20, -30), textcoords="offset points",
        fontsize=9, ha="left",
        arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.9),
        zorder=6,
    )

    # Axis arrow (along the fold lines)
    ax.annotate(
        "", xy=(cx + m.axis[0] * arrow_len, cy + m.axis[1] * arrow_len),
        xytext=(cx, cy),
        arrowprops=dict(arrowstyle="-|>", color="#009900", lw=2.5),
        zorder=5,
    )
    ax.text(cx + m.axis[0] * arrow_len * 1.1,
            cy + m.axis[1] * arrow_len * 1.1,
            r"$\vec{axis}$", fontsize=13, color="#009900",
            fontweight="bold", ha="left", va="center", zorder=6)

    # Perpendicular arrow
    ax.annotate(
        "", xy=(cx + perp[0] * arrow_len, cy + perp[1] * arrow_len),
        xytext=(cx, cy),
        arrowprops=dict(arrowstyle="-|>", color="#cc6600", lw=2.5),
        zorder=5,
    )
    ax.text(cx + perp[0] * arrow_len * 1.1,
            cy + perp[1] * arrow_len * 1.1,
            r"$\vec{perp}$", fontsize=13, color="#cc6600",
            fontweight="bold", ha="center", va="bottom", zorder=6)

    # Zone width annotation (double-headed arrow between lines)
    # Place it at the right end of the boundary lines
    # Use perpendicular projection to connect corresponding points on the lines
    right_a = m.line_a_end  # right endpoint of line A
    right_b = m.line_b_end  # right endpoint of line B
    offset_along_axis = 1.5  # shift the arrow slightly right of line ends
    wa_x = right_a[0] + m.axis[0] * offset_along_axis
    wa_y = right_a[1] + m.axis[1] * offset_along_axis
    wb_x = right_b[0] + m.axis[0] * offset_along_axis
    wb_y = right_b[1] + m.axis[1] * offset_along_axis
    ax.annotate(
        "", xy=(wa_x, wa_y), xytext=(wb_x, wb_y),
        arrowprops=dict(arrowstyle="<->", color="#8800aa", lw=2.0),
        zorder=5,
    )
    label_x = (wa_x + wb_x) / 2 + 0.4
    label_y = (wa_y + wb_y) / 2
    ax.text(label_x, label_y,
            f"zone_width\n= {m.zone_width:.3f} mm",
            fontsize=9, color="#8800aa", va="center", zorder=6,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#8800aa",
                      alpha=0.85, linewidth=0.6))

    # Formula box (lower-left)
    formula_text = (
        f"$R = w / |\\theta|$\n"
        f"$= {m.zone_width:.3f} / |{math.radians(m.angle_degrees):.4f}|$\n"
        f"$= {m.radius:.2f}$ mm\n\n"
        f"angle = {m.angle_degrees}$^\\circ$"
    )
    ax.text(
        0.02, 0.03, formula_text,
        transform=ax.transAxes, fontsize=10, verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.4", fc="#f5f5ff", ec="#8888cc",
                  alpha=0.95),
        zorder=7,
    )

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_title(f"Fold Marker Anatomy  (marker 0, angle = {m.angle_degrees}$^\\circ$)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("x  (mm)")
    ax.set_ylabel("y  (mm)")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, linewidth=0.3, alpha=0.4)
    fig.tight_layout()

    out = FIGURES_DIR / "01_fold_marker_anatomy.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"\nFigure 4 saved: {out}")
    print(f"  Zoomed view of marker 0: angle = {m.angle_degrees} deg, "
          f"radius = {m.radius:.2f} mm, zone_width = {m.zone_width:.3f} mm.")


# ===================================================================
# Figure 5: All fold markers on the board
# ===================================================================
def fig_all_fold_markers():
    """Overlay all 8 fold markers on the board outline."""
    fig, ax = plt.subplots(figsize=(10, 9))

    # Board outline (light gray fill)
    draw_board_outline(ax, fill_color="#f2f2f2", edge_color="#bbbbbb",
                       linewidth=1.0, alpha=0.5)

    # Color cycle for marker indices
    cmap = plt.cm.tab10
    arrow_scale = 1.8  # length multiplier for arrows

    for i, m in enumerate(markers):
        color = cmap(i / max(len(markers) - 1, 1))
        perp = (-m.axis[1], m.axis[0])

        # Draw the two boundary lines
        for la, lb in [(m.line_a_start, m.line_a_end),
                       (m.line_b_start, m.line_b_end)]:
            ax.plot([la[0], lb[0]], [la[1], lb[1]],
                    color=color, linewidth=1.4, linestyle="--", zorder=2)

        # Center dot
        cx, cy = m.center
        ax.plot(cx, cy, "o", color=color, ms=5, zorder=4)

        # Axis arrow (green tint)
        al = m.zone_width * arrow_scale
        ax.annotate(
            "", xy=(cx + m.axis[0] * al, cy + m.axis[1] * al),
            xytext=(cx, cy),
            arrowprops=dict(arrowstyle="-|>", color="#009900", lw=1.5),
            zorder=5,
        )

        # Perpendicular arrow (orange tint)
        ax.annotate(
            "", xy=(cx + perp[0] * al, cy + perp[1] * al),
            xytext=(cx, cy),
            arrowprops=dict(arrowstyle="-|>", color="#cc6600", lw=1.5),
            zorder=5,
        )

        # Angle label near center
        ax.text(cx + perp[0] * al * 0.6 + 0.4,
                cy + perp[1] * al * 0.6 - 0.4,
                f"{m.angle_degrees:g}$^\\circ$",
                fontsize=8, color=color, fontweight="bold",
                ha="center", va="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec=color, alpha=0.85, linewidth=0.6))

    # Legend proxy artists
    legend_elements = [
        Line2D([0], [0], color="#009900", lw=2, label=r"$\vec{axis}$"),
        Line2D([0], [0], color="#cc6600", lw=2, label=r"$\vec{perp}$"),
        Line2D([0], [0], color="#888888", lw=1.4, linestyle="--",
               label="Zone boundaries"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper right")

    ax.set_title(f"All {len(markers)} Fold Markers on H-Shape Board",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("x  (mm)")
    ax.set_ylabel("y  (mm)")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.grid(True, linewidth=0.3, alpha=0.4)
    fig.tight_layout()

    out = FIGURES_DIR / "01_all_fold_markers.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"\nFigure 5 saved: {out}")
    print(f"  All {len(markers)} markers shown with axis (green) and perp (orange) arrows.")
    for i, m in enumerate(markers):
        print(f"    [{i}] angle={m.angle_degrees:g} deg, "
              f"center=({m.center[0]:.1f}, {m.center[1]:.1f}), "
              f"axis=({m.axis[0]:.2f}, {m.axis[1]:.2f})")


# ===================================================================
# Figure 6: Key formulas
# ===================================================================
def fig_formulas():
    """Clean reference sheet of key fold-marker relationships."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.invert_yaxis()
    ax.axis("off")

    ax.set_title("Key Fold Marker Relationships", fontsize=15,
                 fontweight="bold", pad=12)

    # Each entry: (y_position, math_text, description)
    entries = [
        (1.0,
         r"$R = \frac{w}{|\theta|}$",
         "Bend radius from zone width and angle (radians)"),
        (2.5,
         r"$\vec{perp} = (-axis_y,\; axis_x)$",
         "Perpendicular direction (90-degree CCW rotation of axis)"),
        (4.0,
         r"Axis normalization:",
         "Horizontal folds: force axis to $+x$\n"
         "Vertical folds: force axis to $+y$\n"
         "Ensures consistent perpendicular across parallel folds"),
        (5.8,
         r"$w = |line\_a - line\_b|_\perp$",
         "Zone width = perpendicular distance between boundary lines"),
    ]

    for y, formula, desc in entries:
        ax.text(0.5, y, formula, fontsize=14, va="center",
                color="#222222", fontweight="bold")
        ax.text(5.0, y, desc, fontsize=9, va="center",
                color="#555555", style="italic")

    # Decorative separator lines
    for y_sep in [1.75, 3.25, 4.9]:
        ax.axhline(y=y_sep, xmin=0.03, xmax=0.97, color="#cccccc",
                    linewidth=0.6)

    fig.tight_layout()

    out = FIGURES_DIR / "01_formulas.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"\nFigure 6 saved: {out}")
    print("  Reference sheet with bend radius, perp direction, axis normalization,")
    print("  and zone width formulas.")


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    fig_board_outline()
    fig_layer_stack()
    fig_traces_and_pads()
    fig_fold_marker_anatomy()
    fig_all_fold_markers()
    fig_formulas()

    print("\n" + "=" * 60)
    print(f"All figures written to {FIGURES_DIR}/")
    print("=" * 60)
