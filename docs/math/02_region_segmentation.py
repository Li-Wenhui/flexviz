#!/usr/bin/env python3
"""
02_region_segmentation.py -- Visualize how fold markers split a board into regions.

Loads the h_shape board, generates cutting lines from fold markers, runs the
planar subdivision algorithm, and produces annotated figures showing each step.

Figures saved to docs/math/figures/:
  02_cutting_lines.png        -- Cutting lines from all 8 fold markers
  02_all_regions.png          -- Color-coded regions after subdivision
  02_region_classification.png -- Regions colored by fold recipe classification
  02_face_tracing.png         -- Illustration of the CW edge-tracing rule
  02_winding_order.png        -- CCW vs CW winding and region filtering
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Arc
import numpy as np

# ---------------------------------------------------------------------------
# Module path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "plugins" / "com_github_aightech_flexviz"))

from kicad_parser import KiCadPCB
from geometry import extract_geometry
from markers import detect_fold_markers
from planar_subdivision import (
    split_board_into_regions,
    create_cutting_lines_from_marker_segments,
    signed_area,
    PlanarSubdivision,
    filter_valid_board_regions,
    polygon_centroid,
    classify_point_vs_fold,
)

FIGURES_DIR = ROOT / "docs" / "math" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DPI = 300

# ---------------------------------------------------------------------------
# Load the h_shape board
# ---------------------------------------------------------------------------
pcb_path = ROOT / "tests" / "test_data" / "h_shape.kicad_pcb"
pcb = KiCadPCB.load(str(pcb_path))
geom = extract_geometry(pcb)
markers = detect_fold_markers(pcb)

outline_verts = geom.outline.vertices
holes = [c.vertices for c in geom.cutouts]

NUM_SUBDIVISIONS = 8

print(f"Board outline: {len(outline_verts)} vertices")
print(f"Cutouts: {len(holes)}")
print(f"Fold markers: {len(markers)}")
for i, m in enumerate(markers):
    print(f"  [{i}] angle={m.angle_degrees:+.1f} deg, "
          f"center=({m.center[0]:.2f}, {m.center[1]:.2f}), "
          f"zone_width={m.zone_width:.3f} mm")

# ---------------------------------------------------------------------------
# Helper: draw the board outline (with fill) on an axes
# ---------------------------------------------------------------------------

def draw_board_outline(ax, fill_color="#e8e8e8", edge_color="#888888",
                       linewidth=1.0, zorder=0):
    """Draw the board outline polygon as a filled shape with cutouts."""
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import PathPatch

    # Build a compound path: outer CCW + each hole CW
    def ring_codes(n, close=True):
        codes = [MplPath.MOVETO] + [MplPath.LINETO] * (n - 1)
        if close:
            codes.append(MplPath.CLOSEPOLY)
        return codes

    verts_all = list(outline_verts) + [outline_verts[0]]
    codes_all = ring_codes(len(outline_verts))

    for hole in holes:
        verts_all += list(hole) + [hole[0]]
        codes_all += ring_codes(len(hole))

    path = MplPath(verts_all, codes_all)
    patch = PathPatch(path, facecolor=fill_color, edgecolor=edge_color,
                      linewidth=linewidth, zorder=zorder)
    ax.add_patch(patch)


def set_board_limits(ax, margin=2.0):
    """Set axis limits to fit the board with a margin."""
    xs = [v[0] for v in outline_verts]
    ys = [v[1] for v in outline_verts]
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # KiCad Y-down


# ==========================================================================
# Figure 1: Cutting lines from all fold markers
# ==========================================================================

def figure_cutting_lines():
    print("\n--- Figure 1: Cutting Lines ---")
    fig, ax = plt.subplots(figsize=(10, 10))
    draw_board_outline(ax)

    # We will annotate the first marker that has a clear layout
    annotate_idx = 0

    for midx, marker in enumerate(markers):
        lines = create_cutting_lines_from_marker_segments(
            marker, num_subdivisions=NUM_SUBDIVISIONS
        )
        n_lines = len(lines)

        for lidx, (eq, p1, p2) in enumerate(lines):
            is_boundary = (lidx == 0) or (lidx == n_lines - 1)
            lw = 1.4 if is_boundary else 0.5
            lc = "#d32f2f" if is_boundary else "#64b5f6"
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    color=lc, linewidth=lw, zorder=2)

    # Annotate one fold zone
    m0 = markers[annotate_idx]
    cl0 = create_cutting_lines_from_marker_segments(m0, NUM_SUBDIVISIONS)
    first_line = cl0[0]
    last_line = cl0[-1]
    mid_x = m0.center[0]

    # Draw a double-headed arrow between the first and last cutting line
    perp = (-m0.axis[1], m0.axis[0])
    hw = m0.zone_width / 2
    y_top = m0.center[1] - hw * perp[1] / abs(perp[1]) if abs(perp[1]) > 0.01 else first_line[1][1]
    y_bot = m0.center[1] + hw * perp[1] / abs(perp[1]) if abs(perp[1]) > 0.01 else last_line[1][1]

    p_top = (first_line[1][0] + first_line[2][0]) / 2, first_line[1][1]
    p_bot = (last_line[1][0] + last_line[2][0]) / 2, last_line[1][1]

    # Place annotation to the right of the cutting line extent
    ann_x = max(first_line[2][0], last_line[2][0]) + 1.0

    ax.annotate(
        "",
        xy=(ann_x, p_bot[1]), xytext=(ann_x, p_top[1]),
        arrowprops=dict(arrowstyle="<->", color="black", lw=1.2),
        zorder=5,
    )
    ax.text(ann_x + 0.5, (p_top[1] + p_bot[1]) / 2,
            f"zone width\n= {m0.zone_width:.2f} mm",
            fontsize=8, va="center", ha="left", zorder=5)

    ax.text(ann_x + 0.5, p_bot[1] + 1.0,
            f"{NUM_SUBDIVISIONS} subdivisions\n"
            r"$\rightarrow$ " f"{NUM_SUBDIVISIONS + 1} cutting lines",
            fontsize=8, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray",
                      alpha=0.9),
            zorder=5)

    # Mark fold centers
    for midx, marker in enumerate(markers):
        ax.plot(*marker.center, "k+", markersize=6, markeredgewidth=1.0, zorder=4)

    # Legend entries
    ax.plot([], [], color="#d32f2f", linewidth=1.4, label="Boundary lines (first/last)")
    ax.plot([], [], color="#64b5f6", linewidth=0.7, label="Interior subdivision lines")
    ax.plot([], [], "k+", markersize=6, label="Fold center")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

    set_board_limits(ax, margin=5.0)
    ax.set_title(
        f"Cutting Lines from {len(markers)} Fold Markers "
        f"({NUM_SUBDIVISIONS} subdivisions each)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")

    out = FIGURES_DIR / "02_cutting_lines.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ==========================================================================
# Figure 2: All regions after subdivision
# ==========================================================================

def figure_all_regions():
    print("\n--- Figure 2: All Regions ---")
    regions = split_board_into_regions(
        outline_verts, holes, markers, num_bend_subdivisions=NUM_SUBDIVISIONS
    )
    print(f"  Total regions: {len(regions)}")

    fig, ax = plt.subplots(figsize=(10, 10))

    # Use a broad colormap
    cmap = plt.cm.Set3

    for r in regions:
        verts = np.array(r.outline + [r.outline[0]])
        color = cmap(r.index % cmap.N / cmap.N)
        ax.fill(verts[:, 0], verts[:, 1], color=color, edgecolor="black",
                linewidth=0.4, zorder=1)

        # Label with index at centroid
        cx, cy = polygon_centroid(r.outline)
        ax.text(cx, cy, str(r.index), fontsize=5, ha="center", va="center",
                fontweight="bold", zorder=3,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                          alpha=0.7))

    set_board_limits(ax, margin=2.0)
    ax.set_title(
        f"Board Regions After Planar Subdivision ({len(regions)} regions)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")

    out = FIGURES_DIR / "02_all_regions.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ==========================================================================
# Figure 3: Region classification by fold recipe
# ==========================================================================

def figure_region_classification():
    print("\n--- Figure 3: Region Classification ---")
    regions = split_board_into_regions(
        outline_verts, holes, markers, num_bend_subdivisions=NUM_SUBDIVISIONS
    )

    fig, ax = plt.subplots(figsize=(10, 10))

    color_map = {
        "flat": "#66bb6a",       # green
        "IN_ZONE": "#ffa726",    # orange
        "AFTER": "#42a5f5",      # blue
    }

    counts = {"flat": 0, "IN_ZONE": 0, "AFTER": 0}

    for r in regions:
        recipe = r.fold_recipe
        if not recipe:
            category = "flat"
        else:
            last_classification = recipe[-1][1]
            category = last_classification

        color = color_map.get(category, "#cccccc")
        counts[category] += 1

        verts = np.array(r.outline + [r.outline[0]])
        ax.fill(verts[:, 0], verts[:, 1], color=color, edgecolor="black",
                linewidth=0.3, alpha=0.75, zorder=1)

    # Draw fold marker center lines (dashed)
    for marker in markers:
        axis_x, axis_y = marker.axis
        cx, cy = marker.center
        extent = 20.0
        x0 = cx - extent * axis_x
        y0 = cy - extent * axis_y
        x1 = cx + extent * axis_x
        y1 = cy + extent * axis_y
        ax.plot([x0, x1], [y0, y1], color="black", linewidth=0.7,
                linestyle="--", alpha=0.5, zorder=2)

    # Legend
    legend_patches = [
        mpatches.Patch(color=color_map["flat"], label=f"Flat / BEFORE ({counts['flat']})"),
        mpatches.Patch(color=color_map["IN_ZONE"],
                       label=f"Bend Zone / IN_ZONE ({counts['IN_ZONE']})"),
        mpatches.Patch(color=color_map["AFTER"],
                       label=f"Rotated / AFTER ({counts['AFTER']})"),
    ]
    ax.legend(handles=legend_patches, loc="lower left", fontsize=8,
                  framealpha=0.9)

    set_board_limits(ax, margin=2.0)
    ax.set_title("Region Classification by Fold Recipe", fontsize=12,
                     fontweight="bold")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")

    out = FIGURES_DIR / "02_region_classification.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")
    print(f"  Flat: {counts['flat']}, IN_ZONE: {counts['IN_ZONE']}, "
          f"AFTER: {counts['AFTER']}")


# ==========================================================================
# Figure 4: Face tracing -- next CW edge rule
# ==========================================================================

def figure_face_tracing():
    print("\n--- Figure 4: Face Tracing ---")

    # Use a synthetic hexagonal polygon to clearly illustrate the algorithm.
    # This avoids thin-strip real regions that are hard to read.
    # Standard math coords (Y-up) for clarity.
    hex_r = 2.0
    verts = []
    for i in range(6):
        theta = np.pi / 6 + i * np.pi / 3  # start at 30 deg
        verts.append((hex_r * np.cos(theta), hex_r * np.sin(theta)))
    n = len(verts)

    # Add a "T-junction" extra edge at vertex 3 to show edge choice.
    # We will draw the hex boundary + one extra edge from vertex 3 outward
    # to demonstrate that the algorithm picks the next CW edge among options.
    extra_tip = (verts[3][0] - 1.5, verts[3][1] - 1.0)

    fig, ax = plt.subplots(figsize=(8, 8))

    poly = np.array(verts + [verts[0]])
    ax.fill(poly[:, 0], poly[:, 1], color="#d4e6f1", edgecolor="none", zorder=0)

    # Draw all edges with arrows
    for i in range(n):
        p_s = np.array(verts[i])
        p_e = np.array(verts[(i + 1) % n])
        ax.annotate(
            "",
            xy=p_e, xytext=p_s,
            arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=1.5,
                            shrinkA=5, shrinkB=5),
            zorder=3,
        )

    # Draw extra edge (alternative path at the junction vertex)
    vi = 3  # junction vertex
    ax.annotate(
        "",
        xy=extra_tip, xytext=verts[vi],
        arrowprops=dict(arrowstyle="-|>", color="#bdbdbd", lw=1.5,
                        shrinkA=5, shrinkB=5, linestyle="dashed"),
        zorder=2,
    )
    ax.text(extra_tip[0] - 0.3, extra_tip[1] - 0.2,
            "other edge\n(not chosen)",
            fontsize=7, ha="center", va="top", color="#757575",
            fontstyle="italic", zorder=5)

    # Draw numbered vertex dots
    for i, v in enumerate(verts):
        ax.plot(v[0], v[1], "o", color="#e74c3c", markersize=14, zorder=4,
                markeredgecolor="white", markeredgewidth=1.0)
        ax.text(v[0], v[1], str(i), fontsize=9, ha="center", va="center",
                color="white", fontweight="bold", zorder=5)

    # Highlight incoming edge (2 -> 3) and outgoing edge (3 -> 4)
    p_prev = np.array(verts[vi - 1])
    p_curr = np.array(verts[vi])
    p_next = np.array(verts[(vi + 1) % n])

    ax.annotate(
        "",
        xy=p_curr, xytext=p_prev,
        arrowprops=dict(arrowstyle="-|>", color="#27ae60", lw=3.0,
                        shrinkA=8, shrinkB=8),
        zorder=6,
    )
    ax.annotate(
        "",
        xy=p_next, xytext=p_curr,
        arrowprops=dict(arrowstyle="-|>", color="#e67e22", lw=3.0,
                        shrinkA=8, shrinkB=8),
        zorder=6,
    )

    # Draw the CW sweep arc at the junction vertex
    incoming_dir = p_curr - p_prev
    outgoing_dir = p_next - p_curr

    angle_reversed = np.degrees(np.arctan2(incoming_dir[1], incoming_dir[0])) + 180
    angle_chosen = np.degrees(np.arctan2(outgoing_dir[1], outgoing_dir[0]))

    # CW sweep from reversed-incoming to chosen outgoing
    # In standard coords, CW means decreasing angle.  We draw the arc
    # from the chosen angle to the reversed angle (the short way CW).
    # matplotlib Arc draws CCW from theta1 to theta2, so set them accordingly.
    # We want the arc from angle_chosen to angle_reversed going CCW (the long
    # way is the CW visual sweep).

    # Normalize angles to [0, 360)
    def norm360(a):
        return a % 360

    a_rev = norm360(angle_reversed)
    a_out = norm360(angle_chosen)

    # Arc CCW from a_out to a_rev (this is the CW "sweep" in visual terms)
    if a_rev < a_out:
        a_rev += 360

    arc_r = 0.7
    arc = Arc(p_curr, 2 * arc_r, 2 * arc_r,
              angle=0, theta1=a_out, theta2=a_rev,
              color="#8e44ad", lw=2.0, linestyle="--", zorder=7)
    ax.add_patch(arc)

    # Small label near the arc
    mid_angle = np.radians((a_out + a_rev) / 2)
    label_r = arc_r + 0.45
    ax.text(
        p_curr[0] + label_r * np.cos(mid_angle),
        p_curr[1] + label_r * np.sin(mid_angle),
        "CW\nsweep",
        fontsize=8, ha="center", va="center",
        color="#8e44ad", fontweight="bold", zorder=8,
    )

    # Draw the reversed-incoming dashed ray
    ray_len = 0.9
    rev_angle_rad = np.radians(angle_reversed)
    ax.annotate(
        "",
        xy=(p_curr[0] + ray_len * np.cos(rev_angle_rad),
            p_curr[1] + ray_len * np.sin(rev_angle_rad)),
        xytext=p_curr,
        arrowprops=dict(arrowstyle="-|>", color="#888888", lw=1.0,
                        shrinkA=8, shrinkB=2, linestyle="dotted"),
        zorder=6,
    )
    ax.text(
        p_curr[0] + (ray_len + 0.3) * np.cos(rev_angle_rad),
        p_curr[1] + (ray_len + 0.3) * np.sin(rev_angle_rad),
        "reversed\nincoming",
        fontsize=6, ha="center", va="center", color="#888888",
        fontstyle="italic", zorder=8,
    )

    # Annotation text box
    ax.text(
        0.5, 0.03,
        "At each vertex, take the next edge in clockwise angular order\n"
        "relative to the reversed incoming edge direction.\n"
        "This traces the boundary of one face of the planar graph.",
        transform=ax.transAxes, fontsize=9, ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9),
    )

    # Legend
    ax.plot([], [], color="#27ae60", linewidth=3.0, label="Incoming edge")
    ax.plot([], [], color="#e67e22", linewidth=3.0, label="Next CW edge (chosen)")
    ax.plot([], [], color="#bdbdbd", linewidth=1.5, linestyle="--",
            label="Alternative edge (not chosen)")
    ax.plot([], [], color="#8e44ad", linewidth=2.0, linestyle="--",
            label="CW angle sweep")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)
    ax.set_aspect("equal")
    ax.set_title("Face Tracing: Next Clockwise Edge Rule",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    out = FIGURES_DIR / "02_face_tracing.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ==========================================================================
# Figure 5: Winding order and filtering
# ==========================================================================

def figure_winding_order():
    print("\n--- Figure 5: Winding Order ---")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # --- Left panel: CCW (valid) region ---
    ax_ccw = axes[0]
    # Pentagon that is CCW in standard math coords (Y-up):
    # Go counter-clockwise: bottom-left, bottom-right, right, top, left
    ccw_verts = [
        (0.0, 1.0),
        (2.5, 0.0),
        (4.0, 1.5),
        (3.0, 3.0),
        (1.0, 3.0),
    ]
    area_ccw = signed_area(ccw_verts)
    assert area_ccw > 0, f"Expected positive area for CCW, got {area_ccw}"
    poly_ccw = np.array(ccw_verts + [ccw_verts[0]])

    ax_ccw.fill(poly_ccw[:, 0], poly_ccw[:, 1], color="#a5d6a7", edgecolor="none",
                zorder=0)

    n_ccw = len(ccw_verts)
    for i in range(n_ccw):
        p_s = np.array(ccw_verts[i])
        p_e = np.array(ccw_verts[(i + 1) % n_ccw])
        ax_ccw.annotate(
            "",
            xy=p_e, xytext=p_s,
            arrowprops=dict(arrowstyle="-|>", color="#2e7d32", lw=2.0,
                            shrinkA=5, shrinkB=5),
            zorder=3,
        )

    for i, v in enumerate(ccw_verts):
        ax_ccw.plot(v[0], v[1], "o", color="#2e7d32", markersize=8, zorder=4,
                    markeredgecolor="white", markeredgewidth=0.5)

    cx_ccw = sum(v[0] for v in ccw_verts) / n_ccw
    cy_ccw = sum(v[1] for v in ccw_verts) / n_ccw
    ax_ccw.text(cx_ccw, cy_ccw,
                f"$A = {area_ccw:+.1f}$\n$A > 0$  (CCW)",
                fontsize=11, ha="center", va="center", fontweight="bold",
                color="#1b5e20",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#2e7d32",
                          alpha=0.85),
                zorder=5)

    ax_ccw.set_title("Valid Region (CCW winding)", fontsize=11,
                     fontweight="bold", color="#2e7d32")
    ax_ccw.text(0.5, -0.06, "Board material -- kept",
                transform=ax_ccw.transAxes, fontsize=10, ha="center",
                color="#2e7d32", fontweight="bold")

    ax_ccw.set_xlim(-0.8, 5.0)
    ax_ccw.set_ylim(-0.8, 4.0)
    ax_ccw.set_aspect("equal")
    ax_ccw.set_xlabel("X")
    ax_ccw.set_ylabel("Y")

    # --- Right panel: CW (invalid / outer boundary) region ---
    ax_cw = axes[1]
    cw_verts = list(reversed(ccw_verts))
    area_cw = signed_area(cw_verts)
    assert area_cw < 0, f"Expected negative area for CW, got {area_cw}"
    poly_cw = np.array(cw_verts + [cw_verts[0]])

    ax_cw.fill(poly_cw[:, 0], poly_cw[:, 1], color="#ef9a9a", edgecolor="none",
               zorder=0)

    n_cw = len(cw_verts)
    for i in range(n_cw):
        p_s = np.array(cw_verts[i])
        p_e = np.array(cw_verts[(i + 1) % n_cw])
        ax_cw.annotate(
            "",
            xy=p_e, xytext=p_s,
            arrowprops=dict(arrowstyle="-|>", color="#c62828", lw=2.0,
                            shrinkA=5, shrinkB=5),
            zorder=3,
        )

    for i, v in enumerate(cw_verts):
        ax_cw.plot(v[0], v[1], "o", color="#c62828", markersize=8, zorder=4,
                   markeredgecolor="white", markeredgewidth=0.5)

    cx_cw = sum(v[0] for v in cw_verts) / n_cw
    cy_cw = sum(v[1] for v in cw_verts) / n_cw
    ax_cw.text(cx_cw, cy_cw,
               f"$A = {area_cw:+.1f}$\n$A < 0$  (CW)",
               fontsize=11, ha="center", va="center", fontweight="bold",
               color="#b71c1c",
               bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#c62828",
                         alpha=0.85),
               zorder=5)

    # Big X overlay
    ax_cw.plot([0.0, 4.0], [0.0, 3.0], color="#c62828", linewidth=3,
               alpha=0.3, zorder=2)
    ax_cw.plot([0.0, 4.0], [3.0, 0.0], color="#c62828", linewidth=3,
               alpha=0.3, zorder=2)

    ax_cw.set_title("Filtered Region (CW winding)", fontsize=11,
                     fontweight="bold", color="#c62828")
    ax_cw.text(0.5, -0.06, "Outer boundary face -- filtered out",
               transform=ax_cw.transAxes, fontsize=10, ha="center",
               color="#c62828", fontweight="bold")

    ax_cw.set_xlim(-0.8, 5.0)
    ax_cw.set_ylim(-0.8, 4.0)
    ax_cw.set_aspect("equal")
    ax_cw.set_xlabel("X")
    ax_cw.set_ylabel("Y")

    # Shared formula at the bottom
    fig.text(
        0.5, -0.02,
        r"$A = \frac{1}{2} \sum_{i=0}^{n-1} "
        r"\left( x_i\, y_{i+1} \;-\; x_{i+1}\, y_i \right)$"
        r"$\qquad\qquad A > 0$: CCW (valid)"
        r"$\qquad A < 0$: CW (filtered)",
        fontsize=11, ha="center", va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="gray",
                  alpha=0.9),
    )

    fig.suptitle("Region Filtering by Winding Order", fontsize=13,
                 fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0.0, 1, 0.97])

    out = FIGURES_DIR / "02_winding_order.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ==========================================================================
# Main
# ==========================================================================

if __name__ == "__main__":
    figure_cutting_lines()
    figure_all_regions()
    figure_region_classification()
    figure_face_tracing()
    figure_winding_order()
    print("\nDone. All figures saved to", FIGURES_DIR)
