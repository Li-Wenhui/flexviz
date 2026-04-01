#!/usr/bin/env python3
"""
04 — 3D Fold Transformation
============================

Visualises how 2D board coordinates are mapped to 3D through
cylindrical bending.  Six figures:

  1. IN_ZONE cylindrical mapping cross-section
  2. AFTER zone: tangent plane continuation
  3. Continuity of local_perp / local_up across zone boundaries
  4. Full 3D folded H-shape board
  5. Grid deformation: flat vs. bent side-by-side
  6. Surface normal vectors on folded board
"""

import sys
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "plugins" / "com_github_aightech_flexviz"))
FIGURES_DIR = ROOT / "docs" / "math" / "figures"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — needed for projection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from kicad_parser import KiCadPCB
from geometry import extract_geometry
from markers import detect_fold_markers
from planar_subdivision import (
    split_board_into_regions,
    build_region_adjacency,
    polygon_centroid,
)
from bend_transform import (
    FoldDefinition,
    transform_point,
    compute_normal,
    create_fold_definitions,
)

# ---------------------------------------------------------------------------
# Load board
# ---------------------------------------------------------------------------
PCB_PATH = str(ROOT / "tests" / "test_data" / "h_shape.kicad_pcb")
pcb = KiCadPCB.load(PCB_PATH)
geom = extract_geometry(pcb)
markers = detect_fold_markers(pcb, "User.1")
outline = geom.outline.vertices
holes = [c.vertices for c in geom.cutouts]
regions = split_board_into_regions(outline, holes, markers, num_bend_subdivisions=8)
adjacency = build_region_adjacency(regions)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

marker_idx = {id(m): i for i, m in enumerate(markers)}
region_by_index = {r.index: r for r in regions}

# Build fold_defs_map: FoldMarker id -> FoldDefinition
fold_defs_map = {id(m): FoldDefinition.from_marker(m) for m in markers}


def recipe_for_transform(region_recipe):
    """Convert region fold_recipe (FoldMarker-based) to transform recipe
    (FoldDefinition-based)."""
    result = []
    for entry in region_recipe:
        m = entry[0]
        classification = entry[1]
        back_entry = entry[2] if len(entry) > 2 else False
        fold_def = fold_defs_map[id(m)]
        result.append((fold_def, classification, back_entry))
    return result


# Pick a fold with a large angle for figures 1-3
# Prefer a 90-degree fold; fall back to the largest angle
best_marker = None
best_angle = 0
for m in markers:
    if abs(m.angle_degrees) == 90:
        best_marker = m
        break
    if abs(m.angle_degrees) > best_angle:
        best_angle = abs(m.angle_degrees)
        best_marker = m
# F6 is -90 degrees
fold_demo = FoldDefinition.from_marker(best_marker)
demo_marker_idx = marker_idx[id(best_marker)]


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1: IN_ZONE Cylindrical Mapping Cross-Section
# ═══════════════════════════════════════════════════════════════════════════

def draw_cylindrical_mapping():
    fig, ax = plt.subplots(figsize=(10, 8))

    angle = fold_demo.angle
    R = fold_demo.radius
    hw = fold_demo.zone_width / 2
    w = fold_demo.zone_width

    # The cylinder axis is at the BEFORE boundary (perp_dist = -hw)
    # Draw the arc from theta=0 to theta=angle
    abs_angle = abs(angle)
    theta_range = np.linspace(0, abs_angle, 100)

    # Arc coordinates (cylinder centre at origin for drawing)
    arc_x = R * np.sin(theta_range)
    arc_y = R * (1 - np.cos(theta_range))
    if angle < 0:
        arc_y = -arc_y

    # Shift so the BEFORE boundary maps to perp = -hw
    # At theta=0: local_perp = R*sin(0) - hw = -hw  (correct)
    arc_perp = arc_x - hw

    ax.plot(arc_perp, arc_y, "b-", linewidth=2.5, label="Cylinder arc (IN_ZONE)")

    # Mark BEFORE boundary (theta=0)
    ax.plot(-hw, 0, "go", markersize=10, zorder=5)
    ax.annotate(r"BEFORE boundary ($\theta=0$)", (-hw, 0),
                textcoords="offset points", xytext=(-60, -30),
                fontsize=9, arrowprops=dict(arrowstyle="->"),
                bbox=dict(boxstyle="round,pad=0.3", fc="#d5f5e3"))

    # Mark AFTER boundary (theta=angle)
    end_perp = R * math.sin(abs_angle) - hw
    end_up = R * (1 - math.cos(abs_angle))
    if angle < 0:
        end_up = -end_up
    ax.plot(end_perp, end_up, "rs", markersize=10, zorder=5)
    ax.annotate(r"AFTER boundary ($\theta=\alpha$)", (end_perp, end_up),
                textcoords="offset points", xytext=(15, -25),
                fontsize=9, arrowprops=dict(arrowstyle="->"),
                bbox=dict(boxstyle="round,pad=0.3", fc="#fadbd8"))

    # Sample point at arc_fraction = 0.5
    f = 0.5
    theta_p = f * angle
    sp_perp = R * math.sin(abs(theta_p)) - hw
    sp_up = R * (1 - math.cos(abs(theta_p)))
    if angle < 0:
        sp_up = -sp_up
    ax.plot(sp_perp, sp_up, "ko", markersize=8, zorder=5)
    ax.annotate(f"f=0.5\n({sp_perp:.2f}, {sp_up:.2f})",
                (sp_perp, sp_up),
                textcoords="offset points", xytext=(15, 15),
                fontsize=8, arrowprops=dict(arrowstyle="->"),
                bbox=dict(boxstyle="round,pad=0.2", fc="#fef9e7"))

    # Draw radial lines to show the cylinder
    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        theta_v = frac * abs_angle
        rx = R * math.sin(theta_v)
        ry = R * (1 - math.cos(theta_v))
        if angle < 0:
            ry = -ry
        ax.plot([0 - hw, rx - hw], [0, ry], "k--", linewidth=0.4, alpha=0.4)

    # Formula annotations
    formulas = [
        r"$R = w / |\alpha| = %.2f$" % R,
        r"$dist = perp\_dist + hw$",
        r"$f = dist / w$  (arc fraction)",
        r"$\theta_p = f \cdot \alpha$",
        r"$perp' = R \sin|\theta_p| - hw$",
        r"$up' = R(1 - \cos|\theta_p|)$",
    ]
    # Place formulas in an inset text block within the plot area
    formula_block = "\n".join(formulas)
    # Position: upper-right quadrant of the plot
    fx = end_perp * 0.3
    fy_base = min(arc_y) * 0.05 if angle < 0 else max(arc_y) * 0.05
    ax.text(fx, fy_base, formula_block, fontsize=8.5, fontfamily="serif",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.9, lw=0.5))

    ax.set_xlabel(r"$perp'$ (mm)")
    ax.set_ylabel(r"$up'$ (mm)")
    ax.set_title(f"IN_ZONE: Cylindrical Mapping Cross-Section (F{demo_marker_idx}, "
                 f"{best_marker.angle_degrees:.0f} deg)", fontsize=12)
    ax.set_aspect("equal")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(FIGURES_DIR / "04_cylindrical_mapping.png"), dpi=300)
    plt.close(fig)
    print("  Saved 04_cylindrical_mapping.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: AFTER Zone — Rotated Plane Continuation
# ═══════════════════════════════════════════════════════════════════════════

def draw_after_plane():
    fig, ax = plt.subplots(figsize=(10, 8))

    angle = fold_demo.angle
    R = fold_demo.radius
    hw = fold_demo.zone_width / 2
    abs_angle = abs(angle)

    # Draw cylinder arc (IN_ZONE)
    theta_range = np.linspace(0, abs_angle, 100)
    arc_x = R * np.sin(theta_range) - hw
    arc_y = R * (1 - np.cos(theta_range))
    if angle < 0:
        arc_y = -arc_y

    ax.plot(arc_x, arc_y, "b-", linewidth=2.5, label="IN_ZONE arc")

    # AFTER: tangent plane from arc end
    zone_end_perp = R * math.sin(abs_angle) - hw
    zone_end_up = R * (1 - math.cos(abs_angle))
    if angle < 0:
        zone_end_up = -zone_end_up

    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    # Draw tangent line extending from the arc end
    excess_range = np.linspace(0, 3.0 * hw, 50)
    after_perp = zone_end_perp + excess_range * cos_a
    after_up = zone_end_up + excess_range * sin_a

    ax.plot(after_perp, after_up, "r-", linewidth=2.5, label="AFTER plane")

    # Mark zone end
    ax.plot(zone_end_perp, zone_end_up, "ks", markersize=10, zorder=5)
    ax.annotate(f"zone end\n({zone_end_perp:.2f}, {zone_end_up:.2f})",
                (zone_end_perp, zone_end_up),
                textcoords="offset points", xytext=(-80, -30),
                fontsize=8, arrowprops=dict(arrowstyle="->"),
                bbox=dict(boxstyle="round,pad=0.2", fc="#fef9e7"))

    # Mark a sample excess point
    excess = 2.0 * hw
    sp_perp = zone_end_perp + excess * cos_a
    sp_up = zone_end_up + excess * sin_a
    ax.plot(sp_perp, sp_up, "ro", markersize=8, zorder=5)
    ax.annotate(f"excess = {excess:.2f} mm",
                (sp_perp, sp_up),
                textcoords="offset points", xytext=(15, 15),
                fontsize=8, arrowprops=dict(arrowstyle="->"),
                bbox=dict(boxstyle="round,pad=0.2", fc="#fadbd8"))

    # Draw tangent direction arrow
    arrow_len = 1.5 * hw
    ax.annotate("", xy=(zone_end_perp + arrow_len * cos_a,
                        zone_end_up + arrow_len * sin_a),
                xytext=(zone_end_perp, zone_end_up),
                arrowprops=dict(arrowstyle="-|>", color="green", lw=2))
    ax.text(zone_end_perp + 0.5 * arrow_len * cos_a + hw * 0.3,
            zone_end_up + 0.5 * arrow_len * sin_a + hw * 0.3,
            "tangent dir", fontsize=8, color="green", fontweight="bold")

    # Formulas — placed as a single text block inside the plot
    formula_block = "\n".join([
        r"$excess = perp\_dist - hw$",
        r"$perp' = perp_{end} + excess \cdot \cos(\alpha)$",
        r"$up' = up_{end} + excess \cdot \sin(\alpha)$",
    ])
    # Place near the top-left of the plot area
    ax.text(min(arc_x) * 0.9, 0.02, formula_block, fontsize=9, fontfamily="serif",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, lw=0.5))

    # BEFORE boundary
    ax.plot(-hw, 0, "go", markersize=8, zorder=5)

    ax.set_xlabel(r"$perp'$ (mm)")
    ax.set_ylabel(r"$up'$ (mm)")
    ax.set_title("AFTER: Rotated Plane Continuation", fontsize=12)
    ax.set_aspect("equal")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(FIGURES_DIR / "04_after_plane.png"), dpi=300)
    plt.close(fig)
    print("  Saved 04_after_plane.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3: Continuity of Transformation Across Zone Boundaries
# ═══════════════════════════════════════════════════════════════════════════

def draw_continuity():
    fig, ax = plt.subplots(figsize=(10, 6))

    angle = fold_demo.angle
    R = fold_demo.radius
    hw = fold_demo.zone_width / 2
    w = fold_demo.zone_width
    abs_angle = abs(angle)

    # Sweep perp_dist from -2*hw to +3*hw
    pd = np.linspace(-2 * hw, 3 * hw, 500)

    local_perp = np.zeros_like(pd)
    local_up = np.zeros_like(pd)

    for i, perp_dist in enumerate(pd):
        if perp_dist < -hw:
            # BEFORE: identity
            local_perp[i] = perp_dist
            local_up[i] = 0.0
        elif perp_dist <= hw:
            # IN_ZONE
            dist_into = perp_dist + hw
            arc_frac = dist_into / w if w > 0 else 0
            theta = arc_frac * angle
            local_perp[i] = R * math.sin(abs(theta)) - hw
            local_up[i] = R * (1 - math.cos(abs(theta)))
            if angle < 0:
                local_up[i] = -local_up[i]
        else:
            # AFTER
            zone_end_perp = R * math.sin(abs_angle) - hw
            zone_end_up = R * (1 - math.cos(abs_angle))
            if angle < 0:
                zone_end_up = -zone_end_up
            excess = perp_dist - hw
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            local_perp[i] = zone_end_perp + excess * cos_a
            local_up[i] = zone_end_up + excess * sin_a

    ax.plot(pd, local_perp, "b-", linewidth=2, label=r"$perp'$ (local perp)")
    ax.plot(pd, local_up, "r-", linewidth=2, label=r"$up'$ (local up)")

    # Zone boundaries
    ax.axvline(-hw, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.axvline(hw, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(-hw, ax.get_ylim()[1] * 0.9, "$-hw$", ha="center", fontsize=9)
    ax.text(hw, ax.get_ylim()[1] * 0.9, "$+hw$", ha="center", fontsize=9)

    # Shade zones
    ymin, ymax = ax.get_ylim()
    ax.axvspan(-2 * hw, -hw, alpha=0.1, color="#2ecc71")
    ax.axvspan(-hw, hw, alpha=0.1, color="#e67e22")
    ax.axvspan(hw, 3 * hw, alpha=0.1, color="#3498db")
    ax.text(-1.5 * hw, ymin + 0.1 * (ymax - ymin), "BEFORE",
            ha="center", fontsize=8, color="#27ae60")
    ax.text(0, ymin + 0.1 * (ymax - ymin), "IN_ZONE",
            ha="center", fontsize=8, color="#d35400")
    ax.text(2 * hw, ymin + 0.1 * (ymax - ymin), "AFTER",
            ha="center", fontsize=8, color="#2980b9")

    # Mark continuity at boundaries
    # At -hw
    before_perp = -hw
    inzone_perp_start = R * math.sin(0) - hw  # = -hw
    ax.plot(-hw, before_perp, "ko", markersize=8, zorder=5)
    ax.plot(-hw, 0, "ko", markersize=8, zorder=5)

    # At +hw
    inzone_perp_end = R * math.sin(abs_angle) - hw
    inzone_up_end = R * (1 - math.cos(abs_angle))
    if angle < 0:
        inzone_up_end = -inzone_up_end
    ax.plot(hw, inzone_perp_end, "ko", markersize=6, zorder=5)
    ax.plot(hw, inzone_up_end, "ko", markersize=6, zorder=5)
    ax.annotate("continuous", (hw, inzone_perp_end),
                textcoords="offset points", xytext=(12, 8),
                fontsize=8, color="blue",
                arrowprops=dict(arrowstyle="->", color="blue"))

    ax.set_xlabel(r"$perp\_dist$ (mm)", fontsize=11)
    ax.set_ylabel("Transformed coordinate (mm)", fontsize=11)
    ax.set_title(f"Continuity of Transformation Across Zone Boundaries "
                 f"(F{demo_marker_idx})", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(FIGURES_DIR / "04_continuity.png"), dpi=300)
    plt.close(fig)
    print("  Saved 04_continuity.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4: Full 3D Folded H-Shape Board
# ═══════════════════════════════════════════════════════════════════════════

def draw_multi_fold_3d():
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.cm.tab20
    n_regions = len(regions)

    for r in regions:
        recipe = recipe_for_transform(r.fold_recipe)

        # Transform outline vertices to 3D
        pts3d = []
        for v in r.outline:
            p3 = transform_point(v, recipe)
            pts3d.append(p3)

        if len(pts3d) < 3:
            continue

        xs = [p[0] for p in pts3d]
        ys = [p[1] for p in pts3d]
        zs = [p[2] for p in pts3d]

        # Draw region as a filled polygon
        verts = [list(zip(xs, ys, zs))]
        color = cmap(r.index / max(n_regions - 1, 1))
        poly = Poly3DCollection(verts, alpha=0.6, linewidths=0.3,
                                edgecolors="black")
        poly.set_facecolor(color)
        ax.add_collection3d(poly)

    # Draw fold zone boundaries as thick lines
    for m in markers:
        fold_def = fold_defs_map[id(m)]
        for line_start, line_end in [(m.line_a_start, m.line_a_end),
                                     (m.line_b_start, m.line_b_end)]:
            # Find a region containing each point to get the recipe
            for r in regions:
                c = polygon_centroid(r.outline)
                # Check if line midpoint is close to a region
                mid = ((line_start[0] + line_end[0]) / 2,
                       (line_start[1] + line_end[1]) / 2)
                # Rough check: use the line_start with the region's recipe
                # (The fold line points should be on region boundaries)
                pass  # Skip fold line overlay for clarity

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("H-Shape Board After 3D Folding", fontsize=13)

    # Set reasonable view angle
    ax.view_init(elev=30, azim=-60)

    # Try to equalise axes
    all_pts = []
    for r in regions:
        recipe = recipe_for_transform(r.fold_recipe)
        for v in r.outline:
            all_pts.append(transform_point(v, recipe))
    if all_pts:
        all_pts = np.array(all_pts)
        mid = all_pts.mean(axis=0)
        span = max(all_pts.max(axis=0) - all_pts.min(axis=0)) / 2
        ax.set_xlim(mid[0] - span, mid[0] + span)
        ax.set_ylim(mid[1] - span, mid[1] + span)
        ax.set_zlim(mid[2] - span, mid[2] + span)

    fig.tight_layout()
    fig.savefig(str(FIGURES_DIR / "04_multi_fold_3d.png"), dpi=300)
    plt.close(fig)
    print("  Saved 04_multi_fold_3d.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5: Grid Deformation — Flat vs. Bent
# ═══════════════════════════════════════════════════════════════════════════

def draw_grid_deformation():
    fig, (ax_flat, ax_3d) = plt.subplots(
        1, 2, figsize=(16, 7),
        subplot_kw={"projection": "3d"}
    )
    # Actually we need different projections: 2D for flat, 3D for bent
    plt.close(fig)

    fig = plt.figure(figsize=(16, 7))
    ax_flat = fig.add_subplot(121)
    ax_3d = fig.add_subplot(122, projection="3d")

    # Select regions along the left bar of the H (low x, varying y)
    # These cross multiple folds: F6, F0, F5, F3, F1
    left_arm_regions = [r for r in regions if polygon_centroid(r.outline)[0] < 12]

    if not left_arm_regions:
        left_arm_regions = regions[:20]

    # Compute bounding box of selected regions
    all_verts = []
    for r in left_arm_regions:
        all_verts.extend(r.outline)
    all_verts = np.array(all_verts)
    x_min, y_min = all_verts.min(axis=0)
    x_max, y_max = all_verts.max(axis=0)

    # Create grid
    nx, ny = 15, 40
    gx = np.linspace(x_min + 0.3, x_max - 0.3, nx)
    gy = np.linspace(y_min + 0.3, y_max - 0.3, ny)

    # For each grid point, find which region it belongs to
    from planar_subdivision import point_in_polygon

    grid_2d = []
    grid_3d = []
    grid_colors = []

    for xi in gx:
        for yi in gy:
            pt = (xi, yi)
            # Find containing region
            for r in left_arm_regions:
                if point_in_polygon(pt, r.outline):
                    recipe = recipe_for_transform(r.fold_recipe)
                    p3 = transform_point(pt, recipe)
                    grid_2d.append(pt)
                    grid_3d.append(p3)
                    # Colour by original y coordinate
                    grid_colors.append(yi)
                    break

    if not grid_2d:
        print("  Warning: no grid points found in left arm regions")
        plt.close(fig)
        return

    grid_2d = np.array(grid_2d)
    grid_3d = np.array(grid_3d)
    grid_colors = np.array(grid_colors)

    # Left subplot: 2D flat grid
    # Draw region outlines lightly
    for r in left_arm_regions:
        xs = [v[0] for v in r.outline] + [r.outline[0][0]]
        ys = [v[1] for v in r.outline] + [r.outline[0][1]]
        ax_flat.plot(xs, ys, "k-", linewidth=0.3, alpha=0.3)

    sc = ax_flat.scatter(grid_2d[:, 0], grid_2d[:, 1], c=grid_colors,
                         cmap="plasma", s=8, zorder=3)
    ax_flat.set_xlabel("x (mm)")
    ax_flat.set_ylabel("y (mm)")
    ax_flat.set_title("Flat (2D)")
    ax_flat.set_aspect("equal")
    ax_flat.invert_yaxis()

    # Right subplot: 3D bent
    ax_3d.scatter(grid_3d[:, 0], grid_3d[:, 1], grid_3d[:, 2],
                  c=grid_colors, cmap="plasma", s=8)
    ax_3d.set_xlabel("X (mm)")
    ax_3d.set_ylabel("Y (mm)")
    ax_3d.set_zlabel("Z (mm)")
    ax_3d.set_title("Bent (3D)")
    ax_3d.view_init(elev=20, azim=-70)

    fig.suptitle("Grid Deformation: Flat -> Bent", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(str(FIGURES_DIR / "04_grid_deformation.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved 04_grid_deformation.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 6: Surface Normal Vectors on Folded Board
# ═══════════════════════════════════════════════════════════════════════════

def draw_normals():
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Draw region surfaces lightly
    cmap = plt.cm.tab20
    n_regions = len(regions)

    for r in regions:
        recipe = recipe_for_transform(r.fold_recipe)
        pts3d = [transform_point(v, recipe) for v in r.outline]
        if len(pts3d) < 3:
            continue
        xs = [p[0] for p in pts3d]
        ys = [p[1] for p in pts3d]
        zs = [p[2] for p in pts3d]
        verts = [list(zip(xs, ys, zs))]
        poly = Poly3DCollection(verts, alpha=0.3, linewidths=0.2,
                                edgecolors="gray")
        poly.set_facecolor(cmap(r.index / max(n_regions - 1, 1)))
        ax.add_collection3d(poly)

    # Sample points and compute normals
    arrow_scale = 3.0  # Length of normal arrows

    # Colour normals by classification type
    normal_colors = {
        "flat": "#2ecc71",    # green: no folds
        "in_zone": "#e67e22", # orange: last entry is IN_ZONE
        "after": "#3498db",   # blue: all entries are AFTER
    }

    arrow_data = {"flat": [], "in_zone": [], "after": []}

    for r in regions:
        recipe = recipe_for_transform(r.fold_recipe)

        # Sample 3-5 points per region
        c = polygon_centroid(r.outline)
        sample_pts = [c]
        # Add a few outline midpoints
        n_outline = len(r.outline)
        for i in range(0, n_outline, max(1, n_outline // 3)):
            p1 = r.outline[i]
            p2 = r.outline[(i + 1) % n_outline]
            mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            sample_pts.append(mid)

        # Limit samples
        sample_pts = sample_pts[:5]

        # Determine classification category
        if not r.fold_recipe:
            cat = "flat"
        elif any(e[1] == "IN_ZONE" for e in r.fold_recipe):
            cat = "in_zone"
        else:
            cat = "after"

        for pt in sample_pts:
            p3 = transform_point(pt, recipe)
            n3 = compute_normal(pt, recipe)
            arrow_data[cat].append((p3, n3))

    # Draw arrows for each category
    # Subsample to avoid clutter
    max_arrows_per_cat = 50
    for cat, data in arrow_data.items():
        if len(data) > max_arrows_per_cat:
            step = len(data) // max_arrows_per_cat
            data = data[::step]
        color = normal_colors[cat]
        for p3, n3 in data:
            ax.quiver(p3[0], p3[1], p3[2],
                      n3[0] * arrow_scale, n3[1] * arrow_scale, n3[2] * arrow_scale,
                      color=color, linewidth=0.8, arrow_length_ratio=0.15, alpha=0.7)

    # Legend
    legend_handles = [
        mpatches.Patch(color="#2ecc71", label="Flat region"),
        mpatches.Patch(color="#e67e22", label="Bend zone (IN_ZONE)"),
        mpatches.Patch(color="#3498db", label="Folded region (AFTER)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("Surface Normal Vectors on Folded Board", fontsize=13)
    ax.view_init(elev=30, azim=-60)

    # Equalise axes
    all_pts = []
    for r in regions:
        recipe = recipe_for_transform(r.fold_recipe)
        for v in r.outline:
            all_pts.append(transform_point(v, recipe))
    if all_pts:
        all_pts = np.array(all_pts)
        mid = all_pts.mean(axis=0)
        span = max(all_pts.max(axis=0) - all_pts.min(axis=0)) / 2
        ax.set_xlim(mid[0] - span, mid[0] + span)
        ax.set_ylim(mid[1] - span, mid[1] + span)
        ax.set_zlim(mid[2] - span, mid[2] + span)

    fig.tight_layout()
    fig.savefig(str(FIGURES_DIR / "04_normals.png"), dpi=300)
    plt.close(fig)
    print("  Saved 04_normals.png")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Board: {len(markers)} markers, {len(regions)} regions")
    print("Generating figures ...")

    draw_cylindrical_mapping()
    draw_after_plane()
    draw_continuity()
    draw_multi_fold_3d()
    draw_grid_deformation()
    draw_normals()

    print("Done.")
