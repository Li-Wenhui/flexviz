#!/usr/bin/env python3
"""
03 — BFS Fold Recipe Computation
=================================

Visualises how fold recipes are built by BFS traversal of the region
adjacency graph.  Five figures:

  1. Region adjacency graph
  2. Recipe buildup along a BFS path
  3. Conceptual BEFORE / IN_ZONE / AFTER classification diagram
  4. Recipe-length colour map of the H-shape board
  5. Back-entry region highlights
"""

import sys
from pathlib import Path
from collections import deque

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "plugins" / "com_github_aightech_flexviz"))
FIGURES_DIR = ROOT / "docs" / "math" / "figures"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

from kicad_parser import KiCadPCB
from geometry import extract_geometry
from markers import detect_fold_markers
from planar_subdivision import (
    split_board_into_regions,
    build_region_adjacency,
    classify_point_vs_fold,
    polygon_centroid,
)
from bend_transform import FoldDefinition

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

# Build marker-index lookup
marker_idx = {id(m): i for i, m in enumerate(markers)}


def recipe_short(recipe):
    """Compact string for a fold recipe."""
    parts = []
    for entry in recipe:
        m = entry[0]
        cls = entry[1]
        back = entry[2] if len(entry) > 2 else False
        mi = marker_idx[id(m)]
        code = "A" if cls == "AFTER" else "Z"
        s = f"{code}{mi}"
        if back:
            s += "B"
        parts.append(s)
    return ",".join(parts) if parts else "anchor"


# Identify the anchor region (empty recipe)
anchor = None
for r in regions:
    if not r.fold_recipe:
        anchor = r
        break

# Region-index lookup
region_by_index = {r.index: r for r in regions}


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1: Region Adjacency Graph
# ═══════════════════════════════════════════════════════════════════════════

def draw_adjacency_graph():
    fig, ax = plt.subplots(figsize=(14, 10))

    # Assign colours by recipe length
    max_len = max(len(r.fold_recipe) for r in regions)
    cmap = plt.cm.viridis
    norm = plt.Normalize(0, max(max_len, 1))

    centroids = {}
    for r in regions:
        c = polygon_centroid(r.outline)
        centroids[r.index] = c

    # Draw edges
    drawn_edges = set()
    for ri, neighbours in adjacency.items():
        for rj in neighbours:
            key = (min(ri, rj), max(ri, rj))
            if key in drawn_edges:
                continue
            drawn_edges.add(key)
            c1 = centroids[ri]
            c2 = centroids[rj]
            ax.plot([c1[0], c2[0]], [c1[1], c2[1]],
                    color="0.75", linewidth=0.4, zorder=1)

    # Draw nodes
    for r in regions:
        c = centroids[r.index]
        rlen = len(r.fold_recipe)
        color = cmap(norm(rlen))
        size = 50
        marker_shape = "o"
        edgecolor = "black"
        linewidth = 0.5
        if r is anchor:
            marker_shape = "*"
            size = 200
            edgecolor = "red"
            linewidth = 2.0
        ax.scatter(c[0], c[1], s=size, c=[color], marker=marker_shape,
                   edgecolors=edgecolor, linewidths=linewidth, zorder=3)

    # Label a subset of nodes to avoid clutter (every 5th + anchor)
    for r in regions:
        if r.index % 8 == 0 or r is anchor:
            c = centroids[r.index]
            ax.annotate(str(r.index), c, fontsize=5, ha="center", va="center",
                        zorder=4, fontweight="bold")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6)
    cbar.set_label("Number of folds in recipe")

    # Draw board outline faintly
    ox = [v[0] for v in outline] + [outline[0][0]]
    oy = [v[1] for v in outline] + [outline[0][1]]
    ax.plot(ox, oy, "k-", linewidth=0.3, alpha=0.3)

    ax.set_title("Region Adjacency Graph")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(str(FIGURES_DIR / "03_adjacency_graph.png"), dpi=300)
    plt.close(fig)
    print("  Saved 03_adjacency_graph.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: Recipe Buildup Along a BFS Path
# ═══════════════════════════════════════════════════════════════════════════

def draw_recipe_buildup():
    """Trace a BFS path from anchor along the left bar of the H,
    collecting one representative region per distinct recipe length."""

    # BFS to find a path from anchor through 3+ folds
    parent = {anchor.index: None}
    queue = deque([anchor.index])
    visited = {anchor.index}
    order = [anchor.index]

    while queue:
        ci = queue.popleft()
        for ni in adjacency.get(ci, []):
            if ni not in visited:
                visited.add(ni)
                parent[ni] = ci
                queue.append(ni)
                order.append(ni)

    # Find the region with the longest recipe
    longest = max(regions, key=lambda r: len(r.fold_recipe))

    # Reconstruct path from anchor to longest
    path_indices = []
    cur = longest.index
    while cur is not None:
        path_indices.append(cur)
        cur = parent.get(cur)
    path_indices.reverse()

    # Select key regions along the path where recipe changes
    key_regions = []
    prev_len = -1
    for ri in path_indices:
        r = region_by_index[ri]
        rlen = len(r.fold_recipe)
        if rlen != prev_len:
            key_regions.append(r)
            prev_len = rlen
    # Limit to at most 8 for readability
    if len(key_regions) > 8:
        step = max(1, len(key_regions) // 7)
        key_regions = key_regions[::step]
        # Ensure last is included
        last = region_by_index[path_indices[-1]]
        if key_regions[-1] is not last:
            key_regions.append(last)

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.set_xlim(-0.5, len(key_regions) * 3.5)
    ax.set_ylim(-1, 2)
    ax.axis("off")

    colors_by_len = {0: "#2ecc71", 1: "#f1c40f", 2: "#e67e22",
                     3: "#e74c3c", 4: "#9b59b6", 5: "#3498db"}

    for i, r in enumerate(key_regions):
        x = i * 3.5
        rlen = len(r.fold_recipe)
        color = colors_by_len.get(rlen, "#95a5a6")

        # Draw box
        box = FancyBboxPatch((x - 1.2, -0.6), 2.4, 1.7,
                             boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor="black",
                             linewidth=1.2, alpha=0.85)
        ax.add_patch(box)

        # Region label
        ax.text(x, 0.85, f"R{r.index}", ha="center", va="center",
                fontsize=8, fontweight="bold")

        # Recipe text
        rs = recipe_short(r.fold_recipe)
        # Wrap long recipe strings
        if len(rs) > 16:
            mid = len(rs) // 2
            # Find nearest comma
            comma = rs.find(",", mid)
            if comma == -1:
                comma = rs.rfind(",", 0, mid)
            if comma != -1:
                rs = rs[:comma + 1] + "\n" + rs[comma + 1:]
        ax.text(x, 0.2, rs, ha="center", va="center",
                fontsize=5.5, fontfamily="monospace")

        # Arrow to next
        if i < len(key_regions) - 1:
            ax.annotate("", xy=(x + 1.5, 0.4), xytext=(x + 2.0, 0.4),
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    ax.set_title("Fold Recipe Buildup Along BFS Path", fontsize=12, pad=10)
    fig.tight_layout()
    fig.savefig(str(FIGURES_DIR / "03_recipe_buildup.png"), dpi=300)
    plt.close(fig)
    print("  Saved 03_recipe_buildup.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3: Classification Diagram (conceptual)
# ═══════════════════════════════════════════════════════════════════════════

def draw_classification_diagram():
    fig, ax = plt.subplots(figsize=(10, 4))

    hw = 1.0  # half-width of zone
    total_span = 5.0

    # BEFORE zone
    ax.axvspan(-total_span, -hw, alpha=0.25, color="#2ecc71", label="BEFORE")
    # IN_ZONE
    ax.axvspan(-hw, hw, alpha=0.25, color="#e67e22", label="IN_ZONE")
    # AFTER zone
    ax.axvspan(hw, total_span, alpha=0.25, color="#3498db", label="AFTER")

    # Board surface line
    ax.axhline(y=0, color="black", linewidth=2, zorder=5)

    # Zone boundaries
    ax.axvline(x=-hw, color="black", linewidth=1.5, linestyle="--", zorder=4)
    ax.axvline(x=hw, color="black", linewidth=1.5, linestyle="--", zorder=4)
    ax.axvline(x=0, color="gray", linewidth=0.8, linestyle=":", zorder=3)

    # Labels at boundaries
    ax.text(-hw, -0.4, "$-hw$", ha="center", va="top", fontsize=11, fontweight="bold")
    ax.text(hw, -0.4, "$+hw$", ha="center", va="top", fontsize=11, fontweight="bold")
    ax.text(0, -0.4, "center", ha="center", va="top", fontsize=9, color="gray")

    # Axis labels
    ax.annotate("", xy=(total_span - 0.2, -0.7), xytext=(-total_span + 0.2, -0.7),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
    ax.text(0, -0.85, r"$perp\_dist$", ha="center", va="top", fontsize=11)

    # Annotation boxes
    ax.text(-3.0, 0.5, r"$perp\_dist < -hw$" + "\nBEFORE",
            ha="center", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="#2ecc71", alpha=0.7))
    ax.text(0, 0.5, r"$-hw \leq perp\_dist \leq hw$" + "\nIN_ZONE",
            ha="center", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="#e67e22", alpha=0.7))
    ax.text(3.0, 0.5, r"$perp\_dist > hw$" + "\nAFTER",
            ha="center", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="#3498db", alpha=0.7))

    ax.set_xlim(-total_span, total_span)
    ax.set_ylim(-1.2, 1.5)
    ax.set_title("Point Classification Relative to Fold", fontsize=13)
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(str(FIGURES_DIR / "03_classification_diagram.png"), dpi=300)
    plt.close(fig)
    print("  Saved 03_classification_diagram.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4: Recipe Map (regions coloured by recipe length)
# ═══════════════════════════════════════════════════════════════════════════

def draw_recipe_map():
    fig, ax = plt.subplots(figsize=(12, 10))

    colors_by_len = {0: "#2ecc71", 1: "#f1c40f", 2: "#e67e22",
                     3: "#e74c3c", 4: "#9b59b6", 5: "#3498db"}

    for r in regions:
        rlen = len(r.fold_recipe)
        color = colors_by_len.get(rlen, "#95a5a6")
        xs = [v[0] for v in r.outline] + [r.outline[0][0]]
        ys = [v[1] for v in r.outline] + [r.outline[0][1]]
        ax.fill(xs, ys, color=color, alpha=0.55, edgecolor="black", linewidth=0.3)

    # Label regions with compact recipe at centroid
    for r in regions:
        c = polygon_centroid(r.outline)
        rs = recipe_short(r.fold_recipe)
        # Only label a representative subset (every 8th + anchor + ends of arms)
        if r.index % 9 == 0 or r is anchor or len(r.fold_recipe) >= 4:
            ax.text(c[0], c[1], rs, ha="center", va="center",
                    fontsize=3.5, fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7, lw=0.3))

    # Board outline
    ox = [v[0] for v in outline] + [outline[0][0]]
    oy = [v[1] for v in outline] + [outline[0][1]]
    ax.plot(ox, oy, "k-", linewidth=1.2)

    # Draw fold marker lines
    for i, m in enumerate(markers):
        for start, end in [(m.line_a_start, m.line_a_end), (m.line_b_start, m.line_b_end)]:
            ax.plot([start[0], end[0]], [start[1], end[1]],
                    "r-", linewidth=0.8, alpha=0.6)

    # Legend
    legend_handles = []
    labels_map = {0: "0 folds (anchor)", 1: "1 fold", 2: "2 folds",
                  3: "3 folds", 4: "4 folds", 5: "5 folds"}
    for k in sorted(colors_by_len):
        if any(len(r.fold_recipe) == k for r in regions):
            legend_handles.append(mpatches.Patch(color=colors_by_len[k],
                                                 alpha=0.6, label=labels_map.get(k, f"{k} folds")))
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    ax.set_title("Fold Recipe Map (H-Shape Board)", fontsize=13)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(str(FIGURES_DIR / "03_recipe_map.png"), dpi=300)
    plt.close(fig)
    print("  Saved 03_recipe_map.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5: Back-Entry Regions
# ═══════════════════════════════════════════════════════════════════════════

def draw_back_entry():
    fig, ax = plt.subplots(figsize=(12, 10))

    annotated = False

    for r in regions:
        has_back = any(
            (entry[2] if len(entry) > 2 else False)
            for entry in r.fold_recipe
        )
        xs = [v[0] for v in r.outline] + [r.outline[0][0]]
        ys = [v[1] for v in r.outline] + [r.outline[0][1]]

        if has_back:
            ax.fill(xs, ys, color="#e74c3c", alpha=0.55,
                    edgecolor="black", linewidth=0.4)
        else:
            ax.fill(xs, ys, color="#ecf0f1", alpha=0.4,
                    edgecolor="0.7", linewidth=0.2)

    # Draw fold marker lines prominently for folds that have back-entry regions
    back_fold_ids = set()
    for r in regions:
        for entry in r.fold_recipe:
            back = entry[2] if len(entry) > 2 else False
            if back:
                back_fold_ids.add(id(entry[0]))

    for m in markers:
        if id(m) in back_fold_ids:
            for start, end in [(m.line_a_start, m.line_a_end), (m.line_b_start, m.line_b_end)]:
                ax.plot([start[0], end[0]], [start[1], end[1]],
                        color="#c0392b", linewidth=2.0, zorder=5)
            # Label
            mi = marker_idx[id(m)]
            ax.text(m.center[0], m.center[1], f"F{mi}",
                    fontsize=7, ha="center", va="center",
                    fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.2", fc="#c0392b", alpha=0.9),
                    zorder=6)
        else:
            for start, end in [(m.line_a_start, m.line_a_end), (m.line_b_start, m.line_b_end)]:
                ax.plot([start[0], end[0]], [start[1], end[1]],
                        color="gray", linewidth=0.6, alpha=0.5)

    # Annotate one back-entry region
    for r in regions:
        has_back = any(
            (entry[2] if len(entry) > 2 else False)
            for entry in r.fold_recipe
        )
        if has_back and not annotated:
            c = polygon_centroid(r.outline)
            ax.annotate(
                "BFS entered from AFTER side\n"
                r"$\rightarrow$ entered_from_back = True",
                xy=(c[0], c[1]),
                xytext=(c[0] + 8, c[1] - 5),
                fontsize=8,
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
                bbox=dict(boxstyle="round,pad=0.4", fc="#fadbd8", ec="#c0392b", lw=1),
                zorder=7,
            )
            annotated = True
            break

    # Board outline
    ox = [v[0] for v in outline] + [outline[0][0]]
    oy = [v[1] for v in outline] + [outline[0][1]]
    ax.plot(ox, oy, "k-", linewidth=1.0)

    # Legend
    legend_handles = [
        mpatches.Patch(color="#e74c3c", alpha=0.6, label="Back-entry region"),
        mpatches.Patch(color="#ecf0f1", alpha=0.5, label="Normal region"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9)

    ax.set_title("Back-Entry Regions in H-Shape Board", fontsize=13)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(str(FIGURES_DIR / "03_back_entry.png"), dpi=300)
    plt.close(fig)
    print("  Saved 03_back_entry.png")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Board: {len(markers)} markers, {len(regions)} regions")
    print("Generating figures ...")

    draw_adjacency_graph()
    draw_recipe_buildup()
    draw_classification_diagram()
    draw_recipe_map()
    draw_back_entry()

    print("Done.")
