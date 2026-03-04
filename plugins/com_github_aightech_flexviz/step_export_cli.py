#!/usr/bin/env python3
"""Command-line STEP export for flex PCB visualization.

Usage:
    python step_export_cli.py <input.kicad_pcb> <output.step> [options]

Examples:
    python step_export_cli.py board.kicad_pcb board.step
    python step_export_cli.py board.kicad_pcb board.step --stiffener-layer User.2 --stiffener-thickness 0.2
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kicad_parser import KiCadPCB
from geometry import extract_geometry
from markers import detect_fold_markers
from config import FlexConfig
from stiffener import extract_stiffeners
from step_export import board_to_step_native


def main():
    parser = argparse.ArgumentParser(description='Export KiCad flex PCB to STEP')
    parser.add_argument('input', help='Input .kicad_pcb file')
    parser.add_argument('output', help='Output .step file')
    parser.add_argument('--marker-layer', default='User.1',
                        help='Fold marker layer (default: User.1)')
    parser.add_argument('--stiffener-layer', default=None,
                        help='Stiffener layer (default: none)')
    parser.add_argument('--stiffener-thickness', type=float, default=0.2,
                        help='Stiffener thickness in mm (default: 0.2)')
    parser.add_argument('--stiffener-side', default='bottom',
                        choices=['top', 'bottom'],
                        help='Stiffener side (default: bottom)')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} not found")
        sys.exit(1)

    pcb = KiCadPCB.load(args.input)
    geom = extract_geometry(pcb)
    markers = detect_fold_markers(pcb, layer=args.marker_layer)
    print(f"Outline: {len(geom.outline.vertices)} verts, {len(geom.cutouts)} cutouts, {len(markers)} folds")

    config = FlexConfig()
    stiffeners = None
    if args.stiffener_layer:
        config.stiffener_thickness = args.stiffener_thickness
        if args.stiffener_side == 'top':
            config.stiffener_layer_top = args.stiffener_layer
        else:
            config.stiffener_layer_bottom = args.stiffener_layer
        stiffeners = extract_stiffeners(pcb, config)
        print(f"Stiffeners: {len(stiffeners) if stiffeners else 0} ({args.stiffener_side}, {args.stiffener_thickness}mm)")

    ok = board_to_step_native(geom, markers, args.output, pcb=pcb, config=config, stiffeners=stiffeners)
    if ok:
        size = os.path.getsize(args.output)
        print(f"Exported: {args.output} ({size:,} bytes)")
    else:
        print("ERROR: export failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
