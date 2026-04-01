"""Integration tests for KiCad 10 format compatibility."""

import pytest
from pathlib import Path

# Add plugin dir to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "plugins" / "com_github_aightech_flexviz"))

from kicad_parser import KiCadPCB
from geometry import extract_geometry
from markers import detect_fold_markers


@pytest.fixture
def test_data_dir():
    return Path(__file__).parent / "test_data"


class TestKiCad10Integration:
    """End-to-end tests with KiCad 10 format files."""

    def test_parse_h_shape(self, test_data_dir):
        """h_shape.kicad_pcb (re-saved by KiCad 10) parses successfully."""
        pcb = KiCadPCB.load(str(test_data_dir / "h_shape.kicad_pcb"))
        assert pcb is not None

    def test_h_shape_geometry(self, test_data_dir):
        """Extract geometry from KiCad 10 format h_shape."""
        pcb = KiCadPCB.load(str(test_data_dir / "h_shape.kicad_pcb"))
        geom = extract_geometry(pcb)
        assert geom.outline is not None
        assert len(geom.outline.vertices) > 3  # H-shape has many vertices
        assert geom.thickness > 0

    def test_h_shape_markers(self, test_data_dir):
        """Detect fold markers from KiCad 10 format h_shape."""
        pcb = KiCadPCB.load(str(test_data_dir / "h_shape.kicad_pcb"))
        markers = detect_fold_markers(pcb, "User.1")
        assert len(markers) == 8  # H-shape has 8 fold markers

    def test_h_shape_traces(self, test_data_dir):
        """Verify traces extracted from KiCad 10 h_shape."""
        pcb = KiCadPCB.load(str(test_data_dir / "h_shape.kicad_pcb"))
        geom = extract_geometry(pcb)
        total_traces = sum(len(v) for v in geom.traces.values())
        assert total_traces >= 7  # H-shape has at least 7 traces

    def test_neubondv4_simple_parses(self, test_data_dir):
        """neubondV4_simple.kicad_pcb parses successfully."""
        pcb_path = test_data_dir / "neubondV4_simple.kicad_pcb"
        if not pcb_path.exists():
            pytest.skip("neubondV4_simple.kicad_pcb not available")
        pcb = KiCadPCB.load(str(pcb_path))
        assert pcb is not None
        geom = extract_geometry(pcb)
        assert geom.outline is not None
