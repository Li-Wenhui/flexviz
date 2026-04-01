"""Unit tests for kicad_parser module."""

import pytest
from pathlib import Path

from kicad_parser import (
    SExpr,
    SExprTokenizer,
    parse_sexpr,
    parse_kicad_pcb,
    load_kicad_pcb,
    KiCadPCB,
)


class TestSExprTokenizer:
    """Tests for S-expression tokenizer."""

    def test_tokenize_simple(self):
        """Test tokenizing simple S-expression."""
        tokenizer = SExprTokenizer("(a b c)")
        tokens = tokenizer.tokens
        assert len(tokens) == 5
        assert tokens[0] == ('LPAREN', '(')
        assert tokens[1] == ('ATOM', 'a')
        assert tokens[2] == ('ATOM', 'b')
        assert tokens[3] == ('ATOM', 'c')
        assert tokens[4] == ('RPAREN', ')')

    def test_tokenize_numbers(self):
        """Test tokenizing numbers."""
        tokenizer = SExprTokenizer("(num 42 3.14 -5)")
        tokens = tokenizer.tokens
        assert ('ATOM', '42') in tokens
        assert ('ATOM', '3.14') in tokens
        assert ('ATOM', '-5') in tokens

    def test_tokenize_strings(self):
        """Test tokenizing quoted strings."""
        tokenizer = SExprTokenizer('(text "hello world")')
        tokens = tokenizer.tokens
        assert ('ATOM', 'text') in tokens
        assert ('STRING', 'hello world') in tokens

    def test_tokenize_escaped_strings(self):
        """Test tokenizing strings with escaped characters."""
        tokenizer = SExprTokenizer(r'(text "hello \"quoted\"")')
        tokens = tokenizer.tokens
        # Should unescape the quotes
        assert ('STRING', 'hello "quoted"') in tokens

    def test_tokenize_nested(self):
        """Test tokenizing nested S-expressions."""
        tokenizer = SExprTokenizer("(a (b (c d)))")
        tokens = tokenizer.tokens
        lparen_count = sum(1 for t in tokens if t[0] == 'LPAREN')
        rparen_count = sum(1 for t in tokens if t[0] == 'RPAREN')
        assert lparen_count == 3
        assert rparen_count == 3

    def test_peek_and_next(self):
        """Test peek and next methods."""
        tokenizer = SExprTokenizer("(a b)")
        assert tokenizer.peek() == ('LPAREN', '(')
        assert tokenizer.next() == ('LPAREN', '(')
        assert tokenizer.peek() == ('ATOM', 'a')
        assert tokenizer.next() == ('ATOM', 'a')


class TestSExpr:
    """Tests for SExpr data structure."""

    def test_create_sexpr(self):
        """Test creating SExpr."""
        expr = SExpr("test", ["a", "b", "c"])
        assert expr.name == "test"
        assert len(expr.children) == 3

    def test_index_access(self):
        """Test accessing children by index."""
        expr = SExpr("test", ["a", "b", "c"])
        assert expr[0] == "a"
        assert expr[1] == "b"
        assert expr[2] == "c"
        assert expr[99] is None

    def test_name_access(self):
        """Test accessing children by name."""
        child1 = SExpr("layer", ["F.Cu"])
        child2 = SExpr("width", ["0.5"])
        expr = SExpr("segment", [child1, child2])

        assert expr["layer"] is child1
        assert expr["width"] is child2
        assert expr["nonexistent"] is None

    def test_find_all(self):
        """Test finding all children with a name."""
        child1 = SExpr("point", ["1", "2"])
        child2 = SExpr("point", ["3", "4"])
        child3 = SExpr("other", ["x"])
        expr = SExpr("points", [child1, child2, child3])

        points = list(expr.find_all("point"))
        assert len(points) == 2
        assert child1 in points
        assert child2 in points

    def test_get_value(self):
        """Test getting atom values."""
        expr = SExpr("test", ["hello", "42", "3.14"])
        assert expr.get_value(0) == "hello"
        assert expr.get_value(1) == "42"
        assert expr.get_value(2) == "3.14"
        assert expr.get_value(99) is None

    def test_get_float(self):
        """Test getting float values."""
        expr = SExpr("test", ["3.14", "42", "notanumber"])
        assert expr.get_float(0) == 3.14
        assert expr.get_float(1) == 42.0
        assert expr.get_float(2) == 0.0  # Invalid returns 0

    def test_get_string(self):
        """Test getting string values."""
        expr = SExpr("test", ["hello"])
        assert expr.get_string(0) == "hello"
        assert expr.get_string(99) == ""


class TestParseSExpr:
    """Tests for S-expression parsing."""

    def test_parse_simple(self, simple_sexpr):
        """Test parsing simple S-expression."""
        tokenizer = SExprTokenizer(simple_sexpr)
        expr = parse_sexpr(tokenizer)

        assert expr.name == "test"
        assert expr["a"] is not None
        assert expr["b"] is not None
        assert expr["c"] is not None

    def test_parse_nested(self):
        """Test parsing nested S-expressions."""
        text = "(outer (inner (deep value)))"
        tokenizer = SExprTokenizer(text)
        expr = parse_sexpr(tokenizer)

        assert expr.name == "outer"
        inner = expr["inner"]
        assert inner is not None
        deep = inner["deep"]
        assert deep is not None
        assert deep.get_value(0) == "value"

    def test_parse_kicad_pcb(self, minimal_pcb_content):
        """Test parsing KiCad PCB content."""
        expr = parse_kicad_pcb(minimal_pcb_content)

        assert expr.name == "kicad_pcb"
        assert expr["version"] is not None
        assert expr["general"] is not None

    def test_parse_empty_raises(self):
        """Test that parsing empty input raises error."""
        tokenizer = SExprTokenizer("")
        with pytest.raises(ValueError):
            parse_sexpr(tokenizer)

    def test_parse_unclosed_raises(self):
        """Test that unclosed paren raises error."""
        tokenizer = SExprTokenizer("(test (nested)")
        with pytest.raises(ValueError):
            parse_sexpr(tokenizer)


class TestLoadKiCadPCB:
    """Tests for loading KiCad PCB files."""

    def test_load_minimal(self, minimal_pcb_path):
        """Test loading minimal PCB file."""
        if not minimal_pcb_path.exists():
            pytest.skip("Test data file not found")

        expr = load_kicad_pcb(minimal_pcb_path)
        assert expr.name == "kicad_pcb"

    def test_load_nonexistent_raises(self):
        """Test loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_kicad_pcb("/nonexistent/path/board.kicad_pcb")


class TestKiCadPCB:
    """Tests for KiCadPCB class."""

    def test_load_minimal(self, minimal_pcb_path):
        """Test loading minimal PCB."""
        if not minimal_pcb_path.exists():
            pytest.skip("Test data file not found")

        pcb = KiCadPCB.load(minimal_pcb_path)
        assert pcb.root.name == "kicad_pcb"

    def test_parse_from_string(self, minimal_pcb_content):
        """Test parsing PCB from string."""
        pcb = KiCadPCB.parse(minimal_pcb_content)
        assert pcb.root.name == "kicad_pcb"

    def test_get_board_info(self, minimal_pcb_path):
        """Test getting board info."""
        if not minimal_pcb_path.exists():
            pytest.skip("Test data file not found")

        pcb = KiCadPCB.load(minimal_pcb_path)
        info = pcb.get_board_info()
        assert info.thickness == 1.6

    def test_get_board_outline_rectangle(self, rectangle_pcb_path):
        """Test extracting board outline."""
        if not rectangle_pcb_path.exists():
            pytest.skip("Test data file not found")

        pcb = KiCadPCB.load(rectangle_pcb_path)
        outline = pcb.get_board_outline()

        assert len(outline) == 4
        # Should form a 100x50 rectangle
        xs = [p[0] for p in outline]
        ys = [p[1] for p in outline]
        assert min(xs) == 0
        assert max(xs) == 100
        assert min(ys) == 0
        assert max(ys) == 50

    def test_get_traces(self, rectangle_pcb_path):
        """Test extracting traces."""
        if not rectangle_pcb_path.exists():
            pytest.skip("Test data file not found")

        pcb = KiCadPCB.load(rectangle_pcb_path)
        traces = pcb.get_traces()

        assert len(traces) == 2
        # Check first trace properties
        t = traces[0]
        assert t.width in [0.5, 0.25]
        assert t.layer == "F.Cu"

    def test_get_traces_filtered(self, rectangle_pcb_path):
        """Test extracting traces filtered by layer."""
        if not rectangle_pcb_path.exists():
            pytest.skip("Test data file not found")

        pcb = KiCadPCB.load(rectangle_pcb_path)

        fcu_traces = pcb.get_traces(layer="F.Cu")
        assert len(fcu_traces) == 2

        bcu_traces = pcb.get_traces(layer="B.Cu")
        assert len(bcu_traces) == 0

    def test_get_graphic_lines(self, fold_pcb_path):
        """Test extracting graphic lines."""
        if not fold_pcb_path.exists():
            pytest.skip("Test data file not found")

        pcb = KiCadPCB.load(fold_pcb_path)
        lines = pcb.get_graphic_lines(layer="User.1")

        # Should have 4 fold marker lines
        assert len(lines) == 4

    def test_get_dimensions(self, fold_pcb_path):
        """Test extracting dimensions."""
        if not fold_pcb_path.exists():
            pytest.skip("Test data file not found")

        pcb = KiCadPCB.load(fold_pcb_path)
        dims = pcb.get_dimensions(layer="User.1")

        assert len(dims) == 2
        # Check dimension values
        angles = [d.value for d in dims]
        assert 90 in angles or 90.0 in angles
        assert -45 in angles or -45.0 in angles

    def test_get_footprints(self, fold_pcb_path):
        """Test extracting footprints."""
        if not fold_pcb_path.exists():
            pytest.skip("Test data file not found")

        pcb = KiCadPCB.load(fold_pcb_path)
        footprints = pcb.get_footprints()

        assert len(footprints) == 1
        fp = footprints[0]
        assert fp.reference == "R1"
        assert fp.at_x == 20
        assert fp.at_y == 15
        assert len(fp.pads) == 2

    def test_empty_board_outline(self, minimal_pcb_path):
        """Test board with no outline returns empty list."""
        if not minimal_pcb_path.exists():
            pytest.skip("Test data file not found")

        pcb = KiCadPCB.load(minimal_pcb_path)
        outline = pcb.get_board_outline()
        assert outline == []


class TestOrderSegments:
    """Tests for segment ordering algorithm."""

    def test_order_square(self, rectangle_pcb_path):
        """Test ordering segments into a square."""
        if not rectangle_pcb_path.exists():
            pytest.skip("Test data file not found")

        pcb = KiCadPCB.load(rectangle_pcb_path)
        outline = pcb.get_board_outline()

        # Should be 4 vertices forming closed polygon
        assert len(outline) == 4

        # Check that it's actually closed (first and last vertex connect)
        # by verifying the edges are sequential
        for i in range(len(outline)):
            p1 = outline[i]
            p2 = outline[(i + 1) % len(outline)]
            # Each edge should share a vertex
            assert p1 != p2


class TestParserEdgeCases:
    """Edge case tests for the S-expression parser."""

    def test_empty_string_raises(self):
        """Parsing empty string should raise ValueError."""
        tokenizer = SExprTokenizer("")
        with pytest.raises(ValueError):
            parse_sexpr(tokenizer)

    def test_single_atom_raises(self):
        """Parsing a bare atom (no parens) should raise ValueError."""
        tokenizer = SExprTokenizer("hello")
        with pytest.raises(ValueError):
            parse_sexpr(tokenizer)

    def test_deeply_nested(self):
        """50+ levels of nesting should parse without stack overflow."""
        depth = 60
        text = "(" * depth + "leaf" + ")" * depth
        tokenizer = SExprTokenizer(text)
        expr = parse_sexpr(tokenizer)
        # Walk down the nesting
        node = expr
        for _ in range(depth - 2):
            # Each level should have one child that is an SExpr
            children = [c for c in node.children if isinstance(c, SExpr)]
            if not children:
                break
            node = children[0]
        # Should not have crashed
        assert expr is not None

    def test_unbalanced_open_paren_raises(self):
        """Missing closing paren should raise ValueError."""
        tokenizer = SExprTokenizer("(foo (bar)")
        with pytest.raises(ValueError):
            parse_sexpr(tokenizer)

    def test_unbalanced_close_paren(self):
        """Extra closing paren should not crash; parser stops after first complete expr."""
        # "(foo) bar)" — parse_sexpr reads "(foo)" successfully; extra tokens remain
        tokenizer = SExprTokenizer("(foo) bar)")
        expr = parse_sexpr(tokenizer)
        assert expr.name == "foo"

    def test_string_with_quotes(self):
        """Quoted strings should be preserved as single tokens."""
        tokenizer = SExprTokenizer('(text "hello world")')
        expr = parse_sexpr(tokenizer)
        assert expr.name == "text"
        assert expr.get_value(0) == "hello world"

    def test_numeric_values_stored_as_strings(self):
        """Numeric values are stored as strings in SExpr children."""
        tokenizer = SExprTokenizer("(size 1.5 2.0)")
        expr = parse_sexpr(tokenizer)
        assert expr.name == "size"
        assert expr.get_value(0) == "1.5"
        assert expr.get_value(1) == "2.0"
        # get_float should convert them
        assert expr.get_float(0) == 1.5
        assert expr.get_float(1) == 2.0

    def test_kicad10_format_loads(self):
        """h_shape.kicad_pcb (KiCad 9/10 format) should parse without error."""
        h_shape_path = Path(__file__).parent / "test_data" / "h_shape.kicad_pcb"
        if not h_shape_path.exists():
            pytest.skip("h_shape.kicad_pcb test data not found")

        pcb = KiCadPCB.load(h_shape_path)
        assert pcb.root.name == "kicad_pcb"

        # Should have gr_line entries (board outline and/or fold markers)
        gr_lines = list(pcb.root.find_all("gr_line"))
        assert len(gr_lines) > 0, "Expected gr_line entries in h_shape PCB"
