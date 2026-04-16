"""
tests/test_analyzer.py
Unit tests for ASTAnalyzer — verifies every metric bucket.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from core.analyzer import ASTAnalyzer


# ── Helpers ───────────────────────────────────────────────────────────

def analyze(source: str):
    return ASTAnalyzer(source).build_feature_vector()


# ── Lines of code ─────────────────────────────────────────────────────

class TestLinesOfCode:

    def test_single_line(self):
        assert analyze("x = 1").lines_of_code == 1

    def test_multi_line(self):
        code = "a = 1\nb = 2\nc = 3\n"
        assert analyze(code).lines_of_code == 3

    def test_empty_string(self):
        fv = analyze("")
        assert fv.lines_of_code == 0


# ── Function detection ────────────────────────────────────────────────

class TestFunctionDetection:

    def test_no_functions(self):
        assert analyze("x = 1").num_functions == 0

    def test_single_function(self):
        code = "def foo(): pass"
        assert analyze(code).num_functions == 1

    def test_multiple_functions(self):
        code = "def a(): pass\ndef b(): pass\ndef c(): pass"
        assert analyze(code).num_functions == 3

    def test_async_function_counted(self):
        code = "async def fetch(): pass"
        assert analyze(code).num_functions == 1

    def test_nested_function_counted(self):
        code = "def outer():\n    def inner(): pass"
        assert analyze(code).num_functions == 2


# ── Class detection ───────────────────────────────────────────────────

class TestClassDetection:

    def test_no_classes(self):
        assert analyze("x = 1").num_classes == 0

    def test_single_class(self):
        code = "class Foo:\n    pass"
        assert analyze(code).num_classes == 1

    def test_multiple_classes(self):
        code = "class A: pass\nclass B: pass"
        assert analyze(code).num_classes == 2


# ── Cyclomatic complexity ─────────────────────────────────────────────

class TestCyclomaticComplexity:

    def test_trivial_code(self):
        # 1 (base) + 0 branches = 1
        fv = analyze("x = 1")
        assert fv.cyclomatic_complexity == 1.0

    def test_one_if_branch(self):
        code = "if True:\n    pass"
        fv = analyze(code)
        assert fv.cyclomatic_complexity == 2.0   # 1 + 1 branch

    def test_for_loop_adds_branch(self):
        code = "for i in range(10):\n    pass"
        fv = analyze(code)
        assert fv.cyclomatic_complexity == 2.0

    def test_nested_branches(self):
        code = (
            "if a:\n"
            "    if b:\n"
            "        pass\n"
            "    else:\n"
            "        pass\n"
        )
        fv = analyze(code)
        # 1 base + if(a) + if(b) + else = 4
        assert fv.cyclomatic_complexity >= 3


# ── Nesting depth ─────────────────────────────────────────────────────

class TestNestingDepth:

    def test_no_nesting(self):
        fv = analyze("x = 1")
        assert fv.max_nesting_depth == 0

    def test_function_depth(self):
        code = "def foo():\n    pass"
        fv = analyze(code)
        assert fv.max_nesting_depth >= 1

    def test_deep_nesting(self):
        code = (
            "def f():\n"
            "    if True:\n"
            "        for i in range(10):\n"
            "            if i:\n"
            "                pass\n"
        )
        fv = analyze(code)
        assert fv.max_nesting_depth >= 3


# ── Documentation ─────────────────────────────────────────────────────

class TestDocumentation:

    def test_no_module_docstring(self):
        fv = analyze("x = 1")
        assert fv.has_module_docstring is False

    def test_has_module_docstring(self):
        code = '"""Module docstring."""\nx = 1'
        fv = analyze(code)
        assert fv.has_module_docstring is True

    def test_docstring_coverage_full(self):
        code = (
            'def foo():\n'
            '    """Docstring."""\n'
            '    pass\n'
        )
        fv = analyze(code)
        assert fv.docstring_coverage == 1.0

    def test_docstring_coverage_partial(self):
        code = (
            "def foo():\n"
            "    pass\n"
            "def bar():\n"
            '    """Bar."""\n'
            "    pass\n"
        )
        fv = analyze(code)
        assert fv.docstring_coverage == pytest.approx(0.5)

    def test_no_functions_coverage_is_full(self):
        # Edge case: no functions → coverage defaults to 1.0
        fv = analyze("x = 1")
        assert fv.docstring_coverage == 1.0

    def test_comment_density(self):
        code = "# comment\nx = 1\n# another\ny = 2\n"
        fv = analyze(code)
        assert fv.comment_density == pytest.approx(0.5)


# ── Naming ────────────────────────────────────────────────────────────

class TestNaming:

    def test_descriptive_names(self):
        code = "total_count = 0\nuser_name = 'Alice'\n"
        fv = analyze(code)
        assert fv.naming_convention_score > 0.5

    def test_short_names_detected(self):
        code = "a = 1\nb = 2\nc = 3\nd = 4\n"
        fv = analyze(code)
        assert fv.short_names_ratio > 0.0


# ── Error handling ────────────────────────────────────────────────────

class TestErrorHandling:

    def test_no_try_blocks(self):
        fv = analyze("x = 1")
        assert fv.try_except_count == 0
        assert fv.bare_except_count == 0

    def test_specific_except(self):
        code = (
            "try:\n"
            "    x = 1\n"
            "except ValueError:\n"
            "    pass\n"
        )
        fv = analyze(code)
        assert fv.try_except_count == 1
        assert fv.bare_except_count == 0

    def test_bare_except_detected(self):
        code = (
            "try:\n"
            "    x = 1\n"
            "except:\n"
            "    pass\n"
        )
        fv = analyze(code)
        assert fv.bare_except_count == 1

    def test_multiple_bare_excepts(self):
        code = (
            "try:\n    pass\nexcept:\n    pass\n"
            "try:\n    pass\nexcept:\n    pass\n"
        )
        fv = analyze(code)
        assert fv.bare_except_count == 2


# ── Magic numbers ─────────────────────────────────────────────────────

class TestMagicNumbers:

    def test_no_magic_numbers(self):
        code = "x = 0\ny = 1\n"
        fv = analyze(code)
        assert fv.magic_numbers_count == 0   # 0 and 1 are whitelisted

    def test_magic_number_detected(self):
        code = "x = 42\n"
        fv = analyze(code)
        assert fv.magic_numbers_count >= 1

    def test_multiple_magic_numbers(self):
        code = "a = 3.14\nb = 2.71\nc = 9999\n"
        fv = analyze(code)
        assert fv.magic_numbers_count >= 3


# ── Type hints ────────────────────────────────────────────────────────

class TestTypeHints:

    def test_no_type_hints(self):
        code = "def add(a, b):\n    return a + b\n"
        fv = analyze(code)
        assert fv.uses_type_hints is False
        assert fv.type_hint_coverage == 0.0

    def test_return_annotation(self):
        code = "def greet() -> str:\n    return 'hi'\n"
        fv = analyze(code)
        assert fv.uses_type_hints is True
        assert fv.type_hint_coverage == 1.0

    def test_arg_annotation(self):
        code = "def add(a: int, b: int):\n    return a + b\n"
        fv = analyze(code)
        assert fv.uses_type_hints is True


# ── List comp / generators ────────────────────────────────────────────

class TestPythonic:

    def test_list_comp_detected(self):
        code = "result = [x * 2 for x in range(10)]\n"
        fv = analyze(code)
        assert fv.uses_list_comp is True

    def test_generator_detected(self):
        code = "total = sum(x for x in range(10))\n"
        fv = analyze(code)
        assert fv.uses_generators is True

    def test_plain_loop_not_detected(self):
        code = "result = []\nfor x in range(10):\n    result.append(x)\n"
        fv = analyze(code)
        assert fv.uses_list_comp is False
        assert fv.uses_generators is False


# ── Long lines ────────────────────────────────────────────────────────

class TestLongLines:

    def test_no_long_lines(self):
        code = "x = 1\ny = 2\n"
        fv = analyze(code)
        assert fv.long_lines_ratio == 0.0

    def test_long_line_detected(self):
        long_line = "x = " + "a" * 80 + "\n"
        fv = analyze(long_line)
        assert fv.long_lines_ratio > 0.0


# ── Syntax errors ─────────────────────────────────────────────────────

class TestSyntaxErrors:

    def test_syntax_error_raises(self):
        with pytest.raises(SyntaxError):
            analyze("def broken(:\n    pass")

    def test_valid_code_does_not_raise(self):
        analyze("def ok(): pass")   # Should not raise
