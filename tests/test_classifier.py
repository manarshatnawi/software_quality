"""
tests/test_classifier.py
Unit tests for ProblemClassifier — one test per rule.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from core.models import FeatureVector
from core.classifier import ProblemClassifier

clf = ProblemClassifier()


def _problem_types(problems):
    return [p["type"] for p in problems]

def _severities(problems, ptype):
    return [p["severity"] for p in problems if p["type"] == ptype]


# ── Complexity rules ──────────────────────────────────────────────────

class TestComplexityRules:

    def test_high_cyclomatic_flagged(self):
        fv = FeatureVector(cyclomatic_complexity=15.0)
        _, problems = clf.classify(fv)
        assert "complexity_overload" in _problem_types(problems)
        assert _severities(problems, "complexity_overload")[0] == "high"

    def test_low_cyclomatic_clean(self):
        fv = FeatureVector(cyclomatic_complexity=5.0)
        _, problems = clf.classify(fv)
        assert not any(p["type"] == "complexity_overload" and p["severity"] == "high"
                       for p in problems)

    def test_deep_nesting_flagged(self):
        fv = FeatureVector(max_nesting_depth=6)
        _, problems = clf.classify(fv)
        assert any(p["type"] == "complexity_overload" for p in problems)

    def test_long_function_flagged(self):
        fv = FeatureVector(max_function_lines=80)
        _, problems = clf.classify(fv)
        assert any(p["type"] == "complexity_overload" for p in problems)


# ── Documentation rules ───────────────────────────────────────────────

class TestDocumentationRules:

    def test_missing_module_docstring_flagged(self):
        fv = FeatureVector(has_module_docstring=False)
        _, problems = clf.classify(fv)
        assert any(p["type"] == "poor_documentation" for p in problems)

    def test_has_module_docstring_no_flag(self):
        fv = FeatureVector(
            has_module_docstring=True,
            docstring_coverage=1.0,
            comment_density=0.1,
            lines_of_code=10,
        )
        _, problems = clf.classify(fv)
        assert not any(p["type"] == "poor_documentation" for p in problems)

    def test_low_docstring_coverage_flagged(self):
        fv = FeatureVector(docstring_coverage=0.2, has_module_docstring=True)
        _, problems = clf.classify(fv)
        assert any(p["type"] == "poor_documentation" for p in problems)

    def test_low_comment_density_flagged(self):
        fv = FeatureVector(
            has_module_docstring=True,
            docstring_coverage=1.0,
            comment_density=0.01,
            lines_of_code=50,
        )
        _, problems = clf.classify(fv)
        assert any(p["type"] == "poor_documentation" for p in problems)

    def test_comment_density_below_loc_threshold_not_flagged(self):
        # comment density check is skipped when LOC < threshold
        fv = FeatureVector(
            has_module_docstring=True,
            docstring_coverage=1.0,
            comment_density=0.0,
            lines_of_code=5,
        )
        _, problems = clf.classify(fv)
        assert not any(
            p["type"] == "poor_documentation" and "comment" in p["description"].lower()
            for p in problems
        )


# ── Naming rules ──────────────────────────────────────────────────────

class TestNamingRules:

    def test_high_short_names_ratio_flagged(self):
        fv = FeatureVector(short_names_ratio=0.5)
        _, problems = clf.classify(fv)
        assert any(p["type"] == "naming_issues" for p in problems)

    def test_low_naming_convention_flagged(self):
        fv = FeatureVector(naming_convention_score=0.3)
        _, problems = clf.classify(fv)
        assert any(p["type"] == "naming_issues" for p in problems)

    def test_good_naming_no_flag(self):
        fv = FeatureVector(
            short_names_ratio=0.0,
            naming_convention_score=1.0,
            has_module_docstring=True,
            docstring_coverage=1.0,
        )
        _, problems = clf.classify(fv)
        assert not any(p["type"] == "naming_issues" for p in problems)


# ── Error handling rules ──────────────────────────────────────────────

class TestErrorHandlingRules:

    def test_bare_except_flagged_as_high(self):
        fv = FeatureVector(bare_except_count=1)
        _, problems = clf.classify(fv)
        assert any(
            p["type"] == "error_handling_gaps" and p["severity"] == "high"
            for p in problems
        )

    def test_multiple_bare_excepts(self):
        fv = FeatureVector(bare_except_count=3)
        _, problems = clf.classify(fv)
        assert any(p["type"] == "error_handling_gaps" for p in problems)

    def test_low_exception_coverage_flagged(self):
        fv = FeatureVector(num_functions=5, exception_coverage=0.1)
        _, problems = clf.classify(fv)
        assert any(p["type"] == "error_handling_gaps" for p in problems)

    def test_good_exception_coverage_no_flag(self):
        fv = FeatureVector(
            bare_except_count=0,
            num_functions=4,
            exception_coverage=0.8,
            has_module_docstring=True,
            docstring_coverage=1.0,
        )
        _, problems = clf.classify(fv)
        assert not any(p["type"] == "error_handling_gaps" for p in problems)


# ── Style rules ───────────────────────────────────────────────────────

class TestStyleRules:

    def test_long_lines_flagged(self):
        fv = FeatureVector(long_lines_ratio=0.3)
        _, problems = clf.classify(fv)
        assert any(p["type"] == "style_violations" for p in problems)

    def test_magic_numbers_flagged(self):
        fv = FeatureVector(magic_numbers_count=10)
        _, problems = clf.classify(fv)
        assert any(p["type"] == "style_violations" for p in problems)

    def test_global_vars_flagged(self):
        fv = FeatureVector(global_vars_count=5)
        _, problems = clf.classify(fv)
        assert any(p["type"] == "style_violations" for p in problems)

    def test_clean_style_no_flag(self):
        fv = FeatureVector(
            long_lines_ratio=0.0,
            magic_numbers_count=0,
            global_vars_count=0,
        )
        _, problems = clf.classify(fv)
        assert not any(p["type"] == "style_violations" for p in problems)


# ── Type safety rules ─────────────────────────────────────────────────

class TestTypeSafetyRules:

    def test_missing_type_hints_flagged(self):
        fv = FeatureVector(num_functions=3, type_hint_coverage=0.1)
        _, problems = clf.classify(fv)
        assert any(p["type"] == "type_safety_issues" for p in problems)

    def test_full_type_hints_no_flag(self):
        fv = FeatureVector(num_functions=3, type_hint_coverage=1.0)
        _, problems = clf.classify(fv)
        assert not any(p["type"] == "type_safety_issues" for p in problems)


# ── Category & clean code ─────────────────────────────────────────────

class TestPrimaryCategory:

    def test_clean_code_when_no_problems(self):
        fv = FeatureVector(
            has_module_docstring=True,
            docstring_coverage=1.0,
            comment_density=0.1,
            naming_convention_score=1.0,
            short_names_ratio=0.0,
            type_hint_coverage=1.0,
            num_functions=2,
            cyclomatic_complexity=3.0,
            max_nesting_depth=2,
            max_function_lines=20,
            bare_except_count=0,
            exception_coverage=0.8,
            long_lines_ratio=0.0,
            magic_numbers_count=0,
            global_vars_count=0,
        )
        category, problems = clf.classify(fv)
        assert problems == []
        assert category == "clean_code"

    def test_dominant_category_returned(self):
        # Three complexity problems, one documentation problem
        fv = FeatureVector(
            cyclomatic_complexity=20.0,
            max_nesting_depth=8,
            max_function_lines=100,
            has_module_docstring=False,
        )
        category, _ = clf.classify(fv)
        assert category == "complexity_overload"

    def test_category_in_known_list(self):
        fv = FeatureVector(bare_except_count=2)
        category, _ = clf.classify(fv)
        assert category in ProblemClassifier.CATEGORIES
