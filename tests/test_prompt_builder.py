"""
tests/test_prompt_builder.py
Unit tests for RepairPromptBuilder.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from core.models import FeatureVector, QualityReport
from core.prompt_builder import RepairPromptBuilder

builder = RepairPromptBuilder()

SAMPLE_CODE = "def add(a, b):\n    return a + b\n"


def make_report(**kwargs) -> QualityReport:
    defaults = dict(
        overall_score=50.0,
        readability_score=50.0,
        maintainability_score=50.0,
        complexity_score=50.0,
        documentation_score=50.0,
        best_practices_score=50.0,
        grade="D",
        problems=[],
        problem_category="poor_documentation",
    )
    defaults.update(kwargs)
    return QualityReport(**defaults)


class TestPromptBuilder:

    def test_returns_string(self):
        prompt = builder.build(SAMPLE_CODE, make_report(), FeatureVector(), 1)
        assert isinstance(prompt, str)

    def test_contains_original_code(self):
        prompt = builder.build(SAMPLE_CODE, make_report(), FeatureVector(), 1)
        assert SAMPLE_CODE in prompt

    def test_contains_iteration_number(self):
        for n in (1, 3, 5):
            prompt = builder.build(SAMPLE_CODE, make_report(), FeatureVector(), n)
            assert str(n) in prompt

    def test_contains_score(self):
        report = make_report(overall_score=62.5, grade="C")
        prompt = builder.build(SAMPLE_CODE, report, FeatureVector(), 1)
        assert "62.5" in prompt

    def test_contains_grade(self):
        report = make_report(grade="B")
        prompt = builder.build(SAMPLE_CODE, report, FeatureVector(), 1)
        assert "B" in prompt

    def test_no_problems_shows_none_significant(self):
        report = make_report(problems=[])
        prompt = builder.build(SAMPLE_CODE, report, FeatureVector(), 1)
        assert "None significant" in prompt

    def test_problems_listed_in_prompt(self):
        problems = [
            {"severity": "HIGH", "type": "naming_issues",
             "description": "Bad names", "suggestion": "Fix them"}
        ]
        report = make_report(problems=problems)
        prompt = builder.build(SAMPLE_CODE, report, FeatureVector(), 1)
        assert "Bad names" in prompt
        assert "Fix them" in prompt

    def test_focus_instruction_per_category(self):
        for category, keyword in [
            ("complexity_overload",  "complexity"),
            ("poor_documentation",   "docstring"),
            ("naming_issues",        "Rename"),
            ("error_handling_gaps",  "exception"),
            ("style_violations",     "PEP8"),
            ("type_safety_issues",   "type"),
            ("clean_code",           "subtle"),
        ]:
            report = make_report(problem_category=category)
            prompt = builder.build(SAMPLE_CODE, report, FeatureVector(), 1)
            assert keyword.lower() in prompt.lower(), (
                f"Expected '{keyword}' in prompt for category '{category}'"
            )

    def test_prompt_ends_with_improved_code_marker(self):
        prompt = builder.build(SAMPLE_CODE, make_report(), FeatureVector(), 1)
        assert prompt.strip().endswith("IMPROVED CODE:")

    def test_unknown_category_has_fallback(self):
        report = make_report(problem_category="nonexistent_category")
        prompt = builder.build(SAMPLE_CODE, report, FeatureVector(), 1)
        assert "overall code quality" in prompt.lower()
