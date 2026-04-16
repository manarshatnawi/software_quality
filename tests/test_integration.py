"""
tests/test_integration.py
End-to-end pipeline tests that do NOT call the Groq API.
Tests the full analyze → classify → score → prompt chain.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from core.analyzer import ASTAnalyzer
from core.classifier import ProblemClassifier
from core.scorer import QualityScorer
from core.prompt_builder import RepairPromptBuilder


# ── Shared fixtures ───────────────────────────────────────────────────

BAD_CODE = '''
import requests, json, os

x = 10
y = 20

def calc(a, b, c, d, e):
    r = 0
    if a > 0:
        if b > 0:
            if c > 0:
                if d > 0:
                    for i in range(e):
                        if i % 2 == 0:
                            r = r + a * b
                        else:
                            r = r - c * d
                        if r > 9999:
                            r = 9999
    return r

def fetch(url):
    try:
        resp = requests.get(url)
        return resp.json()
    except:
        pass

def process(data):
    res = []
    for item in data:
        if item["type"] == 1:
            res.append(item["value"] * 3.14159)
        elif item["type"] == 2:
            res.append(item["value"] * 2.71828)
        else:
            res.append(0)
    return res

global_cache = {}

def save(k, v):
    global global_cache
    global_cache[k] = v
'''

GOOD_CODE = '''
"""Module for arithmetic utilities."""

from typing import Union

Number = Union[int, float]


def add(first: Number, second: Number) -> Number:
    """Return the sum of two numbers.

    Args:
        first: The first operand.
        second: The second operand.

    Returns:
        The arithmetic sum.
    """
    return first + second


def multiply(first: Number, second: Number) -> Number:
    """Return the product of two numbers."""
    return first * second


def squares(limit: int) -> list[int]:
    """Return a list of squares from 0 to limit-1."""
    return [x * x for x in range(limit)]
'''


# ── Bad code pipeline ─────────────────────────────────────────────────

class TestBadCodePipeline:

    def setup_method(self):
        fv              = ASTAnalyzer(BAD_CODE).build_feature_vector()
        cat, problems   = ProblemClassifier().classify(fv)
        self.fv         = fv
        self.category   = cat
        self.problems   = problems
        self.report     = QualityScorer().score(fv, problems)
        self.report.problem_category = cat

    def test_bad_code_has_problems(self):
        assert len(self.problems) > 0

    def test_bad_code_grade_not_A(self):
        assert self.report.grade != "A"

    def test_bare_except_detected(self):
        assert self.fv.bare_except_count >= 1

    def test_magic_numbers_detected(self):
        assert self.fv.magic_numbers_count > 0

    def test_global_vars_detected(self):
        assert self.fv.global_vars_count >= 1

    def test_short_names_detected(self):
        assert self.fv.short_names_ratio > 0.0

    def test_no_type_hints(self):
        assert self.fv.uses_type_hints is False

    def test_overall_score_below_60(self):
        assert self.report.overall_score < 60.0

    def test_primary_category_known(self):
        assert self.category in ProblemClassifier.CATEGORIES

    def test_prompt_generated_without_error(self):
        prompt = RepairPromptBuilder().build(BAD_CODE, self.report, self.fv, 1)
        assert len(prompt) > 200
        assert BAD_CODE in prompt


# ── Good code pipeline ────────────────────────────────────────────────

class TestGoodCodePipeline:

    def setup_method(self):
        fv              = ASTAnalyzer(GOOD_CODE).build_feature_vector()
        cat, problems   = ProblemClassifier().classify(fv)
        self.fv         = fv
        self.category   = cat
        self.problems   = problems
        self.report     = QualityScorer().score(fv, problems)
        self.report.problem_category = cat

    def test_good_code_has_fewer_problems_than_bad(self):
        bad_fv            = ASTAnalyzer(BAD_CODE).build_feature_vector()
        _, bad_problems   = ProblemClassifier().classify(bad_fv)
        assert len(self.problems) < len(bad_problems)

    def test_good_code_higher_score_than_bad(self):
        bad_fv            = ASTAnalyzer(BAD_CODE).build_feature_vector()
        _, bad_problems   = ProblemClassifier().classify(bad_fv)
        bad_report        = QualityScorer().score(bad_fv, bad_problems)
        assert self.report.overall_score > bad_report.overall_score

    def test_has_module_docstring(self):
        assert self.fv.has_module_docstring is True

    def test_type_hints_detected(self):
        assert self.fv.uses_type_hints is True

    def test_list_comp_detected(self):
        assert self.fv.uses_list_comp is True

    def test_no_bare_excepts(self):
        assert self.fv.bare_except_count == 0

    def test_no_global_vars(self):
        assert self.fv.global_vars_count == 0


# ── Score monotonicity ────────────────────────────────────────────────

class TestScoreMonotonicity:
    """Adding more problems should never increase the score."""

    def _score_for(self, n_high: int, n_medium: int, n_low: int) -> float:
        fv       = ASTAnalyzer(BAD_CODE).build_feature_vector()
        problems = (
            [{"severity": "high"}]   * n_high   +
            [{"severity": "medium"}] * n_medium +
            [{"severity": "low"}]    * n_low
        )
        return QualityScorer().score(fv, problems).overall_score

    def test_more_high_problems_lower_score(self):
        assert self._score_for(1, 0, 0) >= self._score_for(2, 0, 0)

    def test_high_worse_than_low(self):
        assert self._score_for(1, 0, 0) < self._score_for(0, 0, 1)


# ── Edge cases ────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_empty_string_does_not_crash(self):
        fv = ASTAnalyzer("").build_feature_vector()
        _, problems = ProblemClassifier().classify(fv)
        report = QualityScorer().score(fv, problems)
        assert isinstance(report.overall_score, float)

    def test_single_expression_does_not_crash(self):
        fv = ASTAnalyzer("1 + 1").build_feature_vector()
        _, problems = ProblemClassifier().classify(fv)
        report = QualityScorer().score(fv, problems)
        assert report.overall_score >= 0

    def test_only_comments_does_not_crash(self):
        code = "# just a comment\n# another comment\n"
        fv = ASTAnalyzer(code).build_feature_vector()
        assert fv.lines_of_code == 2
