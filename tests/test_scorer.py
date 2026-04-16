"""
tests/test_scorer.py
Unit tests for QualityScorer — grade boundaries, dimension scores, penalties.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from core.models import FeatureVector
from core.scorer import QualityScorer
from config import WEIGHTS

scorer = QualityScorer()


def score(fv=None, problems=None):
    return scorer.score(fv or FeatureVector(), problems or [])


# ── Grade boundaries ──────────────────────────────────────────────────

class TestGradeBoundaries:

    def test_worst_code_gets_low_grade(self):
        # A deliberately terrible FeatureVector should score below grade B
        fv = FeatureVector(
            cyclomatic_complexity   = 50.0,
            cognitive_complexity    = 80.0,
            max_nesting_depth       = 10,
            short_names_ratio       = 1.0,
            descriptive_names_ratio = 0.0,
            naming_convention_score = 0.0,
            long_lines_ratio        = 1.0,
            has_module_docstring    = False,
            docstring_coverage      = 0.0,
            comment_density         = 0.0,
            bare_except_count       = 10,
            magic_numbers_count     = 20,
            global_vars_count       = 10,
            uses_type_hints         = False,
        )
        problems = [{"severity": "high"}] * 10
        report = score(fv, problems)
        assert report.grade in ("D", "F")
        assert report.overall_score < 40.0

    def test_grade_A_for_perfect_feature_vector(self):
        fv = FeatureVector(
            cyclomatic_complexity   = 1.0,
            cognitive_complexity    = 0.0,
            max_nesting_depth       = 0,
            short_names_ratio       = 0.0,
            descriptive_names_ratio = 1.0,
            naming_convention_score = 1.0,
            long_lines_ratio        = 0.0,
            has_module_docstring    = True,
            docstring_coverage      = 1.0,
            comment_density         = 0.15,
            bare_except_count       = 0,
            magic_numbers_count     = 0,
            global_vars_count       = 0,
            uses_type_hints         = True,
            type_hint_coverage      = 1.0,
            uses_list_comp          = True,
        )
        report = score(fv)
        assert report.grade == "A"
        assert report.overall_score >= WEIGHTS.grade_a

    def test_high_severity_penalty_lowers_score(self):
        fv = FeatureVector(
            naming_convention_score = 1.0,
            has_module_docstring    = True,
            docstring_coverage      = 1.0,
        )
        clean   = score(fv, [])
        penalised = score(fv, [{"severity": "high"}, {"severity": "high"}])
        assert penalised.overall_score < clean.overall_score

    def test_score_never_below_zero(self):
        fv = FeatureVector(cyclomatic_complexity=999.0, max_nesting_depth=999)
        problems = [{"severity": "high"}] * 20
        report = score(fv, problems)
        assert report.overall_score >= 0.0

    def test_score_never_above_100(self):
        fv = FeatureVector(
            cyclomatic_complexity   = 1.0,
            has_module_docstring    = True,
            docstring_coverage      = 1.0,
            comment_density         = 0.5,
            naming_convention_score = 1.0,
            uses_type_hints         = True,
        )
        report = score(fv, [])
        assert report.overall_score <= 100.0


# ── Dimension scoring ─────────────────────────────────────────────────

class TestDimensionScores:

    def test_complexity_score_decreases_with_cyclomatic(self):
        low  = score(FeatureVector(cyclomatic_complexity=2.0))
        high = score(FeatureVector(cyclomatic_complexity=15.0))
        assert low.complexity_score > high.complexity_score

    def test_readability_improves_with_good_naming(self):
        bad  = score(FeatureVector(naming_convention_score=0.0, short_names_ratio=0.8))
        good = score(FeatureVector(naming_convention_score=1.0, short_names_ratio=0.0,
                                   descriptive_names_ratio=1.0))
        assert good.readability_score > bad.readability_score

    def test_documentation_score_with_full_docstrings(self):
        fv = FeatureVector(
            has_module_docstring=True,
            docstring_coverage=1.0,
            comment_density=0.15,
        )
        report = score(fv)
        assert report.documentation_score >= 70.0

    def test_best_practices_improved_by_type_hints_and_listcomp(self):
        without = score(FeatureVector(uses_type_hints=False, uses_list_comp=False))
        with_   = score(FeatureVector(uses_type_hints=True,  uses_list_comp=True))
        assert with_.best_practices_score > without.best_practices_score

    def test_bare_except_penalises_best_practices(self):
        clean   = score(FeatureVector(bare_except_count=0))
        messy   = score(FeatureVector(bare_except_count=5))
        assert clean.best_practices_score > messy.best_practices_score


# ── Severity penalties ────────────────────────────────────────────────

class TestSeverityPenalties:

    def _delta(self, severity: str, count: int = 1):
        fv = FeatureVector()
        base     = score(fv, []).overall_score
        penalised = score(fv, [{"severity": severity}] * count).overall_score
        return base - penalised

    def test_high_penalty_larger_than_medium(self):
        assert self._delta("high") > self._delta("medium")

    def test_medium_penalty_larger_than_low(self):
        assert self._delta("medium") > self._delta("low")

    def test_penalties_are_additive(self):
        one   = self._delta("high", 1)
        three = self._delta("high", 3)
        assert abs(three - one * 3) < 0.01   # strictly linear


# ── Problems stored ───────────────────────────────────────────────────

class TestProblemsStored:

    def test_problems_passed_through(self):
        problems = [{"severity": "low", "type": "style_violations"}]
        report = score(FeatureVector(), problems)
        assert report.problems == problems

    def test_no_problems_gives_empty_list(self):
        report = score(FeatureVector(), [])
        assert report.problems == []


# ── Rounding ─────────────────────────────────────────────────────────

class TestRounding:

    def test_overall_score_is_rounded(self):
        report = score(FeatureVector(cyclomatic_complexity=3.5))
        # should be rounded to 2 decimal places
        assert round(report.overall_score, 2) == report.overall_score
