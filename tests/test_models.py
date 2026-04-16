"""
tests/test_models.py
Unit tests for FeatureVector, QualityReport, and IterationRecord.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from core.models import FeatureVector, QualityReport, IterationRecord


class TestFeatureVector:

    def test_default_values(self):
        fv = FeatureVector()
        assert fv.cyclomatic_complexity == 0.0
        assert fv.lines_of_code == 0
        assert fv.has_module_docstring is False
        assert fv.uses_type_hints is False
        assert fv.uses_list_comp is False
        assert fv.uses_generators is False

    def test_custom_values(self):
        fv = FeatureVector(
            cyclomatic_complexity=5.0,
            lines_of_code=100,
            num_functions=3,
            has_module_docstring=True,
        )
        assert fv.cyclomatic_complexity == 5.0
        assert fv.lines_of_code == 100
        assert fv.num_functions == 3
        assert fv.has_module_docstring is True

    def test_problems_default_empty(self):
        # Two separate instances must not share the same list
        fv1 = FeatureVector()
        fv2 = FeatureVector()
        assert fv1 is not fv2


class TestQualityReport:

    def test_default_grade_is_F(self):
        report = QualityReport()
        assert report.grade == "F"
        assert report.overall_score == 0.0

    def test_problems_list_is_independent(self):
        r1 = QualityReport()
        r2 = QualityReport()
        r1.problems.append({"type": "test"})
        assert len(r2.problems) == 0, "Shared mutable default detected"

    def test_all_score_fields_present(self):
        r = QualityReport(
            overall_score=75.0,
            readability_score=80.0,
            maintainability_score=70.0,
            complexity_score=65.0,
            documentation_score=60.0,
            best_practices_score=55.0,
            grade="B",
        )
        assert r.overall_score == 75.0
        assert r.grade == "B"


class TestIterationRecord:

    def test_basic_construction(self):
        fv  = FeatureVector(lines_of_code=10)
        rep = QualityReport(overall_score=50.0, grade="D")
        rec = IterationRecord(iteration=1, code="x = 1", feature_vector=fv, quality_report=rep)
        assert rec.iteration == 1
        assert rec.score_delta == 0.0
        assert rec.code == "x = 1"

    def test_score_delta_stored(self):
        fv  = FeatureVector()
        rep = QualityReport()
        rec = IterationRecord(iteration=2, code="", feature_vector=fv, quality_report=rep, score_delta=4.5)
        assert rec.score_delta == 4.5
