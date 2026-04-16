"""
Quality scorer — converts a FeatureVector + problem list into a QualityReport.
"""

from config import WEIGHTS
from .models import FeatureVector, QualityReport

_W = WEIGHTS


class QualityScorer:
    """Compute dimension scores and an overall grade from a FeatureVector."""

    def score(self, fv: FeatureVector, problems: list[dict]) -> QualityReport:

        # ── Dimension scores (0‑100) ──────────────────────────────────
        complexity_score = max(
            0,
            100
            - fv.cyclomatic_complexity * 4
            - fv.max_nesting_depth     * 5
            - fv.cognitive_complexity  * 1.5,
        )

        readability_score = (
            (1 - fv.short_names_ratio)      * 30
            + fv.descriptive_names_ratio     * 25
            + fv.naming_convention_score     * 25
            + (1 - fv.long_lines_ratio)      * 20
        )

        doc_score = (
            (1 if fv.has_module_docstring else 0) * 20
            + fv.docstring_coverage               * 50
            + min(fv.comment_density * 500, 30)
        )

        best_practices_score = (
            (1 - min(fv.bare_except_count   * 0.20, 1)) * 30
            + (1 - min(fv.magic_numbers_count * 0.05, 1)) * 20
            + (1 - min(fv.global_vars_count * 0.10, 1))  * 20
            + (1 if fv.uses_type_hints else 0)            * 15
            + (1 if fv.uses_list_comp or fv.uses_generators else 0) * 15
        )

        maintainability_score = (
            complexity_score      * 0.30
            + readability_score   * 0.30
            + doc_score           * 0.20
            + best_practices_score * 0.20
        )

        # ── Penalty for detected problems ─────────────────────────────
        severity_penalty = sum(
            {
                "high":   _W.high_penalty,
                "medium": _W.medium_penalty,
                "low":    _W.low_penalty,
            }.get(p["severity"], 0)
            for p in problems
        )

        overall = max(0, min(100,
            complexity_score       * _W.complexity
            + readability_score    * _W.readability
            + doc_score            * _W.documentation
            + best_practices_score * _W.best_practices
            + maintainability_score * _W.maintainability
            - severity_penalty,
        ))

        grade = (
            "A" if overall >= _W.grade_a else
            "B" if overall >= _W.grade_b else
            "C" if overall >= _W.grade_c else
            "D" if overall >= _W.grade_d else "F"
        )

        return QualityReport(
            overall_score         = round(overall,                2),
            readability_score     = round(readability_score,     2),
            maintainability_score = round(maintainability_score, 2),
            complexity_score      = round(complexity_score,      2),
            documentation_score   = round(doc_score,             2),
            best_practices_score  = round(best_practices_score,  2),
            problems              = problems,
            grade                 = grade,
        )
