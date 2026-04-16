"""
Rule-based problem classifier.
Maps a FeatureVector to a list of problems and a primary category.
Can be swapped for a trained sklearn / xgboost model.
"""

import os
import numpy as np
from tensorflow.keras.models import load_model
from config import THRESHOLDS
from .models import FeatureVector

_T = THRESHOLDS   # short alias


class ProblemClassifier:
    """
    Deterministic rule-based classifier that maps feature vectors
    to problem categories + severity scores.
    Can use a trained DL model if available.
    """

    CATEGORIES = [
        "complexity_overload",
        "poor_documentation",
        "naming_issues",
        "error_handling_gaps",
        "style_violations",
        "type_safety_issues",
        "clean_code",
    ]

    def __init__(self):
        self.model = None
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'final_model.keras')
        if os.path.exists(model_path):
            try:
                self.model = load_model(model_path)
            except Exception as e:
                print(f"Failed to load model: {e}")
                self.model = None

    def classify(self, fv: FeatureVector) -> tuple[str, list[dict]]:
        """Return (primary_category, problems_list)."""
        if self.model:
            # Use DL model for prediction
            features = np.array([[
                fv.cyclomatic_complexity,
                fv.cognitive_complexity,
                fv.max_nesting_depth,
                fv.max_function_lines,
                fv.docstring_coverage,
                fv.comment_density,
                fv.short_names_ratio,
                fv.naming_convention_score,
                fv.bare_except_count,
                fv.exception_coverage,
                fv.long_lines_ratio,
                fv.magic_numbers_count,
                fv.global_vars_count,
                fv.type_hint_coverage,
                fv.has_module_docstring,
                fv.uses_type_hints,
                fv.uses_list_comp,
                fv.uses_generators,
                fv.lines_of_code,
                fv.num_functions,
            ]])
            prediction = self.model.predict(features)
            predicted_class = np.argmax(prediction, axis=1)[0]
            # Map to categories (assuming model outputs 0: clean, 1: naming, 2: complexity)
            if predicted_class == 0:
                primary = "clean_code"
                problems = []
            elif predicted_class == 1:
                primary = "naming_issues"
                problems = [{"type": "naming_issues", "severity": "medium", "description": "Bad naming detected by model", "suggestion": "Improve variable and function names"}]
            elif predicted_class == 2:
                primary = "complexity_overload"
                problems = [{"type": "complexity_overload", "severity": "high", "description": "High complexity detected by model", "suggestion": "Simplify code structure"}]
            else:
                primary = "clean_code"
                problems = []
            return primary, problems
        else:
            # Fallback to rule-based
            problems: list[dict] = []

            # ── Complexity ───────────────────────────────────────────────
            if fv.cyclomatic_complexity > _T.max_cyclomatic_complexity:
                problems.append({
                    "type": "complexity_overload",
                    "severity": "high",
                    "description": (
                        f"Cyclomatic complexity {fv.cyclomatic_complexity:.0f} "
                        f"exceeds threshold ({_T.max_cyclomatic_complexity})"
                    ),
                    "suggestion": "Break large functions into smaller, single-responsibility units",
                })
            if fv.max_nesting_depth > _T.max_nesting_depth:
                problems.append({
                    "type": "complexity_overload",
                    "severity": "medium",
                    "description": f"Max nesting depth {fv.max_nesting_depth} — code is hard to follow",
                    "suggestion": "Use early returns, guard clauses, or extract nested logic",
                })
        if fv.max_function_lines > _T.max_function_lines:
            problems.append({
                "type": "complexity_overload",
                "severity": "medium",
                "description": f"Longest function is {fv.max_function_lines} lines",
                "suggestion": f"Functions should ideally be under {_T.max_function_lines // 2} lines",
            })

        # ── Documentation ────────────────────────────────────────────
        if not fv.has_module_docstring:
            problems.append({
                "type": "poor_documentation",
                "severity": "low",
                "description": "Missing module-level docstring",
                "suggestion": "Add a module docstring describing the file's purpose",
            })
        if fv.docstring_coverage < _T.min_docstring_coverage:
            problems.append({
                "type": "poor_documentation",
                "severity": "medium",
                "description": f"Only {fv.docstring_coverage:.0%} of functions/classes have docstrings",
                "suggestion": "Document all public functions with Args, Returns, Raises",
            })
        if fv.comment_density < _T.min_comment_density and fv.lines_of_code > _T.min_loc_for_comments:
            problems.append({
                "type": "poor_documentation",
                "severity": "low",
                "description": "Very low comment density",
                "suggestion": "Add inline comments for non-obvious logic",
            })

        # ── Naming ───────────────────────────────────────────────────
        if fv.short_names_ratio > _T.max_short_names_ratio:
            problems.append({
                "type": "naming_issues",
                "severity": "medium",
                "description": f"{fv.short_names_ratio:.0%} of names are too short (< 3 chars)",
                "suggestion": "Use descriptive names that reveal intent",
            })
        if fv.naming_convention_score < _T.min_naming_convention:
            problems.append({
                "type": "naming_issues",
                "severity": "medium",
                "description": "Inconsistent naming convention (expected snake_case)",
                "suggestion": "Follow PEP8 naming conventions throughout",
            })

        # ── Error handling ───────────────────────────────────────────
        if fv.bare_except_count > 0:
            problems.append({
                "type": "error_handling_gaps",
                "severity": "high",
                "description": f"{fv.bare_except_count} bare `except:` clause(s) found",
                "suggestion": "Always catch specific exceptions (e.g., except ValueError)",
            })
        if fv.num_functions > _T.min_functions_for_coverage and fv.exception_coverage < _T.min_exception_coverage:
            problems.append({
                "type": "error_handling_gaps",
                "severity": "medium",
                "description": "Low exception coverage across functions",
                "suggestion": "Add try/except blocks to functions that interact with I/O or external data",
            })

        # ── Style ────────────────────────────────────────────────────
        if fv.long_lines_ratio > _T.max_long_lines_ratio:
            problems.append({
                "type": "style_violations",
                "severity": "low",
                "description": f"{fv.long_lines_ratio:.0%} of lines exceed 79 characters",
                "suggestion": "Wrap long lines per PEP8",
            })
        if fv.magic_numbers_count > _T.max_magic_numbers:
            problems.append({
                "type": "style_violations",
                "severity": "medium",
                "description": f"{fv.magic_numbers_count} magic numbers detected",
                "suggestion": "Replace magic numbers with named constants",
            })
        if fv.global_vars_count > _T.max_global_vars:
            problems.append({
                "type": "style_violations",
                "severity": "medium",
                "description": f"{fv.global_vars_count} global variable declarations",
                "suggestion": "Avoid global state; prefer passing data as parameters",
            })

        # ── Type safety ──────────────────────────────────────────────
        if fv.num_functions > _T.min_functions_for_types and fv.type_hint_coverage < _T.min_type_hint_coverage:
            problems.append({
                "type": "type_safety_issues",
                "severity": "low",
                "description": f"Type hints present in only {fv.type_hint_coverage:.0%} of functions",
                "suggestion": "Add type annotations for better tooling support and readability",
            })

        # ── Primary category ─────────────────────────────────────────
        if not problems:
            primary = "clean_code"
        else:
            counts: dict[str, int] = {}
            for p in problems:
                counts[p["type"]] = counts.get(p["type"], 0) + 1
            primary = max(counts, key=counts.get)  # type: ignore

        return primary, problems
