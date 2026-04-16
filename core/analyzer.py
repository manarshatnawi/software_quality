"""
══════════════════════════════════════════════════════════════════════
  Smart Code Quality Analyzer & Iterative Refiner
  نظام تحليل وإصلاح الكود بشكل تكراري ذكي
══════════════════════════════════════════════════════════════════════
"""

import ast
import json
import math
import os
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any
from groq import Groq

# استيراد دالة التنبؤ من النموذج المحلي (Deep Learning)
try:
    from .ml_integration import dl_predictor, gnn_classifier
    from .metrics_extractor import process_all_codes
except ImportError:
    from ml_integration import dl_predictor, gnn_classifier
    from metrics_extractor import process_all_codes


# ─────────────────────────────────────────────────────────────────────
# 1.  DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────

@dataclass
class FeatureVector:
    """Structural & stylistic features extracted via AST."""
    cyclomatic_complexity: float = 0.0
    cognitive_complexity:  float = 0.0
    max_nesting_depth:     int   = 0
    avg_nesting_depth:     float = 0.0
    lines_of_code:      int = 0
    num_functions:      int = 0
    num_classes:        int = 0
    avg_function_lines: float = 0.0
    max_function_lines: int   = 0
    short_names_ratio:      float = 0.0
    descriptive_names_ratio: float = 0.0
    naming_convention_score: float = 0.0
    has_module_docstring: bool  = False
    docstring_coverage:   float = 0.0
    comment_density:      float = 0.0
    try_except_count:   int   = 0
    bare_except_count:  int   = 0
    exception_coverage: float = 0.0
    duplicate_code_score:  float = 0.0
    magic_numbers_count:   int   = 0
    long_lines_ratio:      float = 0.0
    unused_imports:        int   = 0
    global_vars_count:     int   = 0
    return_complexity:     float = 0.0
    uses_type_hints:   bool  = False
    type_hint_coverage: float = 0.0
    uses_list_comp:    bool  = False
    uses_generators:   bool  = False


@dataclass
class QualityReport:
    """Aggregate quality scores and identified problems."""
    overall_score:     float = 0.0
    readability_score: float = 0.0
    maintainability_score: float = 0.0
    complexity_score:  float = 0.0
    documentation_score: float = 0.0
    best_practices_score: float = 0.0
    problems: list[dict] = field(default_factory=list)
    problem_category: str = "unknown"
    grade: str = "F"


# ─────────────────────────────────────────────────────────────────────
# 2.  AST PARSER & FEATURE EXTRACTOR
# ─────────────────────────────────────────────────────────────────────

class ASTAnalyzer(ast.NodeVisitor):
    """Walk the AST and collect raw metrics."""

    def __init__(self, source: str):
        self.source  = source
        self.lines   = source.splitlines()
        self._reset()

    def _reset(self):
        self._functions: list = []
        self._classes:   list = []
        self._imports:   list = []
        self._try_blocks:      int = 0
        self._bare_excepts:    int = 0
        self._global_stmts:    int = 0
        self._magic_numbers:   int = 0
        self._names:           list[str] = []
        self._nesting_depths:  list[int] = []
        self._current_depth:   int = 0
        self._branch_count:    int = 0
        self._uses_type_hints: bool = False
        self._type_hinted_fns: int  = 0
        self._list_comps:      bool = False
        self._generators:      bool = False

    def visit_FunctionDef(self, node):
        self._functions.append(node)
        if node.returns or any(a.annotation for a in node.args.args):
            self._uses_type_hints = True
            self._type_hinted_fns += 1
        self._current_depth += 1
        self._nesting_depths.append(self._current_depth)
        self.generic_visit(node)
        self._current_depth -= 1

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self._classes.append(node)
        self._current_depth += 1
        self._nesting_depths.append(self._current_depth)
        self.generic_visit(node)
        self._current_depth -= 1

    def visit_If(self, node):
        self._branch_count += 1
        self._nesting_depths.append(self._current_depth + 1)
        self._current_depth += 1
        self.generic_visit(node)
        self._current_depth -= 1

    visit_While = visit_If
    visit_For   = visit_If
    visit_ExceptHandler = visit_If

    def visit_Try(self, node):
        self._try_blocks += 1
        for handler in node.handlers:
            if handler.type is None:
                self._bare_excepts += 1
        self.generic_visit(node)

    def visit_Global(self, node):
        self._global_stmts += 1
        self.generic_visit(node)

    def visit_Import(self, node):
        self._imports.append(node)
        self.generic_visit(node)

    visit_ImportFrom = visit_Import

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):
            self._names.append(node.id)
        self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, (int, float)) and node.value not in (0, 1, -1, True, False):
            self._magic_numbers += 1
        self.generic_visit(node)

    def visit_ListComp(self, node):
        self._list_comps = True
        self.generic_visit(node)

    def visit_GeneratorExp(self, node):
        self._generators = True
        self.generic_visit(node)

    def _function_line_lengths(self) -> list[int]:
        lengths = []
        for fn in self._functions:
            start = fn.lineno
            end   = getattr(fn, 'end_lineno', fn.lineno)
            lengths.append(end - start + 1)
        return lengths

    def _docstring_coverage(self) -> float:
        total = len(self._functions) + len(self._classes)
        if total == 0:
            return 1.0
        covered = sum(
            1 for node in [*self._functions, *self._classes]
            if (ast.get_docstring(node) is not None)
        )
        return covered / total

    def _naming_score(self) -> tuple[float, float, float]:
        if not self._names:
            return 0.0, 1.0, 0.5
        short = sum(1 for n in self._names if len(n) < 3 and n not in ('i', 'j', 'k', 'x', 'y', 'n'))
        descriptive = sum(1 for n in self._names if len(n) >= 5)
        snake_case  = sum(1 for n in self._names if re.match(r'^[a-z][a-z0-9_]*$', n))
        short_ratio       = short / len(self._names)
        descriptive_ratio = descriptive / len(self._names)
        convention_score  = snake_case / len(self._names)
        return short_ratio, descriptive_ratio, convention_score

    def _comment_density(self) -> float:
        comment_lines = sum(1 for l in self.lines if l.strip().startswith('#'))
        return comment_lines / max(len(self.lines), 1)

    def _long_lines_ratio(self) -> float:
        long = sum(1 for l in self.lines if len(l) > 79)
        return long / max(len(self.lines), 1)

    def _cognitive_complexity(self) -> float:
        score = 0.0
        for line in self.lines:
            stripped = line.lstrip()
            indent   = len(line) - len(stripped)
            nesting  = indent // 4
            if any(stripped.startswith(kw) for kw in ('if ', 'elif ', 'else:', 'for ', 'while ', 'except')):
                score += 1 + nesting
            if ' and ' in stripped or ' or ' in stripped:
                score += stripped.count(' and ') + stripped.count(' or ')
        return score

    def build_feature_vector(self) -> FeatureVector:
        self.visit(ast.parse(self.source))

        fn_lengths = self._function_line_lengths()
        short_r, desc_r, conv_score = self._naming_score()
        depths = self._nesting_depths or [0]

        total_fns = len(self._functions)
        type_hint_cov = (self._type_hinted_fns / total_fns) if total_fns else 0.0
        cyclomatic = 1 + self._branch_count

        return FeatureVector(
            cyclomatic_complexity  = float(cyclomatic),
            cognitive_complexity   = self._cognitive_complexity(),
            max_nesting_depth      = max(depths),
            avg_nesting_depth      = sum(depths) / len(depths),
            lines_of_code          = len(self.lines),
            num_functions          = total_fns,
            num_classes            = len(self._classes),
            avg_function_lines     = (sum(fn_lengths) / len(fn_lengths)) if fn_lengths else 0,
            max_function_lines     = max(fn_lengths, default=0),
            short_names_ratio      = short_r,
            descriptive_names_ratio= desc_r,
            naming_convention_score= conv_score,
            has_module_docstring   = bool(ast.get_docstring(ast.parse(self.source))),
            docstring_coverage     = self._docstring_coverage(),
            comment_density        = self._comment_density(),
            try_except_count       = self._try_blocks,
            bare_except_count      = self._bare_excepts,
            exception_coverage     = (self._try_blocks / total_fns) if total_fns else 0,
            magic_numbers_count    = self._magic_numbers,
            long_lines_ratio       = self._long_lines_ratio(),
            unused_imports         = 0,
            global_vars_count      = self._global_stmts,
            uses_type_hints        = self._uses_type_hints,
            type_hint_coverage     = type_hint_cov,
            uses_list_comp         = self._list_comps,
            uses_generators        = self._generators,
        )


# ─────────────────────────────────────────────────────────────────────
# 3.  RULE-BASED CLASSIFIER + GNN ENHANCEMENT
# ─────────────────────────────────────────────────────────────────────

class ProblemClassifier:
    CATEGORIES = [
        "complexity_overload",
        "poor_documentation",
        "naming_issues",
        "error_handling_gaps",
        "style_violations",
        "type_safety_issues",
        "clean_code",
    ]

    def classify(self, fv: FeatureVector, code_snippet: str = "") -> tuple[str, list[dict]]:
        problems: list[dict] = []

        # Complexity
        if fv.cyclomatic_complexity > 10:
            problems.append({
                "type": "complexity_overload",
                "severity": "high",
                "description": f"Cyclomatic complexity {fv.cyclomatic_complexity:.0f} exceeds threshold (10)",
                "suggestion": "Break large functions into smaller, single-responsibility units"
            })
        if fv.max_nesting_depth > 4:
            problems.append({
                "type": "complexity_overload",
                "severity": "medium",
                "description": f"Max nesting depth {fv.max_nesting_depth} — code is hard to follow",
                "suggestion": "Use early returns, guard clauses, or extract nested logic"
            })
        if fv.max_function_lines > 50:
            problems.append({
                "type": "complexity_overload",
                "severity": "medium",
                "description": f"Longest function is {fv.max_function_lines} lines",
                "suggestion": "Functions should ideally be under 30 lines"
            })

        # Documentation
        if not fv.has_module_docstring:
            problems.append({
                "type": "poor_documentation",
                "severity": "low",
                "description": "Missing module-level docstring",
                "suggestion": "Add a module docstring describing the file's purpose"
            })
        if fv.docstring_coverage < 0.5:
            problems.append({
                "type": "poor_documentation",
                "severity": "medium",
                "description": f"Only {fv.docstring_coverage:.0%} of functions/classes have docstrings",
                "suggestion": "Document all public functions with Args, Returns, Raises"
            })
        if fv.comment_density < 0.05 and fv.lines_of_code > 30:
            problems.append({
                "type": "poor_documentation",
                "severity": "low",
                "description": "Very low comment density",
                "suggestion": "Add inline comments for non-obvious logic"
            })

        # Naming
        if fv.short_names_ratio > 0.3:
            problems.append({
                "type": "naming_issues",
                "severity": "medium",
                "description": f"{fv.short_names_ratio:.0%} of names are too short (< 3 chars)",
                "suggestion": "Use descriptive names that reveal intent"
            })
        if fv.naming_convention_score < 0.7:
            problems.append({
                "type": "naming_issues",
                "severity": "medium",
                "description": "Inconsistent naming convention (expected snake_case)",
                "suggestion": "Follow PEP8 naming conventions throughout"
            })

        # Error handling
        if fv.bare_except_count > 0:
            problems.append({
                "type": "error_handling_gaps",
                "severity": "high",
                "description": f"{fv.bare_except_count} bare `except:` clause(s) found",
                "suggestion": "Always catch specific exceptions (e.g., except ValueError)"
            })
        if fv.num_functions > 2 and fv.exception_coverage < 0.3:
            problems.append({
                "type": "error_handling_gaps",
                "severity": "medium",
                "description": "Low exception coverage across functions",
                "suggestion": "Add try/except blocks to functions that interact with I/O or external data"
            })

        # Style
        if fv.long_lines_ratio > 0.15:
            problems.append({
                "type": "style_violations",
                "severity": "low",
                "description": f"{fv.long_lines_ratio:.0%} of lines exceed 79 characters",
                "suggestion": "Wrap long lines per PEP8"
            })
        if fv.magic_numbers_count > 3:
            problems.append({
                "type": "style_violations",
                "severity": "medium",
                "description": f"{fv.magic_numbers_count} magic numbers detected",
                "suggestion": "Replace magic numbers with named constants"
            })
        if fv.global_vars_count > 2:
            problems.append({
                "type": "style_violations",
                "severity": "medium",
                "description": f"{fv.global_vars_count} global variable declarations",
                "suggestion": "Avoid global state; prefer passing data as parameters"
            })

        # Type safety
        if fv.num_functions > 1 and fv.type_hint_coverage < 0.4:
            problems.append({
                "type": "type_safety_issues",
                "severity": "low",
                "description": f"Type hints present in only {fv.type_hint_coverage:.0%} of functions",
                "suggestion": "Add type annotations for better tooling support and readability"
            })

        # Determine primary category (with GNN enhancement if available)
        gnn_category = None
        if code_snippet:
            gnn_category = gnn_classifier().predict(code_snippet)

        if not problems and gnn_category:
            primary = gnn_category
        elif not problems:
            primary = "clean_code"
        else:
            category_counts: dict[str, int] = {}
            for p in problems:
                category_counts[p["type"]] = category_counts.get(p["type"], 0) + 1
            primary = max(category_counts, key=category_counts.get)

        return primary, problems


# ─────────────────────────────────────────────────────────────────────
# 4.  QUALITY SCORER (with DL model integration)
# ─────────────────────────────────────────────────────────────────────

class QualityScorer:

    def score(self, fv: FeatureVector, problems: list[dict]) -> QualityReport:
        dl_score = dl_predictor().predict(fv)

        complexity_score = max(0, 100 - fv.cyclomatic_complexity * 4 - fv.max_nesting_depth * 5 - fv.cognitive_complexity * 1.5)
        readability_score = ((1 - fv.short_names_ratio) * 30 + fv.descriptive_names_ratio * 25 + fv.naming_convention_score * 25 + (1 - fv.long_lines_ratio) * 20)
        doc_score = ((1 if fv.has_module_docstring else 0) * 20 + fv.docstring_coverage * 50 + min(fv.comment_density * 500, 30))
        best_practices_score = ((1 - min(fv.bare_except_count * 0.2, 1)) * 30 + (1 - min(fv.magic_numbers_count * 0.05, 1)) * 20 + (1 - min(fv.global_vars_count * 0.1, 1)) * 20 + (1 if fv.uses_type_hints else 0) * 15 + (1 if fv.uses_list_comp or fv.uses_generators else 0) * 15)
        maintainability_score = (complexity_score * 0.3 + readability_score * 0.3 + doc_score * 0.2 + best_practices_score * 0.2)
        severity_penalty = sum({"high": 8, "medium": 4, "low": 1}.get(p["severity"], 0) for p in problems)

        heuristic_score = max(0, min(100, complexity_score * 0.20 + readability_score * 0.20 + doc_score * 0.20 + best_practices_score * 0.20 + maintainability_score * 0.20 - severity_penalty))

        if dl_score is not None:
            overall_score = heuristic_score * 0.7 + float(dl_score) * 0.3
            if severity_penalty >= 8:
                overall_score = min(overall_score, 79.0)
            if severity_penalty >= 16:
                overall_score = min(overall_score, 64.0)
        else:
            overall_score = heuristic_score

        grade = "A" if overall_score >= 85 else "B" if overall_score >= 70 else "C" if overall_score >= 55 else "D" if overall_score >= 40 else "F"
        return QualityReport(
            overall_score=round(overall_score, 2),
            readability_score=round(readability_score, 2),
            maintainability_score=round(maintainability_score, 2),
            complexity_score=round(complexity_score, 2),
            documentation_score=round(doc_score, 2),
            best_practices_score=round(best_practices_score, 2),
            problems=problems,
            grade=grade,
        )


# ─────────────────────────────────────────────────────────────────────
# 5.  PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────

class RepairPromptBuilder:

    def build(self, code: str, report: QualityReport, fv: FeatureVector, iteration: int) -> str:
        problem_lines = "\n".join(
            f"  [{p['severity'].upper()}] ({p['type']}) {p['description']}\n"
            f"    → Suggestion: {p['suggestion']}"
            for p in report.problems
        )

        scores_summary = (
            f"Overall: {report.overall_score}/100 (Grade {report.grade})\n"
            f"  Readability:      {report.readability_score:.1f}\n"
            f"  Maintainability:  {report.maintainability_score:.1f}\n"
            f"  Complexity:       {report.complexity_score:.1f}\n"
            f"  Documentation:    {report.documentation_score:.1f}\n"
            f"  Best Practices:   {report.best_practices_score:.1f}"
        )

        primary = report.problem_category

        focus_instruction = {
            "complexity_overload":   "PRIMARY FOCUS: Reduce complexity. Split functions, flatten nesting, apply early returns.",
            "poor_documentation":    "PRIMARY FOCUS: Add comprehensive docstrings (Google style) and meaningful inline comments.",
            "naming_issues":         "PRIMARY FOCUS: Rename all ambiguous identifiers to descriptive, PEP8-compliant names.",
            "error_handling_gaps":   "PRIMARY FOCUS: Add specific exception handling and error logging.",
            "style_violations":      "PRIMARY FOCUS: Fix all PEP8 violations, replace magic numbers with constants.",
            "type_safety_issues":    "PRIMARY FOCUS: Add complete type annotations using the `typing` module.",
            "clean_code":            "Code is mostly clean. Make subtle improvements only.",
        }.get(primary, "Improve overall code quality.")

        return f"""You are an expert Python code quality engineer performing iteration {iteration} of an automated refactoring pipeline.

CURRENT QUALITY SCORES:
{scores_summary}

DETECTED PROBLEMS:
{problem_lines if problem_lines else "  None significant"}

{focus_instruction}

STRICT RULES:
1. Return ONLY the improved Python code — no markdown fences, no explanations.
2. Preserve ALL original functionality exactly.
3. Do not add new features or change the public API.
4. Apply ALL suggestions listed above, not just the primary focus.
5. Ensure the result is syntactically valid Python.
6. Target an overall score above 85/100.
7. Make the code SHORTER and more CONCISE — remove unnecessary code, combine similar logic.
8. Organize the code better: group related functions, use clear structure, improve readability.
9. Perform REAL FIXES: actually resolve the identified problems, not just cosmetic changes.
10. Prefer modern Python idioms: list comprehensions, context managers, etc. where appropriate.

ORIGINAL CODE:
{code}

IMPROVED CODE:"""


# ─────────────────────────────────────────────────────────────────────
# 6.  ITERATIVE REFINER (main orchestrator)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class IterationRecord:
    iteration: int
    code: str
    feature_vector: FeatureVector
    quality_report: QualityReport
    score_delta: float = 0.0


class IterativeRefiner:

    def __init__(
        self,
        max_iterations: int = 5,
        target_score:   float = 85.0,
        min_improvement: float = 2.0,
        patience:       int = 2,
        api_key: str | None = None,
    ):
        self.max_iterations  = max_iterations
        self.target_score    = target_score
        self.min_improvement = min_improvement
        self.patience        = patience

        self._analyzer    = ASTAnalyzer
        self._classifier  = ProblemClassifier()
        self._scorer      = QualityScorer()
        self._prompt_builder = RepairPromptBuilder()

        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if api_key:
            self._client = Groq(api_key=api_key)
        else:
            self._client = None

        self.last_api_error: str = ""
        self.history: list[IterationRecord] = []

    def refine(self, source_code: str) -> str:
        """Run the full iterative refinement pipeline."""
        print(self._banner("SMART CODE QUALITY REFINER"))

        current_code = source_code
        no_improve_streak = 0
        prev_score = -math.inf

        for iteration in range(1, self.max_iterations + 1):
            print(f"\n{'─'*60}")
            print(f"  ITERATION {iteration}/{self.max_iterations}")
            print(f"{'─'*60}")

            print("  [1] Parsing AST & extracting features …")
            try:
                fv = self._analyzer(current_code).build_feature_vector()
            except SyntaxError as e:
                print(f"  ✗ Syntax error in code: {e}. Stopping.")
                break

            print("  [2] Classifying problems …")
            primary_category, problems = self._classifier.classify(fv, current_code)

            print("  [3] Scoring quality …")
            report = self._scorer.score(fv, problems)
            report.problem_category = primary_category

            delta = report.overall_score - prev_score if prev_score > -math.inf else 0.0

            record = IterationRecord(
                iteration      = iteration,
                code           = current_code,
                feature_vector = fv,
                quality_report = report,
                score_delta    = delta,
            )
            self.history.append(record)

            self._print_report(report, fv, delta)

            if report.overall_score >= self.target_score and iteration > 1:
                print(f"\n  ✅ Target score {self.target_score} reached. Stopping.")
                break

            if report.problem_category == "clean_code" and iteration > 1:
                print(f"\n  ✅ Code classified as clean. Stopping.")
                break

            if iteration > 1 and delta < self.min_improvement:
                no_improve_streak += 1
                if no_improve_streak >= self.patience:
                    print(f"\n  ⚠  No significant improvement for {self.patience} iterations. Stopping.")
                    break
            else:
                no_improve_streak = 0

            if iteration == self.max_iterations:
                print(f"\n  ℹ  Max iterations reached.")
                break

            print("  [4] Building repair prompt …")
            prompt = self._prompt_builder.build(current_code, report, fv, iteration)

            print("  [5] Calling Groq API for improved code …")
            improved_code = self._call_api(prompt)

            if not improved_code or improved_code.strip() == current_code.strip():
                print("  ⚠  API returned unchanged code. Stopping.")
                break

            current_code = improved_code
            prev_score   = report.overall_score

        best = self._best_iteration()
        print(self._banner(f"DONE — Best score: {best.quality_report.overall_score:.1f} (Grade {best.quality_report.grade}) @ iteration {best.iteration}"))
        return best.code

    def _call_api(self, prompt: str) -> str:
        if self._client is None:
            self.last_api_error = "No Groq API key configured"
            return "# API key not configured - cannot refine code"

        try:
            self.last_api_error = ""
            message = self._client.chat.completions.create(
                model      = "llama-3.3-70b-versatile",
                max_tokens = 4096,
                messages   = [{"role": "user", "content": prompt}],
            )
            raw = message.choices[0].message.content.strip()
            raw = re.sub(r'^```(?:python)?\n?', '', raw)
            raw = re.sub(r'\n?```$', '', raw)
            return raw.strip()
        except Exception as e:
            print(f"  ✗ API error: {e}")
            return ""

    def call_api_with_error(self, prompt: str) -> tuple[str, str]:
        """Return improved code plus the raw API error message if the request fails."""
        try:
            message = self._client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = message.choices[0].message.content.strip()
            raw = re.sub(r"^```(?:python)?\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
            return raw.strip(), ""
        except Exception as e:
            self.last_api_error = str(e)
            return "", self.last_api_error

    def _best_iteration(self) -> IterationRecord:
        return max(self.history, key=lambda r: r.quality_report.overall_score)

    def _print_report(self, report: QualityReport, fv: FeatureVector, delta: float):
        delta_str = f"({'+' if delta >= 0 else ''}{delta:.1f})" if delta != 0 else ""
        print(f"\n  Quality Score: {report.overall_score:.1f}/100  Grade [{report.grade}]  {delta_str}")
        print(f"  Primary Category: {report.problem_category}")
        print(f"  Dimensions → "
              f"Readability:{report.readability_score:.0f}  "
              f"Complexity:{report.complexity_score:.0f}  "
              f"Docs:{report.documentation_score:.0f}  "
              f"BestPractices:{report.best_practices_score:.0f}")
        if report.problems:
            print(f"  Problems ({len(report.problems)}):")
            for p in report.problems:
                sev_icon = {"high": "🔴", "medium": "🟡", "low": "🔵"}.get(p["severity"], "⚪")
                print(f"    {sev_icon} [{p['severity']}] {p['description']}")

    @staticmethod
    def _banner(text: str) -> str:
        width = 62
        line  = "═" * width
        return f"\n{line}\n  {text}\n{line}"

    def export_report(self) -> dict[str, Any]:
        return {
            "iterations": [
                {
                    "iteration": r.iteration,
                    "score":     r.quality_report.overall_score,
                    "grade":     r.quality_report.grade,
                    "category":  r.quality_report.problem_category,
                    "problems":  len(r.quality_report.problems),
                    "delta":     r.score_delta,
                }
                for r in self.history
            ],
            "best_iteration": self._best_iteration().iteration,
            "final_score":    self._best_iteration().quality_report.overall_score,
        }


# ─────────────────────────────────────────────────────────────────────
# 7.  CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────

EXAMPLE_CODE = '''
import requests, json, os

x = 10
y = 20
z = 30

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


def main():
    print("Paste your Python code below.")
    print("Enter a blank line followed by END to finish:\n")

    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "END":
            break
        lines.append(line)

    user_code = "\n".join(lines).strip()

    if not user_code:
        print("No code provided — using built-in example.\n")
        user_code = textwrap.dedent(EXAMPLE_CODE)

    refiner = IterativeRefiner(
        max_iterations  = 4,
        target_score    = 85.0,
        min_improvement = 3.0,
        patience        = 2,
    )

    best_code = refiner.refine(user_code)

    print("\n" + "═"*62)
    print("  FINAL REFINED CODE")
    print("═"*62 + "\n")
    print(best_code)

    print("\n" + "═"*62)
    print("  PIPELINE SUMMARY (JSON)")
    print("═"*62)
    print(json.dumps(refiner.export_report(), indent=2))


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "process":
        process_all_codes()
    else:
        main()