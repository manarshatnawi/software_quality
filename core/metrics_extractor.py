"""
Utilities for extracting ML-friendly metrics from Python code.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
from radon.raw import analyze
import radon.metrics as rm
from radon.visitors import ComplexityVisitor
from tqdm import tqdm


class CodeMetricsExtractor:
    def extract_all_metrics(self, code: str) -> dict[str, Any] | None:
        metrics: dict[str, Any] = {}

        try:
            raw = analyze(code)
            metrics["loc"] = raw.loc
            metrics["lloc"] = raw.lloc
            metrics["sloc"] = raw.sloc
            metrics["comments"] = raw.comments
            metrics["blank_lines"] = raw.blank

            try:
                complexity_visitor = ComplexityVisitor.from_code(code)
                functions = complexity_visitor.functions
                classes = complexity_visitor.classes
                all_complexities = [comp.complexity for comp in functions + classes]

                metrics["num_functions"] = len(functions)
                metrics["num_classes"] = len(classes)
                metrics["total_complexity"] = sum(all_complexities) if all_complexities else 0
                metrics["avg_complexity"] = round(sum(all_complexities) / len(all_complexities), 2) if all_complexities else 1
                metrics["max_complexity"] = max(all_complexities) if all_complexities else 1
                metrics["high_complexity_methods"] = sum(1 for c in all_complexities if c >= 10)
            except Exception:
                metrics["num_functions"] = 0
                metrics["num_classes"] = 0
                metrics["total_complexity"] = 0
                metrics["avg_complexity"] = 1
                metrics["max_complexity"] = 1
                metrics["high_complexity_methods"] = 0

            try:
                mi = rm.mi_visit(code, multi=True)
                metrics["maintainability_index"] = round(mi[0], 2) if isinstance(mi, tuple) else round(mi, 2)
            except Exception:
                metrics["maintainability_index"] = 50

            metrics["comment_density"] = round((metrics["comments"] / metrics["loc"] * 100), 2) if metrics["loc"] > 0 else 0
            magic_numbers = re.findall(r"\b\d{2,}\b", code)
            common_numbers = {0, 1, -1, 2, 10, 100}
            metrics["magic_numbers"] = sum(1 for num in magic_numbers if int(num) not in common_numbers)
            metrics["avg_function_length"] = self._calculate_avg_function_length(code)
            metrics["max_nesting_depth"] = self._calculate_max_nesting(code)
            metrics["code_smells"] = self._detect_code_smells(code)
            metrics["pep8_score"] = self._check_pep8(code)
            metrics["duplicate_ratio"] = self._estimate_duplication(code)
        except Exception as exc:
            print(f"Error: {exc}")
            return None

        return metrics

    def _calculate_avg_function_length(self, code: str) -> float:
        function_lengths: list[int] = []
        current_length = 0
        in_function = False

        for line in code.split("\n"):
            if "def " in line and ":" in line:
                if in_function and current_length > 0:
                    function_lengths.append(current_length)
                in_function = True
                current_length = 1
            elif in_function:
                if line.strip() and not line.strip().startswith("#"):
                    current_length += 1
                if line.strip() == "":
                    function_lengths.append(current_length)
                    in_function = False

        if in_function and current_length > 0:
            function_lengths.append(current_length)

        return round(sum(function_lengths) / len(function_lengths), 2) if function_lengths else 0

    def _calculate_max_nesting(self, code: str) -> int:
        max_depth = 0
        current_depth = 0

        for line in code.split("\n"):
            stripped = line.strip()
            if stripped.startswith(("if ", "elif ", "else:", "for ", "while ", "try:")) and ":" in stripped:
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif not stripped or stripped.startswith("#"):
                continue
            else:
                current_depth = max(0, current_depth - 1)

        return max_depth

    def _detect_code_smells(self, code: str) -> int:
        smells = 0
        current_method_lines = 0
        in_method = False

        for line in code.split("\n"):
            if "def " in line and ":" in line:
                if in_method and current_method_lines > 30:
                    smells += 1
                in_method = True
                current_method_lines = 0
            elif in_method and line.strip() and not line.strip().startswith("#"):
                current_method_lines += 1

        for params in re.findall(r"def\s+\w+\(([^)]*)\)", code):
            if params.count(",") >= 4:
                smells += 1

        return min(smells, 10)

    def _check_pep8(self, code: str) -> int:
        score = 100
        for line in code.split("\n"):
            if len(line) > 79:
                score -= 2
            if " =" in line and "= " not in line:
                score -= 1
            if ";" in line:
                score -= 2
        return max(0, min(100, score))

    def _estimate_duplication(self, code: str) -> float:
        lines = [line.strip() for line in code.split("\n") if line.strip() and not line.strip().startswith("#")]
        if not lines:
            return 0
        duplication = 1 - (len(set(lines)) / len(lines))
        return round(duplication * 100, 2)


def process_all_codes(codes_dir: str = "downloaded_codes", output_file: str = "metrics_data/all_metrics.csv") -> None:
    extractor = CodeMetricsExtractor()
    results = []

    codes_path = Path(codes_dir)
    py_files = sorted(codes_path.glob("code_*.py"))
    quality_scores: dict[str, int] = {}
    quality_file = codes_path / "quality_scores.txt"

    if quality_file.exists():
        with open(quality_file, "r", encoding="utf-8") as handle:
            for line in handle.readlines()[1:]:
                if "," not in line:
                    continue
                parts = line.strip().split(",")
                if len(parts) == 2:
                    quality_scores[parts[0]] = int(parts[1])

    print(f"Processing {len(py_files)} Python files...")
    print("=" * 50)

    for py_file in tqdm(py_files, desc="Extracting metrics"):
        try:
            with open(py_file, "r", encoding="utf-8") as handle:
                code = handle.read()
            metrics = extractor.extract_all_metrics(code)
            if metrics:
                results.append({
                    "filename": py_file.name,
                    "quality_score": quality_scores.get(py_file.name, 50),
                    **metrics,
                })
        except Exception as exc:
            print(f"Error processing {py_file.name}: {exc}")

    if results:
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True)
        pd.DataFrame(results).to_csv(output_path, index=False)
        print("\n" + "=" * 50)
        print(f"Metrics saved to: {output_path}")
        print(f"Total processed: {len(results)} files")
