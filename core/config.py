"""
config.py — Centralized configuration & constants.
Import from here instead of scattering magic values across modules.
"""

from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────────
# LLM / API
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class APIConfig:
    model:      str = "llama-3.3-70b-versatile"
    max_tokens: int = 4096
    provider:   str = "groq"


# ─────────────────────────────────────────────────────────────────────
# Refiner defaults
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RefinerConfig:
    max_iterations:  int   = 5
    target_score:    float = 85.0
    min_improvement: float = 2.0   # stop if score gain < this value
    patience:        int   = 2     # stop after N non-improving iterations


# ─────────────────────────────────────────────────────────────────────
# Classifier thresholds
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ClassifierThresholds:
    # Complexity
    max_cyclomatic_complexity: int   = 10
    max_nesting_depth:         int   = 4
    max_function_lines:        int   = 50

    # Documentation
    min_docstring_coverage:    float = 0.50
    min_comment_density:       float = 0.05
    min_loc_for_comments:      int   = 30

    # Naming
    max_short_names_ratio:     float = 0.30
    min_naming_convention:     float = 0.70

    # Style
    max_long_lines_ratio:      float = 0.15
    max_magic_numbers:         int   = 3
    max_global_vars:           int   = 2

    # Type safety
    min_type_hint_coverage:    float = 0.40
    min_functions_for_types:   int   = 1

    # Error handling
    min_functions_for_coverage: int  = 2
    min_exception_coverage:    float = 0.30


# ─────────────────────────────────────────────────────────────────────
# Scorer weights
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ScorerWeights:
    # Dimension weights in overall score (must sum to 1.0)
    complexity:     float = 0.20
    readability:    float = 0.20
    documentation:  float = 0.20
    best_practices: float = 0.20
    maintainability: float = 0.20

    # Problem severity penalties
    high_penalty:   int = 8
    medium_penalty: int = 4
    low_penalty:    int = 1

    # Grade boundaries
    grade_a: float = 85.0
    grade_b: float = 70.0
    grade_c: float = 55.0
    grade_d: float = 40.0


# ─────────────────────────────────────────────────────────────────────
# Singleton instances (import these)
# ─────────────────────────────────────────────────────────────────────

API_CONFIG        = APIConfig()
REFINER_CONFIG    = RefinerConfig()
THRESHOLDS        = ClassifierThresholds()
WEIGHTS           = ScorerWeights()
