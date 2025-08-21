import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ValidationMode(Enum):
    """Validation strictness levels for parameter parsing"""

    STRICT = "strict"  # Strict validation, raise errors for invalid params
    LENIENT = "lenient"  # Warn about issues but continue parsing
    NONE = "none"  # No validation, parse everything as-is


@dataclass
class ParseResult:
    """Encapsulates parsing results with metadata and warnings"""

    optuna_params: Dict[str, Any]
    warnings: List[str]
    metadata: Dict[str, Any]

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}


class OptunaParam:
    """Represents a single Optuna parameter suggestion"""

    def __init__(self, suggest_method: str, **kwargs):
        self.suggest_method = suggest_method  # 'float', 'int', 'categorical', etc.
        self.kwargs = kwargs

    def suggest(self, trial, name: str):
        """Call the appropriate trial.suggest_* method"""
        suggest_func = getattr(trial, f"suggest_{self.suggest_method}")
        return suggest_func(name, **self.kwargs)

    def __repr__(self):
        kwargs_str = ", ".join(f"{k}={v}" for k, v in self.kwargs.items())
        return f"OptunaParam(suggest_{self.suggest_method}, {kwargs_str})"
