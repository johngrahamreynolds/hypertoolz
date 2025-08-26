import logging
from typing import Any, Dict, List, Optional, Union

from ..core.config import TunerConfig
from ..core.hypertypes import OptunaParam, ParseResult, ValidationMode
from ..parsers.base import BaseParser

logger = logging.getLogger(__name__)


class ParamParser(BaseParser):
    """Parser for hyperparameter ranges for RL algorithms"""

    # Algorithm-specific parameter constraints and defaults
    _ALGORITHM_CONSTRAINTS = {
        "PPO": {
            "learning_rate": {"min": 1e-6, "max": 1e-1, "log": True},
            "clip_range": {"min": 0.01, "max": 1.0, "log": False},
            "n_epochs": {"min": 1, "max": 20, "type": "int"},
            "batch_size": {"choices": [32, 64, 128, 256, 512], "type": "categorical"},
        },
        "A2C": {
            "learning_rate": {"min": 1e-6, "max": 1e-1, "log": True},
            "n_steps": {"min": 1, "max": 10000, "type": "int"},
            "vf_coef": {"min": 0.1, "max": 1.0, "log": False},
            "activation_fn": {
                "choices": ["tanh", "relu"],
                "type": "categorical",
            },  # TODO
        },
        "SAC": {
            "learning_rate": {"min": 1e-6, "max": 1e-1, "log": True},
            "tau": {"min": 0.001, "max": 0.1, "log": True},
            "gamma": {"min": 0.9, "max": 0.9999, "log": False},
        },
        "DQN": {
            "learning_rate": {"min": 1e-6, "max": 1e-2, "log": True},
            "buffer_size": {"min": 10000, "max": 1000000, "type": "int", "log": True},
            "learning_starts": {"min": 1000, "max": 50000, "type": "int"},
            "batch_size": {"choices": [32, 64, 128, 256], "type": "categorical"},
            "tau": {"min": 0.001, "max": 0.1, "log": True},
            "gamma": {"min": 0.9, "max": 0.9999, "log": False},
            "train_freq": {"min": 1, "max": 16, "type": "int"},
            "gradient_steps": {"min": 1, "max": 4, "type": "int"},
            "target_update_interval": {
                "min": 100,
                "max": 10000,
                "type": "int",
                "log": True,
            },
            "exploration_fraction": {"min": 0.05, "max": 0.5, "log": False},
            "exploration_initial_eps": {"min": 0.8, "max": 1.0, "log": False},
            "exploration_final_eps": {"min": 0.01, "max": 0.1, "log": False},
            "max_grad_norm": {"min": 0.5, "max": 50.0, "log": True},
            # consider using this as a test algo for adding a custom type - framestacking, etc.
        },
    }

    def __init__(
        self,
        algorithm: Optional[str] = None,
        validation_mode: ValidationMode = ValidationMode.STRICT,
        config: Optional["TunerConfig"] = None,
    ):
        super().__init__(validation_mode, config)
        self.algorithm = algorithm
        self._algorithm_constraints = self._get_algorithm_constraints()

    @property
    def supported_input_types(self) -> List[str]:
        return ["param_ranges", "hyperparameter_dict"]

    def parse(self, param_ranges: Dict[str, Any]) -> ParseResult:
        """
        Parse parameter ranges into Optuna-compatible format.

        Args:
            param_ranges: Dictionary mapping parameter names to range specifications

        Returns:
            ParseResult with optuna_params, warnings, and metadata
        """
        warnings = []
        optuna_params = {}
        metadata = {
            "algorithm": self.algorithm,
            "original_param_count": len(param_ranges),
            "parsing_mode": self.validation_mode.value,
        }

        # Validate input first
        validation_issues = self.validate_input(param_ranges)
        for issue in validation_issues:
            self._handle_validation_issue(issue, warnings)

        # Parse each parameter
        for param_name, param_spec in param_ranges.items():
            try:
                optuna_param = self._parse_single_param(
                    param_name, param_spec, warnings
                )
                if optuna_param is not None:
                    optuna_params[param_name] = optuna_param
            except Exception as e:
                error_msg = f"Failed to parse parameter '{param_name}': {str(e)}"
                self._handle_validation_issue(error_msg, warnings)

        metadata["parsed_param_count"] = len(optuna_params)

        return ParseResult(
            optuna_params=optuna_params, warnings=warnings, metadata=metadata
        )

    def validate_input(self, param_ranges: Dict[str, Any]) -> List[str]:
        """Validate parameter ranges and return list of issues"""
        issues = []

        if not isinstance(param_ranges, dict):
            issues.append(
                f"param_ranges must be a dictionary, got {type(param_ranges)}"
            )
            return issues

        if not param_ranges:
            issues.append("param_ranges dictionary is empty")
            return issues

        # Check for algorithm-specific validation
        if self.algorithm and self.algorithm in self._ALGORITHM_CONSTRAINTS:
            constraints = self._ALGORITHM_CONSTRAINTS[self.algorithm]
            for param_name in param_ranges:
                if param_name not in constraints:
                    issues.append(
                        f"Parameter '{param_name}' not recognized for algorithm '{self.algorithm}'"
                    )

        return issues

    def _parse_single_param(
        self, param_name: str, param_spec: Any, warnings: List[str]
    ) -> Optional[OptunaParam]:
        """Parse a single parameter specification into OptunaParam"""

        # Handle custom types first
        if isinstance(param_spec, dict) and "type" in param_spec:
            custom_type = param_spec["type"]
            if custom_type in self._custom_types:
                return self._custom_types[custom_type](param_spec)

        # Handle tuple: continuous range (low, high)
        if isinstance(param_spec, tuple) and len(param_spec) == 2:
            low, high = param_spec
            return self._parse_continuous_range(param_name, low, high, warnings)

        # Handle list: categorical choices
        elif isinstance(param_spec, list):
            return OptunaParam("categorical", choices=param_spec)

        # Handle dict: explicit specification
        elif isinstance(param_spec, dict):
            return self._parse_dict_spec(param_name, param_spec, warnings)

        # Handle single value: fixed parameter (not recommended but supported)
        else:
            warnings.append(
                f"Parameter '{param_name}' has fixed value {param_spec}. Consider using categorical with single choice."
            )
            return OptunaParam("categorical", choices=[param_spec])

    def _parse_continuous_range(
        self,
        param_name: str,
        low: Union[int, float],
        high: Union[int, float],
        warnings: List[str],
    ) -> OptunaParam:
        """Parse continuous range specification"""

        # Determine if should be int or float
        if isinstance(low, int) and isinstance(high, int):
            suggest_method = "int"
            kwargs = {"low": low, "high": high}
        else:
            suggest_method = "float"
            kwargs = {"low": float(low), "high": float(high)}

            # Check if log scale would be appropriate
            if self._should_use_log_scale(param_name, low, high):
                kwargs["log"] = True
                warnings.append(
                    f"Using log scale for '{param_name}' (range spans multiple orders of magnitude)"
                )

        # Apply algorithm-specific constraints
        if self.algorithm and param_name in self._algorithm_constraints:
            constraints = self._algorithm_constraints[param_name]
            if "log" in constraints:
                kwargs["log"] = constraints["log"]

        return OptunaParam(suggest_method, **kwargs)

    def _parse_dict_spec(
        self, param_name: str, spec: Dict[str, Any], warnings: List[str]
    ) -> OptunaParam:
        """Parse explicit dictionary specification"""

        # Handle explicit type specification
        if "type" in spec:
            param_type = spec["type"]

            if param_type == "float":
                kwargs = {k: v for k, v in spec.items() if k != "type"}
                return OptunaParam("float", **kwargs)

            elif param_type == "int":
                kwargs = {k: v for k, v in spec.items() if k != "type"}
                return OptunaParam("int", **kwargs)

            elif param_type == "categorical":
                if "choices" not in spec:
                    raise ValueError(
                        f"Categorical parameter '{param_name}' must specify 'choices'"
                    )
                return OptunaParam("categorical", choices=spec["choices"])

            else:
                raise ValueError(
                    f"Unknown parameter type '{param_type}' for parameter '{param_name}'"
                )

        # Handle implicit specification based on keys
        elif "low" in spec and "high" in spec:
            suggest_method = "int" if spec.get("int_type", False) else "float"
            kwargs = {k: v for k, v in spec.items() if k not in ["int_type"]}
            return OptunaParam(suggest_method, **kwargs)

        elif "choices" in spec:
            return OptunaParam("categorical", choices=spec["choices"])

        else:
            raise ValueError(
                f"Invalid parameter specification for '{param_name}': {spec}"
            )

    def _should_use_log_scale(self, param_name: str, low: float, high: float) -> bool:
        """Determine if log scale should be used based on range"""
        if low <= 0:
            return False

        # Use log scale if range spans more than 2 orders of magnitude
        ratio = high / low
        return ratio >= 100

    def _get_algorithm_constraints(self) -> Dict[str, Any]:
        """Get constraints for the current algorithm"""
        if self.algorithm and self.algorithm in self._ALGORITHM_CONSTRAINTS:
            return self._ALGORITHM_CONSTRAINTS[self.algorithm]
        return {}

    @classmethod
    def from_algorithm(cls, algorithm: str, **kwargs) -> "ParamParser":
        """Convenience constructor with algorithm-specific defaults"""
        return cls(algorithm=algorithm, **kwargs)

    @staticmethod
    def quick_parse(
        param_ranges: Dict[str, Any], algorithm: str = None
    ) -> Dict[str, OptunaParam]:
        """Simple function interface for basic use cases"""
        parser = ParamParser(algorithm=algorithm)
        result = parser.parse(param_ranges)

        if result.warnings:
            for warning in result.warnings:
                logger.warning(warning)

        return result.optuna_params
