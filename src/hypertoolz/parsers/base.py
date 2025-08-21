import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from ..core.config import TunerConfig
from ..core.types import OptunaParam, ParseResult, ValidationMode

logger = logging.getLogger(__name__)


class BaseParser(ABC):
    """Abstract base class for all parameter parsers in hypertoolz"""

    def __init__(
        self,
        validation_mode: ValidationMode = ValidationMode.STRICT,
        config: Optional["TunerConfig"] = None,
    ):
        self.validation_mode = validation_mode
        self.config = config
        self._custom_types = {}

    @abstractmethod
    def parse(self, input_data: Dict[str, Any]) -> ParseResult:
        """
        Parse input data into Optuna-compatible format.

        Args:
            input_data: Raw input data to parse

        Returns:
            ParseResult containing optuna_params, warnings, and metadata
        """
        pass

    @abstractmethod
    def validate_input(self, input_data: Dict[str, Any]) -> List[str]:
        """
        Validate input data and return list of issues.

        Args:
            input_data: Raw input data to validate

        Returns:
            List of validation issues (empty if no issues)
        """
        pass

    def add_custom_type(
        self, name: str, parser_func: Callable[[Any], OptunaParam]
    ) -> None:
        """
        Add a custom parameter type parser.

        Args:
            name: Name of the custom type
            parser_func: Function that converts value to OptunaParam
        """
        self._custom_types[name] = parser_func
        logger.debug(f"Added custom type parser: {name}")

    def set_validation_mode(self, mode: ValidationMode) -> None:
        """Change validation strictness level"""
        self.validation_mode = mode
        logger.debug(f"Validation mode set to: {mode.value}")

    @property
    @abstractmethod
    def supported_input_types(self) -> List[str]:
        """Return list of supported input data types for this parser"""
        pass

    def _handle_validation_issue(self, issue: str, warnings: List[str]) -> None:
        """Handle validation issues based on current validation mode"""
        if self.validation_mode == ValidationMode.STRICT:
            raise ValueError(issue)
        elif self.validation_mode == ValidationMode.LENIENT:
            warnings.append(issue)
            logger.warning(issue)
        # NONE mode: do nothing
