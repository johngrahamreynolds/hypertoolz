from typing import Any, Dict, List

import pytest

from hypertoolz.core.types import ParseResult
from hypertoolz.parsers.base import BaseParser


def test_base_parser_is_abstract() -> None:
    # Arrange, Act, Assert
    with pytest.raises(TypeError):
        faulty_parser = BaseParser()


def test_base_parser_can_subclass() -> None:
    # Arrange, Act
    class Subclass(BaseParser):
        def parse(self, input_data: Dict[str, Any]) -> ParseResult:
            pass

        def validate_input(self, input_data: Dict[str, Any]) -> List[str]:
            pass

        def supported_input_types(self) -> List[str]:
            pass

    # Assert
    assert issubclass(Subclass, BaseParser)
