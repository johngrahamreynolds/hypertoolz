from abc import ABC, abstractmethod
from typing import Any, Dict


class EvalCallback(ABC):
    """
    Abstract base class for hypertoolz evaluation callbacks, used for monitoring
    hyperparameter optimization during objective function evaluation.

    """

    @abstractmethod
    def on_trial_start(self, trial_number: int, params: Dict[str, Any]) -> None:
        """Called when a trial starts"""
        pass

    @abstractmethod
    def on_trial_end(self, trial_number: int, final_value: float) -> None:
        """Called when a trial completes"""
        pass

    @abstractmethod
    def on_evaluation_step(
        self, trial_number: int, step: int, metrics: Dict[str, float]
    ) -> None:
        """Called during intermediate evaluations"""
        pass

    @abstractmethod
    def on_trial_pruned(self, trial_number: int) -> None:
        """Called when a trial is pruned"""
        pass

    def on_trial_error(self, trial_number: int, error: str) -> None:
        """Called when a trial fails (optional to override)"""
        pass
