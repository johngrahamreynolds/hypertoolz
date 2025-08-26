import logging
import time
from typing import Any, Dict, Optional

from .base import EvalCallback

logger = logging.getLogger(__name__)


class DefaultEvalCallback(EvalCallback):
    """
    Default implementation of EvalCallback with basic logging.

    Provides simple console logging for trial events. Suitable for most users
    who want basic monitoring without additional dependencies. Compatible with custom `hypertoolz` objectives
    """

    def __init__(self, verbose: int = 1):
        """
        Initialize DefaultEvalCallback.

        Args:
            verbose: Verbosity level
                0 = silent
                1 = basic trial info (default)
                2 = detailed trial info
        """
        self.verbose = verbose
        self.trial_start_times = {}
        self.trial_count = 0

        if self.verbose > 0:
            logger.info("DefaultEvalCallback initialized")

    def on_trial_start(self, trial_number: int, params: Dict[str, Any]) -> None:
        """Called when a trial starts"""
        self.trial_start_times[trial_number] = time.time()
        self.trial_count += 1

        if self.verbose >= 1:
            # Basic trial start info
            param_summary = self._format_params_summary(params)
            logger.info(f"Trial {trial_number} started: {param_summary}")

        if self.verbose >= 2:
            # Detailed parameter info
            logger.debug(f"Trial {trial_number} full parameters:")
            for param_name, param_value in params.items():
                logger.debug(f"  {param_name}: {param_value}")

    def on_trial_end(self, trial_number: int, final_value: float) -> None:
        """Called when a trial completes"""

        # Calculate trial duration
        duration = None
        if trial_number in self.trial_start_times:
            duration = time.time() - self.trial_start_times[trial_number]
            del self.trial_start_times[trial_number]  # Cleanup

        if self.verbose >= 1:
            duration_str = f" ({duration:.1f}s)" if duration else ""
            logger.info(
                f"Trial {trial_number} completed: {final_value:.4f}{duration_str}"
            )

        if self.verbose >= 2:
            logger.debug(
                f"Trial {trial_number} details: value={final_value}, duration={duration}s"
            )

    def on_evaluation_step(
        self, trial_number: int, step: int, metrics: Dict[str, float]
    ) -> None:
        """Called during intermediate evaluations"""

        if self.verbose >= 2:
            # Only show intermediate steps in verbose mode
            metrics_str = ", ".join(f"{k}={v:.3f}" for k, v in metrics.items())
            logger.debug(f"Trial {trial_number}, Step {step}: {metrics_str}")

    def on_trial_pruned(self, trial_number: int) -> None:
        """Called when a trial is pruned"""

        # Calculate trial duration if available
        duration = None
        if trial_number in self.trial_start_times:
            duration = time.time() - self.trial_start_times[trial_number]
            del self.trial_start_times[trial_number]  # Cleanup

        if self.verbose >= 1:
            duration_str = f" after {duration:.1f}s" if duration else ""
            logger.info(f"Trial {trial_number} pruned{duration_str}")

    def on_trial_error(self, trial_number: int, error: str) -> None:
        """Called when a trial fails"""

        # Calculate trial duration if available
        duration = None
        if trial_number in self.trial_start_times:
            duration = time.time() - self.trial_start_times[trial_number]
            del self.trial_start_times[
                trial_number
            ]  # Cleanup # TODO is this really necessary?

        if self.verbose >= 1:
            duration_str = f" after {duration:.1f}s" if duration else ""
            logger.error(f"Trial {trial_number} failed{duration_str}: {error}")

    def _format_params_summary(self, params: Dict[str, Any]) -> str:
        """Create a concise summary of parameters for logging"""

        if not params:
            return "no parameters"

        # Show up to 3 most "important" parameters
        # Priority: learning_rate, batch_size, then alphabetical
        priority_params = [
            "learning_rate",
            "batch_size",
            "n_epochs",
            "clip_range",
        ]  # TODO implement params of priority based on algorithm we are training with ? Or leave as fixed set?

        # Get priority parameters that exist
        summary_params = []
        for priority_param in priority_params:
            if priority_param in params:
                value = params[priority_param]
                # Format value nicely
                if isinstance(value, float) and value < 0.01:
                    summary_params.append(f"{priority_param}={value:.2e}")
                else:
                    summary_params.append(f"{priority_param}={value}")

                if len(summary_params) >= 2:  # Show max 2 params in summary
                    break

        # If we don't have enough priority params, add others
        if len(summary_params) < 2:
            remaining_params = [k for k in params.keys() if k not in priority_params]
            for param in sorted(remaining_params):
                if len(summary_params) >= 2:
                    break
                value = params[param]
                if isinstance(value, float) and value < 0.01:
                    summary_params.append(f"{param}={value:.2e}")
                else:
                    summary_params.append(f"{param}={value}")

        # Add "..." if there are more parameters
        param_summary = ", ".join(summary_params)
        if len(params) > len(summary_params):
            param_summary += f", ... ({len(params)} total)"

        return param_summary

    def get_stats(self) -> Dict[str, Any]:
        """Get callback statistics (optional utility method)"""
        return {
            "total_trials_seen": self.trial_count,
            "active_trials": len(self.trial_start_times),
            "verbose_level": self.verbose,
        }
