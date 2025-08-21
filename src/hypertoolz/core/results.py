import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import optuna
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResults:
    """
    Container for hyperparameter optimization results.

    Provides easy access to optimization outcomes, analysis methods,
    and utilities for saving/loading results.
    """

    study: optuna.Study
    best_model: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and set up the results object"""
        if self.study is None:
            raise ValueError("Study cannot be None")

        if not self.study.trials:
            logger.warning("Study contains no completed trials")

        # Add creation metadata if not provided
        if "created_at" not in self.metadata:
            import datetime

            self.metadata["created_at"] = datetime.datetime.now().isoformat()

    # --- Core Properties ---

    @property
    def best_params(self) -> Dict[str, Any]:
        """Get the best hyperparameters found during optimization"""
        try:
            return self.study.best_params
        except ValueError as e:
            logger.error("No completed trials found in study")
            raise ValueError(
                "No best parameters available - no trials completed successfully"
            ) from e

    @property
    def best_value(self) -> float:
        """Get the best objective value achieved during optimization"""
        try:
            return self.study.best_value
        except ValueError as e:
            logger.error("No completed trials found in study")
            raise ValueError(
                "No best value available - no trials completed successfully"
            ) from e

    @property
    def best_trial(self) -> optuna.Trial:
        """Get the best trial object"""
        try:
            return self.study.best_trial
        except ValueError as e:
            logger.error("No completed trials found in study")
            raise ValueError(
                "No best trial available - no trials completed successfully"
            ) from e

    @property
    def n_trials(self) -> int:
        """Total number of trials run"""
        return len(self.study.trials)

    @property
    def n_completed_trials(self) -> int:
        """Number of successfully completed trials"""
        return len(
            [
                t
                for t in self.study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            ]
        )

    @property
    def n_failed_trials(self) -> int:
        """Number of failed trials"""
        return len(
            [t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL]
        )

    @property
    def n_pruned_trials(self) -> int:
        """Number of pruned trials"""
        return len(
            [t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]
        )

    # --- Data Access Methods ---

    def trials_dataframe(self) -> pd.DataFrame:
        """
        Get all trials as a pandas DataFrame.

        Returns:
            DataFrame with trial data including parameters, values, and states
        """
        try:
            df = self.study.trials_dataframe()
            logger.debug(f"Created DataFrame with {len(df)} trials")
            return df
        except Exception as e:
            logger.error(f"Failed to create trials DataFrame: {e}")
            raise

    def get_trial_params(self, trial_number: int) -> Dict[str, Any]:
        """
        Get parameters for a specific trial.

        Args:
            trial_number: Trial number to retrieve

        Returns:
            Dictionary of parameter names and values
        """
        if trial_number >= len(self.study.trials):
            raise ValueError(
                f"Trial {trial_number} does not exist. Only {len(self.study.trials)} trials available."
            )

        trial = self.study.trials[trial_number]
        return trial.params

    def get_completed_trials(self) -> List[optuna.Trial]:
        """Get all successfully completed trials"""
        return [
            t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]

    def get_failed_trials(self) -> List[optuna.Trial]:
        """Get all failed trials"""
        return [t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL]

    # --- Analysis Methods ---

    def parameter_importance(self) -> Dict[str, float]:
        """
        Calculate parameter importance scores.

        Returns:
            Dictionary mapping parameter names to importance scores (0-1)
        """
        try:
            if self.n_completed_trials < 2:
                logger.warning(
                    "Need at least 2 completed trials for parameter importance analysis"
                )
                return {}

            importance = optuna.importance.get_param_importances(self.study)
            logger.debug(f"Calculated importance for {len(importance)} parameters")
            return importance
        except Exception as e:
            logger.error(f"Failed to calculate parameter importance: {e}")
            return {}

    def get_optimization_history(self) -> List[float]:
        """
        Get the history of best values throughout optimization.

        Returns:
            List of best values at each trial
        """
        completed_trials = self.get_completed_trials()
        if not completed_trials:
            return []

        history = []
        best_so_far = (
            float("-inf")
            if self.study.direction == optuna.study.StudyDirection.MAXIMIZE
            else float("inf")
        )

        for trial in completed_trials:
            if self.study.direction == optuna.study.StudyDirection.MAXIMIZE:
                best_so_far = max(best_so_far, trial.value)
            else:
                best_so_far = min(best_so_far, trial.value)
            history.append(best_so_far)

        return history

    def summary(self) -> Dict[str, Any]:
        """
        Get a summary of optimization results.

        Returns:
            Dictionary with key statistics and information
        """
        summary_data = {
            "study_name": self.study.study_name,
            "direction": self.study.direction.name,
            "n_trials": self.n_trials,
            "n_completed_trials": self.n_completed_trials,
            "n_failed_trials": self.n_failed_trials,
            "n_pruned_trials": self.n_pruned_trials,
            "metadata": self.metadata,
        }

        # Add best results if available
        try:
            summary_data.update(
                {
                    "best_value": self.best_value,
                    "best_params": self.best_params,
                    "best_trial_number": self.best_trial.number,
                }
            )
        except ValueError:
            logger.debug("No completed trials available for best results")

        return summary_data

    # --- Persistence Methods ---

    def save_study(self, filepath: Union[str, Path]) -> None:
        """
        Save the Optuna study to disk.

        Args:
            filepath: Path where to save the study (will be pickled)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(filepath, "wb") as f:
                pickle.dump(self.study, f)
            logger.info(f"Study saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save study to {filepath}: {e}")
            raise

    def save_results_json(self, filepath: Union[str, Path]) -> None:
        """
        Save optimization results summary to JSON.

        Args:
            filepath: Path where to save the JSON file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            summary_data = self.summary()

            # Convert non-serializable objects to strings
            if "direction" in summary_data:
                summary_data["direction"] = str(summary_data["direction"])

            with open(filepath, "w") as f:
                json.dump(summary_data, f, indent=2, default=str)
            logger.info(f"Results summary saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save results to {filepath}: {e}")
            raise

    def save_best_model(self, filepath: Union[str, Path]) -> None:
        """
        Save the best model to disk.

        Args:
            filepath: Path where to save the model

        Raises:
            ValueError: If no best model is available
        """
        if self.best_model is None:
            raise ValueError("No best model available to save")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Try common RL model save methods
            if hasattr(self.best_model, "save"):
                # Stable-baselines3 style
                self.best_model.save(str(filepath))
            elif hasattr(self.best_model, "save_model"):
                # Alternative save method
                self.best_model.save_model(str(filepath))
            else:
                # Fallback to pickle
                with open(filepath, "wb") as f:
                    pickle.dump(self.best_model, f)

            logger.info(f"Best model saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save best model to {filepath}: {e}")
            raise

    @classmethod
    def load_study(cls, filepath: Union[str, Path]) -> "OptimizationResults":
        """
        Load a previously saved study.

        Args:
            filepath: Path to the saved study file

        Returns:
            OptimizationResults object with loaded study
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Study file not found: {filepath}")

        try:
            with open(filepath, "rb") as f:
                study = pickle.load(f)

            logger.info(f"Study loaded from {filepath}")
            return cls(study=study)
        except Exception as e:
            logger.error(f"Failed to load study from {filepath}: {e}")
            raise

    # --- Display Methods ---

    def __str__(self) -> str:
        """String representation of optimization results"""
        try:
            return (
                f"OptimizationResults("
                f"trials={self.n_trials}, "
                f"best_value={self.best_value:.4f}, "
                f"completed={self.n_completed_trials})"
            )
        except ValueError:
            return f"OptimizationResults(trials={self.n_trials}, no completed trials)"

    def __repr__(self) -> str:
        """Detailed string representation"""
        return (
            f"OptimizationResults("
            f"study_name='{self.study.study_name}', "
            f"n_trials={self.n_trials}, "
            f"n_completed={self.n_completed_trials}, "
            f"n_failed={self.n_failed_trials}, "
            f"n_pruned={self.n_pruned_trials})"
        )

    def print_summary(self) -> None:
        """Print a formatted summary of the optimization results"""
        print("=" * 50)
        print("HYPERPARAMETER OPTIMIZATION RESULTS")
        print("=" * 50)

        summary = self.summary()

        print(f"Study Name: {summary['study_name']}")
        print(f"Direction: {summary['direction']}")
        print(f"Total Trials: {summary['n_trials']}")
        print(f"  ✓ Completed: {summary['n_completed_trials']}")
        print(f"  ✗ Failed: {summary['n_failed_trials']}")
        print(f"  ⚡ Pruned: {summary['n_pruned_trials']}")

        if "best_value" in summary:
            print(f"\nBest Value: {summary['best_value']:.6f}")
            print(f"Best Trial: #{summary['best_trial_number']}")
            print(f"Best Parameters:")
            for param, value in summary["best_params"].items():
                print(f"  {param}: {value}")
        else:
            print("\nNo completed trials available")

        if self.metadata:
            print(f"\nMetadata: {self.metadata}")

        print("=" * 50)
