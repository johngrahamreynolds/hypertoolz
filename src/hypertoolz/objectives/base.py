import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import optuna

from ..callbacks.eval_callback import EvalCallback
from ..core.config import TunerConfig

logger = logging.getLogger(__name__)


class BaseObjective(ABC):
    """
    Abstract base class for RL algorithm objective functions.

    This class defines the interface for training and evaluating RL algorithms
    during hyperparameter optimization. Concrete subclass implementations handle
    specific algorithms (PPO, DQN, SAC, etc.).
    """

    def __init__(
        self,
        env: Any,
        config: "TunerConfig",
        algorithm_kwargs: Optional[Dict[str, Any]] = None,
        eval_callback: Optional["EvalCallback"] = None,
        seed: Optional[int] = None,
        verbose: int = 1,
    ):
        """
        Initialize the objective function.

        Args:
            env: Environment instance (from EnvironmentResolver)
            config: TunerConfig with optimization settings
            algorithm_kwargs: Additional algorithm-specific arguments
            eval_callback: Optional callback for evaluation events
            seed: Random seed for reproducibility
            verbose: Verbosity level (0=silent, 1=info, 2=debug)
        """
        self.env = env
        self.config = config
        self.algorithm_kwargs = algorithm_kwargs or {}
        self.eval_callback = eval_callback
        self.seed = seed
        self.verbose = verbose

        # State tracking
        self.current_trial = None
        self.current_model = None
        self.best_model = None
        self.best_value = float("-inf")
        self.trial_start_time = None

        # Evaluation settings
        self.eval_env = None
        self.eval_freq = self.config.eval_freq or (
            self.config.budget // self.config.evaluations
        )

        logger.debug(
            f"Initialized {self.__class__.__name__} with eval_freq={self.eval_freq}"
        )

    def __call__(self, trial: optuna.Trial, parsed_params: Dict[str, Any]) -> float:
        """
        Main objective function called by Optuna.

        Args:
            trial: Optuna trial object
            parsed_params: Parameter ranges parsed by ParamParser

        Returns:
            Objective value (higher is better for maximize, lower for minimize)
        """
        self.current_trial = trial
        self.trial_start_time = time.time()

        try:
            # Suggest hyperparameters using parsed parameter specifications
            params = self._suggest_hyperparameters(trial, parsed_params)

            # Notify callback of trial start
            if self.eval_callback:
                self.eval_callback.on_trial_start(trial.number, params)

            # Create model with suggested hyperparameters
            model = self.create_model(params)
            self.current_model = model

            # Train the model
            trained_model = self.train_model(model, trial)

            # Evaluate the trained model
            final_score = self.evaluate_model(trained_model)

            # Update best model tracking
            if final_score > self.best_value:
                self.best_value = final_score
                self.best_model = trained_model

            # Notify callback of trial completion
            if self.eval_callback:
                self.eval_callback.on_trial_end(trial.number, final_score)

            # Log trial results
            trial_duration = time.time() - self.trial_start_time
            logger.info(
                f"Trial {trial.number}: score={final_score:.4f}, "
                f"duration={trial_duration:.1f}s, params={params}"
            )

            return final_score

        except optuna.TrialPruned:
            logger.info(f"Trial {trial.number} was pruned")
            if self.eval_callback:
                self.eval_callback.on_trial_pruned(trial.number)
            raise

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {str(e)}")
            if self.eval_callback:
                self.eval_callback.on_trial_error(trial.number, str(e))
            # Re-raise the exception to mark trial as failed
            raise

        finally:
            # Cleanup
            self.current_trial = None
            self.current_model = None

    def _suggest_hyperparameters(
        self, trial: optuna.Trial, parsed_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Use trial to suggest hyperparameters based on parsed parameter specs.

        Args:
            trial: Optuna trial object
            parsed_params: Dict mapping param names to OptunaParam objects

        Returns:
            Dictionary of suggested hyperparameter values
        """
        suggested_params = {}

        for param_name, optuna_param in parsed_params.items():
            try:
                # Use the OptunaParam's suggest method
                value = optuna_param.suggest(trial, param_name)
                suggested_params[param_name] = value

            except Exception as e:
                logger.error(f"Failed to suggest parameter '{param_name}': {e}")
                raise ValueError(
                    f"Parameter suggestion failed for '{param_name}': {e}"
                ) from e

        logger.debug(f"Suggested parameters: {suggested_params}")
        return suggested_params

    @abstractmethod
    def create_model(self, params: Dict[str, Any]) -> Any:
        """
        Create RL model instance with given hyperparameters.

        Args:
            params: Dictionary of hyperparameter values

        Returns:
            Initialized model instance
        """
        pass

    @abstractmethod
    def train_model(self, model: Any, trial: optuna.Trial) -> Any:
        """
        Train the model for the specified budget.

        Args:
            model: Model instance to train
            trial: Optuna trial (for pruning and intermediate reporting)

        Returns:
            Trained model instance
        """
        pass

    def evaluate_model(
        self, model: Any, n_episodes: Optional[int] = None, deterministic: bool = True
    ) -> float:
        """
        Evaluate trained model and return performance score.

        Args:
            model: Trained model to evaluate
            n_episodes: Number of episodes for evaluation (uses config default if None)
            deterministic: Whether to use deterministic policy

        Returns:
            Average reward over evaluation episodes
        """
        n_episodes = n_episodes or self.config.num_eval_eps

        if self.eval_env is None:
            self.eval_env = self._create_eval_env()

        logger.debug(f"Evaluating model over {n_episodes} episodes")

        total_reward = 0.0
        episode_rewards = []

        for episode in range(n_episodes):
            obs, _ = self.eval_env.reset(
                seed=self.seed + episode if self.seed else None
            )
            episode_reward = 0.0
            done = False

            while not done:
                # Get action from model (implementation depends on RL library)
                action = self._get_model_action(model, obs, deterministic)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward

            episode_rewards.append(episode_reward)
            total_reward += episode_reward

        avg_reward = total_reward / n_episodes
        std_reward = float(np.std(episode_rewards)) if len(episode_rewards) > 1 else 0.0

        logger.debug(f"Evaluation: avg_reward={avg_reward:.3f}, std={std_reward:.3f}")

        return avg_reward

    @abstractmethod
    def _get_model_action(
        self, model: Any, obs: Any, deterministic: bool = True
    ) -> Any:
        """
        Get action from model for given observation.

        Args:
            model: Trained model
            obs: Observation from environment
            deterministic: Whether to use deterministic policy

        Returns:
            Action to take in environment
        """
        pass

    def _create_eval_env(self) -> Any:
        """
        Create evaluation environment (can be same as training env or different).

        Returns:
            Environment instance for evaluation
        """
        # By default, use the same environment as training
        # Subclasses can override for vectorized eval envs, etc.
        return self.env

    def should_prune_trial(
        self, trial: optuna.Trial, step: int, intermediate_value: float
    ) -> bool:
        """
        Determine if trial should be pruned based on intermediate results.

        Args:
            trial: Current trial
            step: Training step number
            intermediate_value: Intermediate performance value

        Returns:
            True if trial should be pruned
        """
        # Report intermediate value to Optuna
        trial.report(intermediate_value, step)

        # Let Optuna's pruner decide
        return trial.should_prune()

    def save_model(self, model: Any, filepath: Union[str, Path]) -> None:
        """
        Save model to disk.

        Args:
            model: Model to save
            filepath: Path where to save model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Try common RL model save methods
            if hasattr(model, "save"):
                model.save(str(filepath))
            elif hasattr(model, "save_model"):
                model.save_model(str(filepath))
            else:
                # Fallback to pickle
                import pickle

                with open(filepath, "wb") as f:
                    pickle.dump(model, f)

            logger.info(f"Model saved to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save model to {filepath}: {e}")
            raise

    def get_trial_info(self) -> Dict[str, Any]:
        """
        Get information about the current trial.

        Returns:
            Dictionary with trial information
        """
        if self.current_trial is None:
            return {}

        trial_duration = (
            time.time() - self.trial_start_time if self.trial_start_time else 0
        )

        return {
            "trial_number": self.current_trial.number,
            "trial_duration": trial_duration,
            "params": self.current_trial.params,
            "intermediate_values": self.current_trial.intermediate_values,
            "state": self.current_trial.state.name,
        }

    def cleanup(self) -> None:
        """Clean up resources (environments, models, etc.)"""
        if hasattr(self.env, "close"):
            self.env.close()

        if self.eval_env and hasattr(self.eval_env, "close"):
            self.eval_env.close()

        logger.debug("Objective cleanup completed")

    def __del__(self):
        """Ensure cleanup on object destruction"""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore cleanup errors during destruction
