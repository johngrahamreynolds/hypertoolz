import logging
from typing import Any, Dict, Optional

import optuna
from stable_baselines3.common.callbacks import EvalCallback as SB3EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from ...core.config import TunerConfig
from ..base import BaseObjective

logger = logging.getLogger(__name__)


class SB3TrialEvalCallback(SB3EvalCallback):
    """
    Enhanced stable-baselines3 EvalCallback for Optuna integration.

    Extends SB3's EvalCallback to:
    - Report intermediate values to Optuna trials
    - Handle trial pruning
    - Bridge to hypertoolz callback system (when implemented)
    """

    def __init__(
        self,
        eval_env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
        **kwargs,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
            **kwargs,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        """Called by SB3 during training. Handles evaluation and pruning."""

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Perform evaluation (handled by parent class)
            super()._on_step()
            self.eval_idx += 1

            # Report intermediate result to Optuna
            mean_reward = self.last_mean_reward
            self.trial.report(mean_reward, self.eval_idx)

            logger.debug(
                f"Trial {self.trial.number}: step {self.eval_idx}, "
                f"mean_reward={mean_reward:.3f}"
            )

            # Check if trial should be pruned
            if self.trial.should_prune():
                logger.info(
                    f"Trial {self.trial.number} pruned at evaluation {self.eval_idx} "
                    f"(reward={mean_reward:.3f})"
                )
                self.is_pruned = True
                return False  # Stop training

        return True  # Continue training


class SB3BaseObjective(BaseObjective):
    """
    Base objective class for stable-baselines3 algorithms.

    Provides optimized integration with SB3's training and evaluation systems:
    - Uses SB3's native callback system for evaluation
    - Leverages SB3's optimized training loops
    - Integrates SB3's evaluation utilities
    - Handles pruning through SB3's callback mechanism
    """

    def __init__(
        self,
        env,
        config: "TunerConfig",
        algorithm_kwargs: Optional[Dict[str, Any]] = None,
        eval_callback: Optional["SB3EvalCallback"] = None,
        seed: Optional[int] = None,
        verbose: int = 1,
    ):
        super().__init__(env, config, algorithm_kwargs, eval_callback, seed, verbose)

        # SB3-specific settings
        self.sb3_callback = None
        self.model_class = None  # To be set by subclasses

        logger.debug(f"Initialized {self.__class__.__name__} for SB3 integration")

    def train_model(self, model, trial: optuna.Trial):
        """
        Train SB3 model using native callback system.

        Args:
            model: SB3 model instance
            trial: Optuna trial for pruning

        Returns:
            Trained model

        Raises:
            optuna.TrialPruned: If trial is pruned during training
        """
        # Create SB3 evaluation callback
        eval_env = self._create_eval_env()

        self.sb3_callback = SB3TrialEvalCallback(
            eval_env=eval_env,
            trial=trial,
            n_eval_episodes=self.config.num_eval_eps,
            eval_freq=self.eval_freq,
            deterministic=True,
            verbose=max(0, self.verbose - 1),  # Reduce callback verbosity
        )

        logger.info(
            f"Starting SB3 training: {self.config.budget} timesteps, "
            f"eval_freq={self.eval_freq}"
        )

        try:
            # Use SB3's native training with callback
            model.learn(
                total_timesteps=self.config.budget,
                callback=self.sb3_callback,
                reset_num_timesteps=True,
            )

            # Check if training was pruned
            if self.sb3_callback.is_pruned:
                raise optuna.TrialPruned()

            logger.info(f"SB3 training completed successfully")
            return model

        except optuna.TrialPruned:
            logger.info(f"SB3 training was pruned")
            raise
        except Exception as e:
            logger.error(f"SB3 training failed: {e}")
            raise
        finally:
            # Cleanup evaluation environment
            if hasattr(eval_env, "close"):
                eval_env.close()

    def evaluate_model(
        self, model, n_episodes: Optional[int] = None, deterministic: bool = True
    ) -> float:
        """
        Evaluate SB3 model using stable-baselines3's evaluation utilities.

        Args:
            model: Trained SB3 model
            n_episodes: Number of episodes (uses config default if None)
            deterministic: Whether to use deterministic policy

        Returns:
            Mean reward over evaluation episodes
        """
        n_episodes = n_episodes or self.config.num_eval_eps

        # Use SB3's optimized evaluation
        eval_env = self._create_eval_env()

        try:
            mean_reward, std_reward = evaluate_policy(
                model=model,
                env=eval_env,
                n_eval_episodes=n_episodes,
                deterministic=deterministic,
                render=False,
                return_episode_rewards=False,
            )

            logger.debug(
                f"SB3 evaluation: {n_episodes} episodes, "
                f"mean={mean_reward:.3f}, std={std_reward:.3f}"
            )

            return float(mean_reward)

        finally:
            if hasattr(eval_env, "close"):
                eval_env.close()

    def _get_model_action(self, model, obs, deterministic: bool = True):
        """
        Get action from SB3 model.

        Args:
            model: SB3 model
            obs: Observation from environment
            deterministic: Whether to use deterministic policy

        Returns:
            Action from model
        """
        action, _ = model.predict(obs, deterministic=deterministic)
        return action

    def _create_eval_env(self):
        """
        Create evaluation environment for SB3.

        For SB3, we typically use the same environment as training,
        but this can be overridden for vectorized environments.
        """
        # For now, use the same environment
        # In advanced implementations, this could create vectorized eval envs
        return self.env

    def get_model_info(self, model) -> Dict[str, Any]:
        """Get information about the SB3 model."""
        info = {}

        if hasattr(model, "policy"):
            info["policy_class"] = model.policy.__class__.__name__

        if hasattr(model, "learning_rate"):
            info["learning_rate"] = model.learning_rate

        if hasattr(model, "n_envs"):
            info["n_envs"] = model.n_envs

        if hasattr(model, "_n_updates"):
            info["n_updates"] = model._n_updates

        return info
