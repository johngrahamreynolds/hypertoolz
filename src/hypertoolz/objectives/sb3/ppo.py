import logging
from typing import Any, Dict, Optional

from stable_baselines3 import PPO

from ...core.config import TunerConfig
from .base import SB3BaseObjective, SB3EvalCallback

logger = logging.getLogger(__name__)


class PPOObjectiveSB3(SB3BaseObjective):
    """
    PPO-specific objective using stable-baselines3.

    Implements PPO hyperparameter optimization with:
    - Automatic policy selection based on environment
    - PPO-specific parameter handling
    - Optimized training and evaluation
    """

    def __init__(
        self,
        env,
        config: "TunerConfig",
        algorithm_kwargs: Optional[Dict[str, Any]] = None,
        eval_callback: Optional["SB3EvalCallback"] = None,
        seed: Optional[int] = None,
        verbose: int = 1,
        policy: Optional[str] = None,
    ):
        super().__init__(env, config, algorithm_kwargs, eval_callback, seed, verbose)

        # PPO-specific settings
        self.policy = policy or self._auto_select_policy()

        logger.debug(f"PPOObjectiveSB3 using policy: {self.policy}")

    def create_model(self, params: Dict[str, Any]):
        """
        Create PPO model with given hyperparameters.

        Args:
            params: Hyperparameter dictionary from Optuna

        Returns:
            PPO model instance
        """

        # Merge hyperparameters with algorithm kwargs
        model_kwargs = {
            "policy": self.policy,
            "env": self.env,
            "seed": self.seed,
            "verbose": self.verbose,
            **self.algorithm_kwargs,
            **params,  # Hyperparameters override defaults
        }

        # Handle special parameter transformations
        model_kwargs = self._transform_ppo_params(model_kwargs)

        logger.info(f"Creating PPO model with params: {params}")
        logger.debug(f"Full PPO kwargs: {model_kwargs}")

        model = PPO(**model_kwargs)
        return model

    def _auto_select_policy(self) -> str:
        """
        Automatically select appropriate policy based on environment.

        Returns:
            Policy string for PPO
        """
        # Check observation space to determine policy type
        obs_space = self.env.observation_space

        # Handle different observation space types
        if hasattr(obs_space, "shape"):
            obs_shape = obs_space.shape

            # Image observations (likely Atari)
            if len(obs_shape) == 3:  # (H, W, C) or (C, H, W)
                return "CnnPolicy"

            # Vector observations
            elif len(obs_shape) == 1:
                return "MlpPolicy"

            else:
                logger.warning(
                    f"Unusual observation shape: {obs_shape}, using MlpPolicy"
                )
                return "MlpPolicy"

        else:
            logger.warning(f"Unknown observation space: {obs_space}, using MlpPolicy")
            return "MlpPolicy"

    def _transform_ppo_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform hyperparameters for PPO-specific requirements.

        Args:
            params: Raw hyperparameters

        Returns:
            Transformed parameters suitable for PPO
        """
        transformed = params.copy()

        # Handle batch_size vs n_steps relationship
        if "batch_size" in transformed and "n_steps" in transformed:
            # Ensure batch_size <= n_steps for PPO
            if transformed["batch_size"] > transformed["n_steps"]:
                logger.warning(
                    f"batch_size ({transformed['batch_size']}) > n_steps "
                    f"({transformed['n_steps']}), adjusting batch_size"
                )
                transformed["batch_size"] = transformed["n_steps"]

        # Handle policy_kwargs if specified
        if "net_arch" in transformed:
            net_arch = transformed.pop("net_arch")
            if "policy_kwargs" not in transformed:
                transformed["policy_kwargs"] = {}
            transformed["policy_kwargs"]["net_arch"] = net_arch
        if "activation_fn" in transformed:
            activation_fn = transformed.pop("activation_fn")
            if "policy_kwargs" not in transformed:
                transformed["policy_kwargs"] = {}
            transformed["policy_kwargs"]["activation_fn"] = activation_fn

        return transformed

    def get_ppo_info(self, model) -> Dict[str, Any]:
        """Get PPO-specific model information."""
        info = self.get_model_info(model)

        # Add PPO-specific info
        if hasattr(model, "n_steps"):
            info["n_steps"] = model.n_steps
        if hasattr(model, "batch_size"):
            info["batch_size"] = model.batch_size
        if hasattr(model, "n_epochs"):
            info["n_epochs"] = model.n_epochs
        if hasattr(model, "clip_range"):
            info["clip_range"] = model.clip_range

        return info
