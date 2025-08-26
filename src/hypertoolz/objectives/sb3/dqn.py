import logging
from typing import Any, Dict

from stable_baselines3 import DQN

from .base import SB3BaseObjective

logger = logging.getLogger(__name__)


# ***** Scaffolding for DQN network *****
class DQNObjectiveSB3(SB3BaseObjective):
    """DQN-specific objective using stable-baselines3."""

    def create_model(self, params: Dict[str, Any]):
        # TODO install SB3 natively - likely to make the default RL lib in the ObjectiveFactory
        # try:
        #     from stable_baselines3 import DQN
        # except ImportError:
        #     raise ImportError(
        #         "stable-baselines3 is required for DQNObjectiveSB3. "
        #         "Install with: pip install hypertoolz[to-be-named]"
        #     )

        model_kwargs = {
            "policy": self._auto_select_policy(),
            "env": self.env,
            "seed": self.seed,
            "verbose": self.verbose,
            **self.algorithm_kwargs,
            **params,
        }

        return DQN(**model_kwargs)

    def _auto_select_policy(self) -> str:
        """Auto-select policy for DQN (similar to PPO but DQN-specific)."""
        obs_space = self.env.observation_space

        # Image observations (likely Atari)
        if hasattr(obs_space, "shape") and len(obs_space.shape) == 3:
            return "CnnPolicy"
        # Vector observations
        else:
            return "MlpPolicy"
