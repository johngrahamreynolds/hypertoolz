import logging

from .generic import DQNObjective, PPOObjective
from .ray import PPOObjectiveRay
from .sb3 import DQNObjectiveSB3, PPOObjectiveSB3

logger = logging.getLogger(__name__)


class ObjectiveFactory:
    _ALGORITHMS = {
        "PPO": {
            "sb3": PPOObjectiveSB3,
            "ray": PPOObjectiveRay,
            "generic": PPOObjective,
        },
        "DQN": {
            "sb3": DQNObjectiveSB3,
            "generic": DQNObjective,
        },
    }

    @classmethod
    def create_objective(cls, algorithm: str, library: str = "sb3", **kwargs):
        """Orchestration layer chooses the right implementation with a default of stable_baselines3"""

        if algorithm not in cls._ALGORITHMS:
            raise ValueError(f"Algorithm {algorithm} not yet supported")

        algo_variants = cls._ALGORITHMS[algorithm]

        # Prefer library-specific, fallback to generic
        if library in algo_variants:
            return algo_variants[library](**kwargs)
        elif "generic" in algo_variants:
            logger.warning(
                f"Library {library} not available, using generic implementation"
            )
            return algo_variants["generic"](**kwargs)
        else:
            raise ValueError(f"No implementation available for {algorithm}")
