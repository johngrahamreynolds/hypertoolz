import logging
from typing import Any, Dict, List, Optional, Union

from gymnasium import Env

from .exceptions import EnvironmentError

logger = logging.getLogger(__name__)


class EnvironmentResolver:
    """
    Resolves environment specifications to actual environment instances.

    Supports:
    - Modern gymnasium environments
    - ALE/Atari environments with automatic detection
    - Custom environment configurations
    - Environment type detection and metadata
    """

    # Environment type patterns for automatic detection
    _ENV_TYPE_PATTERNS = {
        "atari": [
            "NoFrameskip-v4",
            "NoFrameskip-v0",
            "Deterministic-v4",
            "ALE/",
            "Breakout",
            "Pong",
            "SpaceInvaders",
            "MsPacman",
            "Qbert",
            "Seaquest",
            "Enduro",
            "BeamRider",
            "Freeway",
            "Frostbite",
            "Kangaroo",
            "Skiing",
            "Tennis",
            "VideoPinball",
        ],
        "classic_control": ["CartPole", "MountainCar", "Acrobot", "Pendulum"],
        "box2d": [
            "LunarLander",
            "BipedalWalker",
            "CarRacing",
            "CarDynamics",
        ],  # TODO need to install box2d-py
        "mujoco": [
            "Ant",
            "HalfCheetah",
            "Hopper",
            "Humanoid",
            "HumanoidStandup",
            "InvertedPendulum",
            "InvertedDoublePendulum",
            "Pusher",
            "Reacher",
            "Swimmer",
            "Walker",
        ],
    }

    def __init__(self):
        self._gymnasium = None
        self._ale_py = None
        self._initialize_backends()

    def _initialize_backends(self) -> None:
        """Initialize environment backends (all guaranteed available as native dependencies)"""

        # Gymnasium is a native dependency
        import gymnasium as gym

        self._gymnasium = gym
        logger.debug("Gymnasium backend initialized")

        # ALE is a native dependency
        import ale_py

        self._ale_py = ale_py
        logger.debug("ALE backend initialized")

        # Register ALE environments with gymnasium
        try:
            gym.register_envs(ale_py)
            logger.debug("ALE environments registered with gymnasium")
        except Exception as e:
            logger.warning(f"Failed to register ALE environments: {e}")
            # Continue anyway - basic gym envs will still work

    def resolve(
        self,
        env_spec: Union[str, Dict[str, Any], Any],
        backend: Optional[
            str
        ] = None,  # TODO to be implemented later for custom backend support
        **kwargs,
    ) -> Env:
        """
        Resolve environment specification to environment instance.

        Args:
            env_spec: Environment specification (string, dict, or existing env)
            backend: Preferred backend ('gymnasium', 'gym', 'auto')
            **kwargs: Additional arguments passed to env creation

        Returns:
            Environment instance

        Raises:
            EnvironmentError: If environment cannot be resolved
        """
        try:
            # If it's already an environment instance, return as-is
            if self._is_env_instance(env_spec):
                logger.debug(
                    f"Environment spec is already an instance: {type(env_spec)}"
                )
                return env_spec

            # Handle string specifications
            if isinstance(env_spec, str):
                return self._resolve_string_spec(env_spec, backend, **kwargs)

            # Handle dictionary specifications
            elif isinstance(env_spec, dict):
                return self._resolve_dict_spec(env_spec, backend, **kwargs)

            else:
                raise ValueError(
                    f"Unsupported environment specification type: {type(env_spec)}"
                )

        except Exception as e:
            logger.error(f"Failed to resolve environment: {env_spec}")
            raise EnvironmentError(
                f"Could not resolve environment '{env_spec}': {str(e)}"
            ) from e

    def _resolve_string_spec(
        self, env_name: str, backend: Optional[str] = None, **kwargs
    ) -> Env:
        """Resolve string environment specification"""

        logger.debug(f"Resolving environment: '{env_name}'")

        # Determine best backend for this environment
        if backend is None:
            backend = self._choose_backend_for_env(env_name)

        # Create environment with chosen backend
        return self._create_env_with_backend(env_name, backend, **kwargs)

    def _resolve_dict_spec(
        self, env_spec: Dict[str, Any], backend: Optional[str] = None, **kwargs
    ) -> Union[Env, Any]:  # TODO make Union of Env and all possible Wrapped Env types
        """Resolve dictionary environment specification"""

        if "name" not in env_spec:
            raise ValueError("Dictionary environment spec must contain 'name' field")

        env_name = env_spec["name"]
        env_kwargs = env_spec.get("kwargs", {})
        env_wrappers = env_spec.get("wrappers", [])

        # Merge kwargs
        combined_kwargs = {**env_kwargs, **kwargs}

        # Create base environment
        env = self._resolve_string_spec(env_name, backend, **combined_kwargs)

        # Apply wrappers if specified
        if env_wrappers:
            env = self._apply_wrappers(env, env_wrappers)

        return env

    def _choose_backend_for_env(self, env_name: str) -> str:
        """Choose the best backend for a given environment"""

        # TODO create issue for possibly handling multiple backends in the futue - gymnasium + ALE our initial dependencies
        # # Detect environment type
        # env_type = self.detect_env_type(env_name)

        # # Atari environments: prefer gymnasium with ALE
        # if env_type == "atari":
        #     if self._gymnasium and self._ale_py:
        #         return "gymnasium"
        #     elif self._gym:
        #         return "gym"

        # # For other environments: prefer gymnasium, fallback to gym
        # if self._gymnasium:
        #     return "gymnasium"
        # elif self._gym:
        #     return "gym"

        # Always use gymnasium (it's our native dependency)
        return "gymnasium"

    def _create_env_with_backend(
        self, env_name: str, backend: str, **kwargs
    ) -> Env:  # TODO initial support for Gymnasmium Envs
        """Create environment using specified backend"""

        logger.debug(f"Creating '{env_name}' with backend '{backend}'")

        if backend == "gymnasium":
            try:
                env = self._gymnasium.make(env_name, **kwargs)
                logger.info(f"Created environment '{env_name}' using gymnasium")
                return env

            except Exception as e:
                logger.error(f"Failed to create '{env_name}' with gymnasium: {e}")
                raise

        else:
            raise ValueError(
                f"Backend '{backend}' not supported. Only 'gymnasium' is available."  # TODO future support of other backends
            )

    def _apply_wrappers(self, env: Any, wrappers: List[Dict[str, Any]]) -> Any:
        """Apply wrapper configurations to environment"""

        for wrapper_config in wrappers:
            if isinstance(wrapper_config, str):
                # Simple wrapper name
                wrapper_config = {"name": wrapper_config}

            wrapper_name = wrapper_config["name"]
            wrapper_kwargs = wrapper_config.get("kwargs", {})

            try:
                # Try to import wrapper
                if "." in wrapper_name:
                    # Full module path
                    module_path, class_name = wrapper_name.rsplit(".", 1)
                    module = __import__(module_path, fromlist=[class_name])
                    wrapper_class = getattr(module, class_name)
                else:
                    # Assume it's from stable_baselines3 or gymnasium
                    wrapper_class = self._find_wrapper_class(wrapper_name)

                env = wrapper_class(env, **wrapper_kwargs)
                logger.debug(f"Applied wrapper: {wrapper_name}")

            except Exception as e:
                logger.error(f"Failed to apply wrapper '{wrapper_name}': {e}")
                raise

        return env

    def _find_wrapper_class(
        self, wrapper_name: str
    ) -> Any:  # TODO return Union of all possible Wrapper Env types
        """Find wrapper class in common locations"""

        # Try stable_baselines3 wrappers - this is our initial RL package integration
        try:
            from stable_baselines3.common.env_util import make_atari_env
            from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

            wrapper_map = {
                "VecFrameStack": VecFrameStack,
                "DummyVecEnv": DummyVecEnv,
            }

            if wrapper_name in wrapper_map:
                return wrapper_map[wrapper_name]
        except ImportError:
            pass

        # Try gymnasium wrappers
        try:
            from gymnasium.wrappers import AtariPreprocessing, FrameStack

            wrapper_map = {
                "AtariPreprocessing": AtariPreprocessing,
                "FrameStack": FrameStack,
            }

            if wrapper_name in wrapper_map:
                return wrapper_map[wrapper_name]
        except ImportError:
            pass

        raise ImportError(f"Could not find wrapper class: {wrapper_name}")

    def _is_env_instance(self, obj: Any) -> bool:
        """Check if object is already an environment instance"""

        # Check for common environment interfaces
        has_step = hasattr(obj, "step")
        has_reset = hasattr(obj, "reset")
        has_action_space = hasattr(obj, "action_space")
        has_observation_space = hasattr(obj, "observation_space")

        return all([has_step, has_reset, has_action_space, has_observation_space])

    def detect_env_type(self, env_name: str) -> str:
        """
        Detect environment type based on name patterns.

        Args:
            env_name: Environment name

        Returns:
            Environment type ('atari', 'classic_control', 'box2d', 'mujoco', 'unknown')
        """
        env_name_upper = env_name.upper()

        for env_type, patterns in self._ENV_TYPE_PATTERNS.items():
            for pattern in patterns:
                if pattern.upper() in env_name_upper:
                    return env_type

        return "unknown"

    def get_env_metadata(self, env_spec: Union[str, Any]) -> Dict[str, Any]:
        """
        Get metadata about an environment.

        Args:
            env_spec: Environment specification or instance

        Returns:
            Dictionary with environment metadata
        """
        metadata = {}

        if isinstance(env_spec, str):
            env_name = env_spec
            metadata["name"] = env_name
            metadata["type"] = self.detect_env_type(env_name)
            metadata["backend"] = "gymnasium"
            # metadata["backend"] = self._choose_backend_for_env(env_name) TODO for future handling of multiple backends
        else:
            # Try to extract metadata from environment instance
            if hasattr(env_spec, "spec") and hasattr(env_spec.spec, "id"):
                metadata["name"] = env_spec.spec.id
                metadata["type"] = self.detect_env_type(env_spec.spec.id)
            else:
                metadata["name"] = str(type(env_spec).__name__)
                metadata["type"] = "unknown"

            metadata["backend"] = "unknown"

        # Add action/observation space info if available
        if hasattr(env_spec, "action_space"):
            metadata["action_space"] = str(env_spec.action_space)
        if hasattr(env_spec, "observation_space"):
            metadata["observation_space"] = str(env_spec.observation_space)

        return metadata

    def list_available_envs(self, env_type: Optional[str] = None) -> List[str]:
        """
        List available environments.

        Args:
            env_type: Filter by environment type ('atari', 'classic_control', etc.)

        Returns:
            List of available environment names
        """
        envs = []

        # Get registered environments from gymnasium
        try:
            from gymnasium.envs import registry

            gym_envs = list(registry.keys())
            if env_type:
                gym_envs = [
                    name for name in gym_envs if self.detect_env_type(name) == env_type
                ]
            envs.extend(gym_envs)

            return envs
        except Exception:
            logger.debug("Could not list gymnasium environments")

        return sorted(list(set(envs)))

    @property
    def available_backends(self) -> List[str]:
        """Get list of available backends"""
        return [
            "gymnasium",
            "ale_py",
        ]  # TODO add support for OpenAI's legacy gym environment in future update
