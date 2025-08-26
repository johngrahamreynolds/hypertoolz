import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

logger = logging.getLogger(__name__)


class TunerConfig:
    """
    A configuration class to pass all parameters necessary for defining the general tuning process.

     Args:
        num_trials (`int`, *optional*, defaults to 100):
            Maximum number of tuning trials to run.
        num_jobs (`int`, *optional*, defaults to 1):
            Number of jobs to run in parallel.
        startup_trials (`int`, *optional*, defaults to 5):
            Number of randomly sampled startup trials to run before tuning.
        evaluations (`int`, *optional*, defaults to 2):
            Number of evaluations to run during the training.
        budget (`int`, *optional*, defaults to 20000):
            Training budegt for (all?) trials.
        num_eval_envs (`int`, *optional*, defaults to 5):
            Number of evaluation environments to stack during evaluation.
        num_eval_eps (`int`, *optional*, defaults to 10):
            Number of evaluation episodes
        timeout (`int`, *optional*, defaults to 15 minutes (`int(60*15)`))
            Tuning timeout time.
        param_dict ('dict` *optional*, defaults to `None`)
            A dictionary of the above parameters to be passed to the model for object instatiation. Params passed explicitly in the constructor \
            take precedence over params passed in the `param_dict`. Likewise, params not passed in the dict take their default values.


    Examples:

    ```python
    >>> from hypertoolz import TunerConfig

    >>> # Initialize the TunerConfig with all the default parameters
    >>> config = TunerConfig()

    >>> Initialize a TunerConfig with chosen parameters passed directly to the constructor
    >>> config = TunerConfig(num_trials=50, num_jobs=2, evaluations=5, budget=50000)

    >>> Initialize a TunerConfig with chosen parameters passed as a Python dictionary
    >>> config_dict = { 'num_trials': 200, 'num_jobs': 8,'budget': 100000}
    >>> config = TunerConfig(param_dict=config_dict)

    >>> A mixture of the two methods above. Params passed explicity to the constructor take precedence
    >>> config_dict = { 'num_trials': 200, 'num_jobs': 8,'budget': 100000}
    >>> config = TunerConfig(param_dict=config_dict, num_trials=150) # config.num_trials = 150

    >>> Initialize a TunerConfig with chosen parameters from a yaml file
    >>> yaml_config = TunerConfig.from_yaml("/path/to/your_yaml.yaml")
    ```

    """

    def __init__(
        self,
        num_trials: Optional[int] = 100,
        num_jobs: Optional[int] = 1,
        startup_trials: Optional[int] = 5,
        evaluations: Optional[int] = 2,
        budget: Optional[int] = int(2e4),
        num_eval_envs: Optional[int] = 5,
        num_eval_eps: Optional[int] = 10,
        timeout: Optional[int] = int(15 * 60),
        param_dict: Optional[Dict[str, Any]] = None,
    ):
        # If param_dict is provided, use it to override defaults and explicit parameters
        if param_dict is not None:
            if not isinstance(param_dict, dict):
                raise TypeError(
                    f"param_dict must be a dictionary, got {type(param_dict)}"
                )

            # Extract and validate parameters from the dict
            dict_params = self._extract_config_params(param_dict, Path("param_dict"))

            # Merge explicit parameters with dict parameters (explicit params take precedence)
            # This allows users to pass both param_dict and explicit parameters
            final_params = {}

            # Start with dict parameters
            final_params.update(dict_params)

            # Override with explicitly passed parameters (if they're not default values)
            defaults = {
                "num_trials": 100,
                "num_jobs": 1,
                "startup_trials": 5,
                "evaluations": 2,
                "budget": int(2e4),
                "num_eval_envs": 5,
                "num_eval_eps": 10,
                "timeout": int(15 * 60),
            }

            explicit_params = {
                "num_trials": num_trials,
                "num_jobs": num_jobs,
                "startup_trials": startup_trials,
                "evaluations": evaluations,
                "budget": budget,
                "num_eval_envs": num_eval_envs,
                "num_eval_eps": num_eval_eps,
                "timeout": timeout,
            }

            # Only override with explicit params that differ from defaults
            for param_name, explicit_value in explicit_params.items():
                if explicit_value != defaults[param_name]:
                    final_params[param_name] = explicit_value

            # Set attributes from final merged parameters
            self.num_trials = final_params.get("num_trials", num_trials)
            self.num_jobs = final_params.get("num_jobs", num_jobs)
            self.startup_trials = final_params.get("startup_trials", startup_trials)
            self.evaluations = final_params.get("evaluations", evaluations)
            self.budget = final_params.get("budget", budget)
            self.num_eval_envs = final_params.get("num_eval_envs", num_eval_envs)
            self.num_eval_eps = final_params.get("num_eval_eps", num_eval_eps)
            self.timeout = final_params.get("timeout", timeout)
        else:
            # Original behavior when no param_dict is provided
            self.num_trials = num_trials
            self.num_jobs = num_jobs
            self.startup_trials = startup_trials
            self.evaluations = evaluations
            self.budget = budget
            self.num_eval_envs = num_eval_envs
            self.num_eval_eps = num_eval_eps
            self.timeout = timeout

        # Calculate eval_freq after all parameters are set
        self.eval_freq = (
            int(self.budget / self.evaluations)
            if self.evaluations and self.evaluations > 0
            else None
        )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "TunerConfig":
        """
        Load TunerConfig from a YAML file.

        Args:
            path: Path to the YAML configuration file

        Returns:
            TunerConfig instance with values loaded from YAML

        Raises:
            FileNotFoundError: If the YAML file doesn't exist
            yaml.YAMLError: If the YAML file is malformed
            ValueError: If required fields are missing or invalid
            TypeError: If field types don't match expected types
        """
        path = Path(path)

        # Check if file exists
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        # Load YAML content
        try:
            with open(path, "r", encoding="utf-8") as file:
                yaml_content = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to parse YAML file {path}: {e}")
        except Exception as e:
            raise IOError(f"Failed to read file {path}: {e}")

        # Handle empty or None YAML content
        if yaml_content is None:
            logger.warning(f"Empty YAML file {path}, using default configuration")
            yaml_content = {}

        if not isinstance(yaml_content, dict):
            raise ValueError(
                f"YAML content must be a dictionary, got {type(yaml_content)}"
            )

        # Extract and validate configuration parameters
        config_params = cls._extract_config_params(yaml_content, path)

        # Create and return TunerConfig instance
        return cls(**config_params)

    @classmethod
    def _extract_config_params(
        cls, param_data: Dict[str, Any], file_path: Path
    ) -> Dict[str, Any]:
        """Extract and validate configuration parameters from YAML data."""

        # Define expected parameter types and validation rules
        param_specs = {
            "num_trials": {"type": int, "min_value": 1},
            "num_jobs": {"type": int, "min_value": 1},
            "startup_trials": {"type": int, "min_value": 0},
            "evaluations": {"type": int, "min_value": 1},
            "budget": {"type": int, "min_value": 1},
            "num_eval_envs": {"type": int, "min_value": 1},
            "num_eval_eps": {"type": int, "min_value": 1},
            "timeout": {"type": int, "min_value": 1},
        }

        config_params = {}

        for param_name, spec in param_specs.items():
            if param_name in param_data:
                value = param_data[param_name]

                # Handle None values
                if value is None:
                    config_params[param_name] = None
                    continue

                # Type validation and conversion
                expected_type = spec["type"]
                if not isinstance(value, expected_type):
                    try:
                        # Attempt type conversion
                        if expected_type == int:
                            value = int(value)
                        else:
                            value = expected_type(value)
                    except (ValueError, TypeError) as e:
                        raise TypeError(
                            f"Invalid type for '{param_name}' in {file_path}: "
                            f"expected {expected_type.__name__}, got {type(value).__name__} ({value})"
                        )

                # Range validation
                if (
                    "min_value" in spec
                    and value is not None
                    and value < spec["min_value"]
                ):
                    raise ValueError(
                        f"Invalid value for '{param_name}' in {file_path}: "
                        f"must be >= {spec['min_value']}, got {value}"
                    )

                config_params[param_name] = value

        # Check for unknown parameters
        known_params = set(param_specs.keys())
        provided_params = set(param_data.keys())
        unknown_params = provided_params - known_params

        if unknown_params:
            logger.warning(
                f"Unknown parameters in {file_path} will be ignored: {', '.join(unknown_params)}"
            )

        # Custom validation rules
        cls._validate_config_consistency(config_params, file_path)

        return config_params

    @classmethod
    def _validate_config_consistency(
        cls, params: Dict[str, Any], file_path: Path
    ) -> None:
        """Validate logical consistency between parameters."""

        # Ensure startup_trials <= num_trials
        if (
            params.get("startup_trials") is not None
            and params.get("num_trials") is not None
            and params["startup_trials"] > params["num_trials"]
        ):
            raise ValueError(
                f"Invalid configuration in {file_path}: "
                f"startup_trials ({params['startup_trials']}) cannot exceed "
                f"num_trials ({params['num_trials']})"
            )

        # Ensure evaluations is compatible with budget
        if (
            params.get("evaluations") is not None
            and params.get("budget") is not None
            and params["evaluations"] > params["budget"]
        ):
            raise ValueError(
                f"Invalid configuration in {file_path}: "
                f"evaluations ({params['evaluations']}) cannot exceed "
                f"budget ({params['budget']})"
            )

    def to_yaml(self, path: Union[str, Path]) -> None:
        """
        Save current configuration to a YAML file.

        Args:
            path: Path where to save the YAML configuration file
        """
        path = Path(path)

        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = {
            "num_trials": self.num_trials,
            "num_jobs": self.num_jobs,
            "startup_trials": self.startup_trials,
            "evaluations": self.evaluations,
            "budget": self.budget,
            "num_eval_envs": self.num_eval_envs,
            "num_eval_eps": self.num_eval_eps,
            "timeout": self.timeout,
        }

        try:
            with open(path, "w", encoding="utf-8") as file:
                yaml.dump(
                    config_dict,
                    file,
                    default_flow_style=False,
                    sort_keys=True,
                    indent=2,
                )
        except Exception as e:
            raise IOError(f"Failed to write configuration to {path}: {e}")

    def __repr__(self) -> str:
        return (
            f"TunerConfig("
            f"num_trials={self.num_trials}, "
            f"num_jobs={self.num_jobs}, "
            f"startup_trials={self.startup_trials}, "
            f"evaluations={self.evaluations}, "
            f"budget={self.budget}, "
            f"eval_freq={self.eval_freq}, "
            f"num_eval_envs={self.num_eval_envs}, "
            f"num_eval_eps={self.num_eval_eps}, "
            f"timeout={self.timeout})"
        )
