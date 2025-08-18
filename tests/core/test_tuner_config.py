from typing import ClassVar

from hypertoolz.core.tuner_config import TunerConfig


class DefaultTestConfig:
    NUM_TRIALS: ClassVar[int] = 100
    NUM_JOBS: ClassVar[int] = 1
    STARTUP_TRIALS: ClassVar[int] = 5
    EVALUATIONS: ClassVar[int] = 2
    BUDGET: ClassVar[int] = 20000
    NUM_EVAL_ENVS: ClassVar[int] = 5
    NUM_EVAL_EPS: ClassVar[int] = 10
    TIMEOUT: ClassVar[int] = int(60 * 15)


class YamlTestConfig:
    NUM_TRIALS: ClassVar[int] = 150
    NUM_JOBS: ClassVar[int] = 4
    STARTUP_TRIALS: ClassVar[int] = 10
    EVALUATIONS: ClassVar[int] = 5
    BUDGET: ClassVar[int] = 50000
    NUM_EVAL_ENVS: ClassVar[int] = 8
    NUM_EVAL_EPS: ClassVar[int] = 20
    TIMEOUT: ClassVar[int] = 1800


def test_default_config() -> None:
    # Arrange, Act
    config = TunerConfig()
    default = DefaultTestConfig()

    # Assert
    assert config.num_trials == default.NUM_TRIALS
    assert config.num_jobs == default.NUM_JOBS
    assert config.startup_trials == default.STARTUP_TRIALS
    assert config.evaluations == default.EVALUATIONS
    assert config.budget == default.BUDGET
    assert config.num_eval_envs == default.NUM_EVAL_ENVS
    assert config.num_eval_eps == default.NUM_EVAL_EPS
    assert config.timeout == default.TIMEOUT


def test_can_instantiate_config_with_constructor() -> None:

    # Arrange
    num_trials_constructor_param = 200
    startup_trials_constructor_param = 15
    budget_constructor_param = 40000
    num_eval_eps_constructor_param = 7

    # Act
    custom_config = TunerConfig(
        num_trials=num_trials_constructor_param,
        startup_trials=startup_trials_constructor_param,
        budget=budget_constructor_param,
        num_eval_eps=num_eval_eps_constructor_param,
    )
    default = DefaultTestConfig()

    # Assert

    # Custom configs from constructor
    assert custom_config.num_trials == num_trials_constructor_param
    assert custom_config.startup_trials == startup_trials_constructor_param
    assert custom_config.budget == budget_constructor_param
    assert custom_config.num_eval_eps == num_eval_eps_constructor_param

    # Unspecified configs remain the default
    assert custom_config.num_jobs == default.NUM_JOBS
    assert custom_config.evaluations == default.EVALUATIONS
    assert custom_config.num_eval_envs == default.NUM_EVAL_ENVS
    assert custom_config.timeout == default.TIMEOUT


def test_can_instantiate_config_with_dict() -> None:

    # Arrange
    config_dict = {"num_trials": 200, "num_jobs": 8, "budget": 100000}

    #  Act
    custom_config = TunerConfig(param_dict=config_dict)
    default_config = DefaultTestConfig()

    # Assert

    # Custom configs from param_dict are correct
    assert custom_config.num_trials == config_dict.get("num_trials")
    assert custom_config.num_jobs == config_dict.get("num_jobs")
    assert custom_config.budget == config_dict.get("budget")

    # Unspecified configs remain the default
    assert custom_config.startup_trials == default_config.STARTUP_TRIALS
    assert custom_config.evaluations == default_config.EVALUATIONS
    assert custom_config.num_eval_envs == default_config.NUM_EVAL_ENVS
    assert custom_config.num_eval_eps == default_config.NUM_EVAL_EPS
    assert custom_config.timeout == default_config.TIMEOUT


def test_constructor_takes_precedence() -> None:

    # Arrange
    config_dict = {"num_trials": 200, "num_jobs": 8, "budget": 50000, "timeout": 600}

    num_trials_constructor_param = 400
    budget_constructor_param = int(1e6)

    # Act
    custom_config = TunerConfig(
        num_trials=num_trials_constructor_param,
        budget=budget_constructor_param,
        param_dict=config_dict,
    )
    default = DefaultTestConfig()

    # Assert

    # Constructor params take precedence
    assert custom_config.num_trials == num_trials_constructor_param
    assert custom_config.budget == budget_constructor_param

    # param_dict params take second priority to constructor
    assert custom_config.num_jobs == config_dict.get("num_jobs")
    assert custom_config.timeout == config_dict.get("timeout")

    # Unspecified configs remain the default
    assert custom_config.startup_trials == default.STARTUP_TRIALS
    assert custom_config.evaluations == default.EVALUATIONS
    assert custom_config.num_eval_envs == default.NUM_EVAL_ENVS
    assert custom_config.num_eval_eps == default.NUM_EVAL_EPS


def test_can_read_config_from_yaml(config_yaml_path: str) -> None:

    # Arrange, Act
    read_from_yaml_config = TunerConfig.from_yaml(config_yaml_path)
    yaml_config = YamlTestConfig()

    # Assert
    assert read_from_yaml_config.num_trials == yaml_config.NUM_TRIALS
    assert read_from_yaml_config.num_jobs == yaml_config.NUM_JOBS
    assert read_from_yaml_config.startup_trials == yaml_config.STARTUP_TRIALS
    assert read_from_yaml_config.evaluations == yaml_config.EVALUATIONS
    assert read_from_yaml_config.budget == yaml_config.BUDGET
    assert read_from_yaml_config.num_eval_envs == yaml_config.NUM_EVAL_ENVS
    assert read_from_yaml_config.num_eval_eps == yaml_config.NUM_EVAL_EPS
    assert read_from_yaml_config.timeout == yaml_config.TIMEOUT


def test_output_repr() -> None:
    pass
