from hypertoolz.core.tuner_config import TunerConfig


def test_default_config() -> None:

    # Arrange, Act
    config = TunerConfig()

    # Assert
    assert config.num_trials == 100
    assert config.num_jobs == 1
    assert config.startup_trials == 5
    assert config.evaluations == 2
    assert config.budget == 20000
    assert config.num_eval_envs == 5
    assert config.num_eval_eps == 10
    assert config.timeout == int(60 * 15)


def test_can_instantiate_config_with_constructor() -> None:

    # Arrange, Act
    custom_config = TunerConfig(
        num_trials=200, startup_trials=15, budget=40000, num_eval_eps=7
    )

    # Assert

    # Custom configs from constructor
    assert custom_config.num_trials == 200
    assert custom_config.startup_trials == 15
    assert custom_config.budget == 40000
    assert custom_config.num_eval_eps == 7

    # Unspecified configs remain the same
    assert custom_config.num_jobs == 1
    assert custom_config.evaluations == 2
    assert custom_config.num_eval_envs == 5
    assert custom_config.timeout == int(60 * 15)


def test_can_read_config_from_yaml(config_yaml_path: str) -> None:

    # Arrange, Act
    yaml_config = TunerConfig.from_yaml(config_yaml_path)

    # Assert
    assert yaml_config.num_trials == 150
    assert yaml_config.num_jobs == 4
    assert yaml_config.startup_trials == 10
    assert yaml_config.evaluations == 5
    assert yaml_config.budget == 50000
    assert yaml_config.num_eval_envs == 8
    assert yaml_config.num_eval_eps == 20
    assert yaml_config.timeout == 1800


def test_output_repr() -> None:
    pass
