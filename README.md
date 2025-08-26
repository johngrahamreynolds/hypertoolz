# hypertoolz

The `hypertoolz` package houses a set of simple, reusable modules for efficiently running hyperparameter tuning on reinforcement learning (RL) optimization problems with native integration with common RL libraries. 

Much of the code produced during hyperparameter tuning can be redundant across different RL problems or vary significantly between common RL libraries (`stable_baselines3`, `ray`, `gym`/`gymnasium`, `torchrl`, etc.) - this package is designed to neatly remove much of this redundancy and library-specific variation without eliminating any core abstraction across the spectrum of RL algorithms, the space of hyperparameters, etc. The code herewithin can be thought to be akin to the same robust simplicity provided by the different packages of the Hugging Face ecosystem. Indeed, our project follows a very similar system architecture to `transformers` for transformer-based model training and inference.

In summation, `hypertoolz` is focused on efficient simplicity for hyperparameter tuning across all possible RL scenarios; the goal is to make the simple case trivial and the complex case possible.

## IMPORTANT NOTE

This package is under active development. Suggestions for implementation of specific algorithms, integration with different RL libraries, and any other ideas are welcomed. This library has been designed to be open-source for the possibility of community collaboration in the same spirit of the various libraries included in the Hugging Face ecosystem.

### Installation

The package is (almost) available (wheel construction still in progress) on the Python Package Index and can be installed easily through the command line.

```bash
python3 -m pip install hypertoolz
```

### `hypertoolz` usage across different levels of complexity

#### Trivial Usage: "Anyone can cook (code)" - Chef Gusteau, almost

    ```python

    >>> from hypertoolz import optimize

    >>>> ....
    ```

#### Experienced Usage: The Practioners Modification

    ```python

    >>> from hypertoolz import optimize
    >>> from hypertoolz import TunerConfig

    >>> param_range = {...}
    ```

#### Advanced Usage: The Researcher's Full Control

    ```python
    >>> from hypertoolz import HyperTuner
    >>> from hypertoolz.core.config import TunerConfig
    >>> from hypertoolz.parsers import ParamParser
    >>> from hypertoolz.objectives.sb3_base import DQNObjectiveSB3


    >>> ....
    ```

### System Architecture

<img width="3840" height="1777" alt="hypertoolz_system_preliminary" src="https://github.com/user-attachments/assets/38153ee6-3c52-4399-8f46-29ed2963d6d6" />


Preliminary system design as a Mermaid diagram.
