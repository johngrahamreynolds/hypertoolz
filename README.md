# Hypertoolz

The `hypertoolz` package houses a set of simple, reusable modules for efficiently running hyperparameter tuning with Optuna on reinforcement learning (RL) optimization problems. 

Much of the code produced during hyperparameter tuning with Optuna can be redundant - this package is designed to neatly remove much of this redundancy without eliminating any core abstraction across the spectrum of RL algorithms, the space of all possible hyperparameters, etc. The code herewithin can be thought to be akin to the same simplicity provided by something like Hugging Face `transformers` in circumstantial favor of the larger, more complete, but much more verbose `torch` atop which it is built.

The goal of `hypertoolz` is efficient simplicity!

## IMPORTANT NOTE

This package is under active development. Its planned completion is by the end of the current month, August, 2025.

### Installation

The package is available on the Python Package Index and can be installed easily through the command line.

```bash
python3 -m pip3 install hypertoolz
```



### Introduction

Commonly, when using `optuna` for hyperparameter tuning, one constructs 4 core Python objects (functions, classes, etc.) that are largely redundant across different optimization problems. Roughly speaking, these entities are a `sampler`, a `callback`, an `objective fn`, and an `optimization loop`. Additionally, one needs default configuration variables which form the scaffolding of the RL model. The section below compares the succinct usage of `hypertoolz` to the traditional, verbose usage of `optuna` through a detailed sample problem.


### Hypertoolz vs Optuna: A Full Tuning

Imagine you want to train a model to excel at the `CartPole` environment task available in OpenAI's `Gym` (now maintained by the Farama Foundation in the `Gymnasium` package). You want to train your model to the best result possible, but you're unsure of what hyperparameters to choose. To confront this confusion, you make the wise decision to do an algorithmic search across the space of all possible hyperparameters to find what set(s) of parameters will most likely give you the best model after a full training.

Before you begin, you write down a range of hyperparameters you're interested in tuning to be used for training your RL model. This initial set of parameters can be dependent upon the chosen method of RL. For this example, let's choose the `A2C` (Advantage Actor Critic) RL method. It must be noted, however, that `hypertoolz` can be efficiently used by any RL method native to RL packages like `stable_baselines3`, or even custom methods! Suppose now with the choice of the `A2C` method, you sketch the following set of possible hyperparameters defined as a Python `dict`:
    
```python
config_params = 
```

Now that we have our set of all possible initial hyperparameters, we simply pump them into either of the hyperparameter tuning packages and watch the code magically (not magically at all) optimize our selection for eventual full training. 

First we show how to do this with the simplicity of `hypertoolz` before comparing the much more tedious process of using `optuna`.

#### Sample Tuning with Hypertoolz

