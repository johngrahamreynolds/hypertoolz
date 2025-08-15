# Hypertoolz

The `hypertoolz` package houses a set of simple, reusable modules for efficiently running hyperparameter tuning with Optuna on reinforcement learning optimization problems. 

Much of the code produced during hyperparameter tuning with Optuna can be redundant - this package is designed to neatly remove much of this redundancy without elimnating any core abstraction across the spectrum of RL algorithms, the space of all possible hyperparameters, etc. The code herewithin can be thought to be akin to the same simplicity provided by something like Hugging Face `transformers` in circumstantial favor of the larger, more complete, but much more verbose `torch` atop which it is built.

The goal of `hypertoolz` is efficient simplicity!

### Installation



### Introduction

Commonly, when using Optuna for hyperparameter tuning, one constructs 4 core Python entities that are largely redundant across different optimization problems. These entities are: a `sampler`, a `callback`, an `objective fn`, and an `optimization loop`. The section below reminds the reader of their traditional, verbose usage for a simple example problem.


### Full Optuna Tuning

