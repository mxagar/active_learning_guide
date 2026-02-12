# Active Learning Guide

This is my small guide (& compilation of examples) for [active learning](https://en.wikipedia.org/wiki/Active_learning).

Libraries used:

- [`scikit-activeml`](https://github.com/scikit-activeml/scikit-activeml)
- [BAAL: Bayesian active learning library](https://github.com/baal-org/baal)

Datasets used:

- []()

Table of contents:

- [Active Learning Guide](#active-learning-guide)
  - [Setup](#setup)
  - [How to Use this Guide](#how-to-use-this-guide)
  - [Basic Concepts](#basic-concepts)
  - [Scikit Activeml](#scikit-activeml)
  - [BAAL](#baal)
  - [Authorship](#authorship)


## Setup

You need to setup a Python environment with the dependencies specified in `conda.yaml` and `requirements.in`. You can do this with the following commands:

```bash
# Create the necessary Python environment
conda env create -f conda.yaml
conda activate activeml

# Compile and install all dependencies
pip-compile requirements.in
pip-sync requirements.txt

# If we need a new dependency,
# add it to requirements.in 
# And then:
pip-compile requirements.in
pip-sync requirements.txt
```

## How to Use this Guide



## Basic Concepts


## Scikit Activeml


## BAAL


## Authorship

Mikel Sagardia, 2026.  
No guarantees.

