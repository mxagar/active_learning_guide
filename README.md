# Active Learning Guide

This is my small guide (& compilation of examples) for [active learning](https://en.wikipedia.org/wiki/Active_learning).

Libraries used:

- [`scikit-activeml`](https://github.com/scikit-activeml/scikit-activeml)
- [BAAL: Bayesian Active Learning Library](https://github.com/baal-org/baal)

Datasets used: [Kaggle Flowers Dataset](https://www.kaggle.com/datasets/imsparsh/flowers-dataset)

Table of contents:

- [Active Learning Guide](#active-learning-guide)
  - [Setup](#setup)
  - [How to Use this Guide](#how-to-use-this-guide)
  - [Active Learning in a Nutshell](#active-learning-in-a-nutshell)
  - [Scikit ActiveML](#scikit-activeml)
  - [BAAL: Bayesian Active Learning Library](#baal-bayesian-active-learning-library)
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

- [`active_learning`](./active_learning.ipynb)
- [`model.py`](./model.py)
- [`utils.py`](./utils.py)
- [`./data`](./data/)

## Active Learning in a Nutshell




## Scikit ActiveML

[`scikit-activeml`](https://github.com/scikit-activeml/scikit-activeml)


## BAAL: Bayesian Active Learning Library

[`baal`](https://github.com/baal-org/baal)


## Authorship

Mikel Sagardia, 2026.  
No guarantees.

