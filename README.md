# Active Learning Guide

This is my small guide (& compilation of examples) for [active learning](https://en.wikipedia.org/wiki/Active_learning).

Libraries used:

- [`scikit-activeml`](https://github.com/scikit-activeml/scikit-activeml)
- [BAAL: Bayesian Active Learning Library](https://github.com/baal-org/baal)

Datasets used: [Kaggle Flowers Dataset](https://www.kaggle.com/datasets/imsparsh/flowers-dataset), classification of 5 types of flowers &mdash; training split:

- daisy: 501 images
- dandelion: 646 images
- rose: 497 images
- sunflower: 495 images
- tulip: 607 images

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

Once the environment is set up, first, you need to download and extract the dataset to the folder [`./data`](./data/) under the subfolder `flowers`.

Then, you can start using the main notebooks and scripts:

- [`active_learning.ipynb`](./active_learning.ipynb): main notebook with the active learning loop, using `scikit-activeml` and `baal`.
- [`model_utils.py`](./model_utils.py): model definition and training/evaluation functions.
- [`data_utils.py`](./data_utils.py): auxiliary functions related to data processing.

## Active Learning in a Nutshell




## Scikit ActiveML

[`scikit-activeml`](https://github.com/scikit-activeml/scikit-activeml)


## BAAL: Bayesian Active Learning Library

[`baal`](https://github.com/baal-org/baal)


## Authorship

Mikel Sagardia, 2026.  
No guarantees.

