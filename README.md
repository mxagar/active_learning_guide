# Active Learning Guide

This is my small guide (& compilation of examples) for [active learning](https://en.wikipedia.org/wiki/Active_learning):

- Library used: [`scikit-activeml`](https://github.com/scikit-activeml/scikit-activeml).
- Datasets used: [Kaggle Flowers Dataset](https://www.kaggle.com/datasets/imsparsh/flowers-dataset), classification of 5 types of flowers &mdash; training split:

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

- [`active_learning.ipynb`](./active_learning.ipynb): main notebook with the active learning loop using `scikit-activeml`.
- [`model_utils.py`](./model_utils.py): model definition and training/evaluation functions.
- [`data_utils.py`](./data_utils.py): auxiliary functions related to data processing.

## Active Learning in a Nutshell



## Scikit ActiveML

[`scikit-activeml`](https://github.com/scikit-activeml/scikit-activeml) requires us to define a query strategy, which is a class that implements the logic for selecting the most informative samples to label. Then, we pass the dataset and the classifier to the query:

```python
# Instantiate query strategy, entropy-based uncertainty sampling in this case.
# This computes the entropy from the class probabilities and selects the samples with the highest entropy
# (i.e., the most uncertain predictions)
query_strategy = UncertaintySampling(
    method="entropy",
    ...
)

# Query the strategy for the next samples to label
query_idx = query_strategy.query(X, y, clf=MyClassifier)
```

Here, the inputs are:

- `X` = feature matrix (n_samples, n_features)
- `y` = labels with -1 for unlabeled
- `clf` = something that has `predict_proba(X)`
- `query_idx` = indices of the samples to label next

Therefore, we need to adapt our setup to fit this API.

There are other query strategies, too:

- Margin Sampling: `UncertaintySampling(method="margin", ...)`
- Least Confidence: `UncertaintySampling(method="least_confidence", ...)`
- Random: `RandomSampling(...)`
- Query by Committee: `QueryByCommittee(...)`
- BADGE: `Badge(...)`
- And many more! Check the documentation for details.


## Authorship

Mikel Sagardia, 2026.  
No guarantees.

