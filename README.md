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

- [`active_learning.ipynb`](./active_learning.ipynb): main notebook with the active learning loop and experiments.
- [`model_utils.py`](./model_utils.py): model definition and training/evaluation functions.
- [`data_utils.py`](./data_utils.py): data processing functions, e.g., loading the dataset and creating the initial labeled/unlabeled splits.
- [`active_ml_utils.py`](./active_ml_utils.py): active learning functions, e.g., computing the next candidates to label.

## Active Learning in a Nutshell

Uncertainty sampling:
[Active Learning Literature Survey (Settles, 2010)](https://burrsettles.com/pub/settles.activelearning.pdf)

Uncertainty sampling with diversity consideration (BADGE algorithm).
Diversity-based methods help when the pool embedding space contains a meaningful structure, e.g., when the data is well clustered.
[BADGE: Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds (Ash et al., 2020)](https://arxiv.org/abs/1906.03671)

[A Survey of Deep Active Learning (Ren at al., 2020)](https://arxiv.org/abs/2009.00236)
Many early AL successes were shown in small-scale or classical ML settings. In deep learning, especially with modern architectures, random sampling is often surprisingly competitive.

In fact, in my experience, I have found that ... random sampling work really well.

Literature says it doesn't always work, but it can be a good starting point for many problems:

- Task too easy to learn (e.g., very distinct class images): If the model can quickly learn the task with a small number of labeled samples, then active learning might not provide significant benefits over random sampling, because almost any sample provides useful information to the model.
- The model's uncertainty estimates might not be reliable, especially in the early stages of training when the model is not well-calibrated.
- The data distribution might be such that the most uncertain samples are not actually the most informative ones for improving the model's performance.
- The model might be strong and high-capacity, learning the task well even with random sampling, thus reducing the potential benefits of active learning.
- On simple tasks, AL methods converge to random sampling, as the model quickly learns the task and all samples become equally informative.
- High data redundancy: dataset is already well clustered, random sampling already spreads across clusters.



## Scikit ActiveML

[`scikit-activeml`](https://github.com/scikit-activeml/scikit-activeml) requires us to define a query strategy, which is a class that implements the logic for selecting the most informative samples to label. Then, we pass the dataset and the classifier to the query:

```python
# Instantiate a query strategy, e.g., an entropy-based uncertainty sampling in this case.
# This computes the entropy (E=-sum(plog(p))) from the class probabilities
# and selects the samples with the highest entropy,
# i.e., the most uncertain predictions.
query_strategy = UncertaintySampling(
    method="entropy",
    ...
)

# Query the strategy for the next samples to label
query_idx = query_strategy.query(X, y, clf=my_classifier)
```

Here, the inputs & outputs are:

- `X` = feature matrix (n_samples, n_features)
- `y` = labels with -1 for unlabeled
- `clf` = some classifier model that has the method `predict_proba(X)`
- `query_idx` = indices of the samples to label next

Therefore, we need to adapt our setup to fit this API.

Note that there are other query strategies, too:

- Random: `RandomSampling(...)`
- Margin Sampling: `UncertaintySampling(method="margin_sampling", ...)`
- Least Confidence: `UncertaintySampling(method="least_confidence", ...)`
- BADGE: `Badge(...)`
- And many more!

## Authorship

Mikel Sagardia, 2026.  
No guarantees.

