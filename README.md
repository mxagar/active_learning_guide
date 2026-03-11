# Active Learning Guide

This is my small guide (& evaluation) of [Active Learning](https://en.wikipedia.org/wiki/Active_learning) (AL) using [`scikit-activeml`](https://github.com/scikit-activeml/scikit-activeml).

The main idea behind AL consists in labeling only the most informative samples iteratively, as discovered by the model. The underlying assumption is that this progressive training and labeling boosts the model's performance faster than random sampling, which is the most common baseline for comparison &mdash; but, is that always the case?

Check [my related blog post](https://mikelsagardia.io/posts/) :smile:

The [Kaggle Flowers Dataset](https://www.kaggle.com/datasets/imsparsh/flowers-dataset) is used, which contains images for classification of 5 types of flowers; in the training split, the class distribution is as follows:

- daisy: 501 images
- dandelion: 646 images
- rose: 497 images
- sunflower: 495 images
- tulip: 607 images

Note that the images are resized to 64x64 pixels for faster training, which is not ideal for a real-world application but serves the purpose of this guide.

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

Then, you can start using the main notebook and scripts. In the following, I briefly explain the structure of the repository and the contents in each file:

- [`active_learning.ipynb`](./active_learning.ipynb): main notebook with the active learning loop and experiments. This is the entry-point for the user to learn about AL interactively. The notebook uses three helper modules, listed below.
- [`model_utils.py`](./model_utils.py): model definition and training/evaluation functions. It contains:
    - `TrainConfig`: datacass which defines the training hyperparameters.
    - `SimpleCNN`: simple convolutional neural network for image classification.
    - `save_model` and `load_model`: utility functions to save and load the model weights.
    - `train_one_epoch` and `train`: functions to train the model for one epoch or multiple epochs, respectively.
    - `validate` and `evaluate`: functions to evaluate the model on a validation or test set, respectively.
    - `plot_history`: function to plot the training history (loss and accuracy curves).
    - `predict` and `predict_image`: functions to run inference with the model and get the predicted class probabilities, either for a batch of samples or for a single image.
- [`data_utils.py`](./data_utils.py): data processing functions, e.g., loading the dataset and creating the initial labeled/unlabeled splits.
    - `build_paths_and_labels`: function to build the file paths and labels from the dataset directory structure.
    - `CustomDataset`: custom PyTorch dataset class that loads the images and applies transformations.
    - `train_test_val_pool_split`: function to split the dataset into training, validation, and pool sets, and to create the initial labeled set for active learning.
    - `visualize_batch`: function to visualize a batch of images with their labels.
    - `train_transform` and `eval_transform`: data augmentation and preprocessing transformations for training and evaluation, respectively.
- [`active_ml_utils.py`](./active_ml_utils.py): active learning functions, e.g., computing the next candidates to label.
    - **`TorchClassifierWrapper`: wrapper class that adapts a PyTorch model to the Scikit-ActiveML API, allowing us to use the query strategies with our model.**
    - **`compute_next_candidates`: function that computes the next samples to label based on the selected query/search strategy (`"random", "least_confident", "margin_sampling", "entropy", "badge"`).**
    - **`transfer_candidates_idx`: function to transfer the indices of the selected candidates to the main dataset.**
    - `plot_embeddings_2d`: function to plot the 2D UMAP embeddings of the samples, colored by several criteria.
    - `evaluate_active_learning`: function to benchmark different AL techniques, which runs the AL loop for each technique and collects the performance metrics for comparison.

The AL selection is implemented in the function `compute_next_candidates`, which is called in the main notebook `active_learning.ipynb` at each iteration of the AL loop. The function takes as input the current model, the pool of unlabeled samples, and the selected query strategy, and returns the indices of the samples to label next. Then, these indices are transferred to the main dataset using the function `transfer_candidates_idx`, which updates the labeled and unlabeled sets accordingly. A key component in this process is the `TorchClassifierWrapper`, which adapts a PyTorch model to be used for active learning.

## Active Learning in a Nutshell

Check [my related blog post](https://mikelsagardia.io/posts/).

## About Scikit ActiveML

The library to run AL is the commonly used [`scikit-activeml`](https://github.com/scikit-activeml/scikit-activeml), which follows the [Scikit-Learn](https://scikit-learn.org/) API conventions. It requires us to define a query strategy, which is a class that implements the logic for selecting the most informative samples to label. Then, we pass the dataset and the classifier to the query:

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

- `X` = feature matrix, usually of shape `(n_samples, n_features)`
- `y` = labels with `-1` for unlabeled samples
- `clf` = some classifier model that *has* the method `predict_proba(X)`
- `query_idx` = indices of the samples to label next

Therefore, we need to adapt our setup to fit this API.

Note that there are other query strategies, too:

- Random: `RandomSampling(...)`
- Margin Sampling: `UncertaintySampling(method="margin_sampling", ...)`
- Least Confidence: `UncertaintySampling(method="least_confidence", ...)`
- BADGE: `Badge(...)`
- And many more!

In the provided code, the adaptation of the PyTorch-based classifier is handled by `TorchClassifierWrapper`:

- During the instantiation of `TorchClassifierWrapper`, we pass a PyTorch `CustomDataset` containing references to all the unlabeled samples in the pool. In addition, we provide the trained PyTorch model.
- Internally, `TorchClassifierWrapper` creates a PyTorch `DataLoader` for the `CustomDataset`, which expects a list of sample indices to load; this list corresponds exactly to the input vector `X`, which no longer contains the sample features themselves.
- It also implements the method `predict_proba(X)`, which runs the `DataLoader` accessing to the indices passed in `X`, calls the PyTorch model, and outputs the model predictions (either the class probabilities or the embedding vectors).


## Experiments and Results





## Authorship

Mikel Sagardia, 2026.  
No guarantees.
