# Active Learning

<!--

Excalidraw:

```bash
# Log in/out to Docker Hub
docker logout
docker login

# Pull the official image (first time)
docker pull excalidraw/excalidraw

# Start app
docker run --rm -dit --name excalidraw -p 5001:80 excalidraw/excalidraw:latest
# Open browser at http://localhost:5001

# Stop
docker stop excalidraw
docker rm excalidraw
docker ps
```

Title: Does Active Learning Really Work in Deep Learning?
Subtitle: Not Always, But It Can Still Be a Better Guess than Random Sampling
-->

<p align="center">
<img src="./assets/mario-mendez-fw7sKxSg5Vs-unsplash.jpg" alt="Some blueberries hanging from a branch." width="1000"/>
<small style="color:grey">
Some blueberries hanging from a branch waiting to be picked. Which ones would you choose?
Photo by <a href="https://unsplash.com/@m_mendez_ix">Mario Mendez</a> on <a href="https://unsplash.com/photos/blue-berries-in-tilt-shift-lens-fw7sKxSg5Vs">Unsplash</a>.</small>
</p>

Deep learning models typically require large amounts of labelled data to achieve good performance. However, every machine learning practitioner knows how difficult it is to achieve that. Even before dealing with the model definition, it is necessary to address two big requirements:

1. We need to set up a pipeline to get enough amounts of good quality, representative data (randomized, non-biased, etc.). Ideally, we should be able to re-trigger that pipeline on demand; this is especially important if we suspect [data drift](https://en.wikipedia.org/wiki/Concept_drift) might occur (as it often does).
2. We have to annotate those obtained samples as efficiently as possible &mdash; and, as known, human labeling is tiresome, expensive, and error-prone.

[Active Learning](https://en.wikipedia.org/wiki/Active_learning_(machine_learning)) assumes the first step is given and it defines a framework to adress the second one by using the model outputs themselves.

> The main idea behind Active Learning consists in labeling only the most informative samples iteratively, as discovered by the model. The underlying assumption is that this progressive training and labeling boosts the model's performance faster than random sampling, which is the most common baseline for comparison &mdash; but, is that always the case?

In this blog post, you will learn:

- Which some of the most popular Active Learning strategies are and how they work behind the scenes.
- How those techniques can be used via the library [`scikit-activeml`](https://github.com/scikit-activeml/scikit-activeml) without any further implementation.
- Some experiments with the [flowers dataset](https://www.kaggle.com/datasets/imsparsh/flowers-dataset) which evaluate the introduced methods. TLDR; random sampling performs surprisingly well compared to the other methods.

<div style="height: 20px;"></div>
<div align="center" style="border: 1px solid #e4f312ff; background-color: #fcd361b9; padding: 1em; border-radius: 6px;">
<strong>
You can find this post's accompanying code in <a href="https://github.com/mxagar/active_learning_guide">this GitHub repository</a>.
</strong>
</div>
<div style="height: 30px;"></div>

## Active Learning Strategies in a Nutshell

There are a plethora of methods under the umbrella of Active Learning (AL); in fact, AL is still a research topic, and new methods are constantly being proposed. The following papers classify and describe the most important ones so far:

- [Active Learning Literature Survey (Settles, 2010)](https://burrsettles.com/pub/settles.activelearning.pdf)
- [A Survey of Deep Active Learning (Ren at al., 2020)](https://arxiv.org/abs/2009.00236)

I won't go into all the details here; instead, I prefer to focus on some methods I have worked with and which are available off-the-shelf in the library [`scikit-activeml`](https://github.com/scikit-activeml/scikit-activeml):

- Random Sampling
- Entropy-based Uncertainty Sampling
- Margin Sampling
- Least Confident Sampling
- BADGE (Batch Active learning by Diverse Gradient Embeddings)

<div style="height: 20px;"></div>
<p align="center">── ◆ ──</p>
<div style="height: 20px;"></div>

In general, AL methods try to iteratively identify the most informative samples from an unlabeled pool of data $U$.
We start by labeling a small subset of samples, which we call $L$, and we train the model with them; that's our first iteration.
In each subsequent iteration, the model trained on the already labeled set $L$ runs inference on the samples in $U$ and we choose the ones expected to improve the model the most. Then, those are annotated and added to $L$, and the model is re-trained. This process continues until we reach a certain stopping criterion, e.g., a maximum number of iterations, a performance threshold, or when the model's performance plateaus.

<p align="center">
<img src="./assets/al_concept.png" alt="Active Learning Concept." width="1000"/>
<small style="color:grey">Active Learning iteratively selects samples to be annotated from an unlabeled pool <i>U</i>. Then, these samples are labeled and added to the training set <i>L</i>. The right figure shows the process in action in a 2D embedding space. If you're not sure what embeddings are, check <a href="https://mikelsagardia.io/blog/diffusion-for-developers.html">this post of mine</a>. Image by the author.</small>
</p>

The selection strategy can depend on several factors, but commonly the model outputs are part of the process. Considering a classification problem, the model typically outputs a class probability distribution $p(y|x)$ for a sample $x$, where $y \in \{1, ..., K\}$ is the predicted class. 

**Random Sampling**

**Entropy-based Uncertainty Sampling**

**Margin Sampling**

**Least Confident Sampling**

**[BADGE (Batch Active learning by Diverse Gradient Embeddings, by Ash et al., 2020)](https://arxiv.org/abs/1906.03671)**

<div style="height: 20px;"></div>
<p align="center">── ◆ ──</p>
<div style="height: 20px;"></div>

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

## Implementation with Scikit ActiveML

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

- `X` = feature matrix `(n_samples, n_features)`
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

## Experiments


<p align="center">
<img src="./assets/performance_benchmark.png" alt="Active Learning Methods Comparison." width="1000"/>
<small style="color:grey">
Performance comparison of different active learning methods: random sampling, maximum entropy sampling, least confident sampling, margin sampling, and BADGE. The x-axis shows the number of labeled samples, and the y-axis shows the model's accuracy on a test set. In this case, random sampling performs surprisingly well, while the other methods do not show significant improvements over random sampling.
</small>
</p>


## Conclusions


