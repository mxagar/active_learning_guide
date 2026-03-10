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

I won't try to write another survey here; instead, I prefer to focus on some practical methods I have worked with and which are available off-the-shelf in the library [`scikit-activeml`](https://github.com/scikit-activeml/scikit-activeml), which is one of the most popular ones when it comes to AL.

In general, AL methods try to iteratively identify the most informative samples from an unlabeled pool of data $U$.
We start by labeling a small subset of samples, which we call $L$, and we train the model with them; that's our first iteration.
In each subsequent iteration, the model trained on the already labeled set $L$ runs inference on the samples in $U$ and we choose the ones expected to improve the model the most. Then, those are annotated and added to $L$, and the model is re-trained. This process continues until we reach a certain stopping criterion, e.g., a maximum number of iterations, a performance threshold, or when the model's performance plateaus.

<p align="center">
<img src="./assets/al_concept.png" alt="Active Learning Concept." width="1000"/>
<small style="color:grey">Active Learning iteratively selects samples to be annotated from an unlabeled pool <i>U</i>. Then, these samples are labeled and added to the training set <i>L</i>. The right figure shows the process in action in a 2D embedding space. If you're not sure what embeddings are, check <a href="https://mikelsagardia.io/blog/diffusion-for-developers.html">this post of mine</a>. Image by the author.</small>
</p>

The selection strategy is key and it can depend on several factors, but commonly the model outputs are part of the process. Considering a classification problem, the model typically outputs class probabilities $p(y_k|x)$ for a sample $x$, where $y_k \in \{1, ..., K\}$ is one of the $K$ classes.

Now, let's have a look at the aforementioned selection methods.

**Random Sampling** &mdash; This is the simplest selection strategy: we randomly select samples from the unlabeled pool $U$ to label; in other words, no model information is used. This serves as a *baseline* for comparison with more sophisticated methods. Mathematically, we can express it as

$$x^*= \text{Uniform}(U),$$

where $x^*$ is the selected sample and $\text{Uniform}(U)$ denotes a function that randomly selects a sample from the unlabeled pool $U$.

**Entropy-based Uncertainty Sampling** &mdash; The entropy of a prediction can be used to measure how confident a model is; if there is one clearly bigger class probability $p(y_k|x)$, the model is certain of its prediction, whereas more homogeneous $p(y_k|x)$ values denote uncertainty. The goal of this selection method is to select the samples with the highest uncertainty, so that the model learns those samples to end up being more confident. The entropy $H$ of a predicted sample $x$ is defined as 

$$H(x) = -\sum_{k=1}^K p(y_k|x) \log p(y_k|x).$$

Then, the next sample with the highest entropy is selected:

$$x^* = \arg\max_{x \in U} H(x).$$

In practice, all the unlabeled samples in $U$ are sorted according to their entropy value (descending), and the first $N$ are selected, being $N$ the number of new samples we'd like to add to the labeled set $L$.
This method is also known as *maximum entropy sampling*, and it is one of the most popular ones.

**Margin Sampling** &mdash; Instead of considering the whole distribution of class probabilities, margin sampling focuses on the difference between the two most probable classes. Let $p_1$ be the highest class probability for a sample $x$, that is,

$$p_1 = \max_{k} p(y_k|x),$$

and let $p_2$ be the second highest class probability. The margin is defined as the difference between these two probabilities:

$$M(x) = p_1 - p_2.$$

Then, the next sample with the smallest margin is selected, as it indicates that the model is uncertain between the top two classes:

$$x^* = \arg\min_{x \in U} M(x).$$

The idea is very similar to entropy-based sampling.

**Least Confident Sampling** &mdash; Least confident sampling selects the samples with the *lowest maximum predicted probabilities*; that is, first we compute the *least confident score* for a sample $x$ as

$$LC(x) = 1 - \max_{k} p(y_k|x).$$

Then, the next sample with the highest least confident score is selected:

$$x^* = \arg\max_{x \in U} LC(x).$$

Again, the idea is similar to the previous two methods, but it only considers the maximum predicted probability, which can be less informative than considering the whole distribution (entropy) or the top two probabilities (margin).

**[BADGE (Batch Active learning by Diverse Gradient Embeddings, by Ash et al., 2020)](https://arxiv.org/abs/1906.03671)** &mdash; The methods presented so far try to select the most uncertain samples, but they don't consider the diversity of the selected samples. In contrast, BADGE is a method that tries to select samples that are both *uncertain and diverse*. The intuitive reason why we'd like to consider *diversity* is that it seems sensible to cover as much feature space as possible, not only the spots where the model is unsure. In order to achieve that, in addition to the predicted probabilities, BADGE requires the embedding or feature vectors of the penultimate layer of the model. I won't go into the mathematical details here, but the main idea is the following:

- A
- B
- C

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



```python
class TorchClassifierWrapper(SkactivemlClassifier):
    def __init__(
        self,
        model: torch.nn.Module,
        pool_ds: CustomDataset,
        batch_size: int = 16,
        missing_label: int = MISSING_LABEL_INT,
        classes: Optional[np.ndarray] = None,
        device: Optional[str | torch.device] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        self.model = model
        self.pool_ds = pool_ds
        self.batch_size = batch_size
        self.device = torch.device(device if device is not None else next(model.parameters()).device)
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.verbose = verbose
        self.num_classes = self.model.num_classes
        self.num_features = self.model.num_features
        if classes is not None:
            self.classes_ = classes
        else:
            self.classes_ = np.arange(self.num_classes)
        super().__init__(classes=self.classes_, missing_label=missing_label, random_state=random_state)

    def _to_indices(self, X: np.ndarray | list[int]) -> list[int]:
        idx = np.asarray(X).reshape(-1).astype(int)
        if len(idx) > 0 and (idx.min() < 0 or idx.max() >= len(self.pool_ds)):
            raise IndexError(f"Some indices in X are out of bounds for pool_ds.")
        return idx.tolist()

    def _prepare_loader(self, X: np.ndarray | list[int]) -> DataLoader:
        idx = self._to_indices(X)        
        subset = Subset(self.pool_ds, idx)
        loader = DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
        return loader

    @torch.no_grad()
    def predict_proba(
        self,
        X: np.ndarray | list[int],
        return_embeddings: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        if len(X) == 0:
            probas = np.empty((0, self.num_classes), dtype=np.float32)
            if return_embeddings:
                emb = np.empty((0, self.num_features), dtype=np.float32)
                return probas, emb
            return probas

        loader = self._prepare_loader(X)
        self.model.eval()
        probs_all, feats_all = [], []

        for batch in tqdm(loader, disable=not self.verbose, desc="predict_proba"):
            x = batch[0].to(self.device, non_blocking=True)
            output = self.model(x, return_embeddings=return_embeddings)
            logits, feats = None, None
            if return_embeddings:
                logits, feats = output  # (B, num_classes), (B, num_features)
                feats_all.append(feats.detach().cpu().numpy().astype(np.float32))
            else:
                logits = output  # (B, num_classes)

            probs = torch.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float32)
            probs_all.append(probs)

        probas = np.concatenate(probs_all, axis=0)

        if return_embeddings:
            embs = np.concatenate(feats_all, axis=0) if len(feats_all) else np.empty((0, self.num_features), np.float32)
            return probas, embs

        return probas

    @torch.no_grad()
    def compute_embeddings(self, X: np.ndarray | list[int]) -> np.ndarray:
        return self.predict_proba(X, return_embeddings=True)[1]

    # Keep sklearn-ish signature compatibility
    # SkactivemlClassifier requires fit to accept sample_weight
    def fit(self, X, y=None, sample_weight=None):
        return self
```

## Experiments


<p align="center">
<img src="./assets/performance_benchmark.png" alt="Active Learning Methods Comparison." width="1000"/>
<small style="color:grey">
Performance comparison of different active learning methods: random sampling, maximum entropy sampling, least confident sampling, margin sampling, and BADGE. The x-axis shows the number of labeled samples, and the y-axis shows the model's accuracy on a test set. In this case, random sampling performs surprisingly well, while the other methods do not show significant improvements over random sampling.
</small>
</p>

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


## Conclusions


