# Enron Spam Classifier

Logistic regression and multinomial Naive Bayes implemented from scratch in
NumPy, trained on the Enron spam corpus (~33k emails). On a held-out 20% split,
Naive Bayes reaches **95.5% accuracy / 93.9% precision / 97.6% recall**, and
L2-regularized logistic regression is close behind at 94.6% accuracy.

Only the bag-of-words step uses scikit-learn's `CountVectorizer`; the training
loops, loss, gradients, regularization, smoothing, and evaluation metrics are
written directly on NumPy arrays.

## Quick start

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

```bash
uv run python main.py
```

Expects `enron_spam_data.csv` in the working directory. A full run takes about
15–20 seconds.

## Pipeline

1. Load the CSV and drop rows with an empty `Message`.
2. Vectorize each message into a count vector over the 1,000 most frequent
   tokens.
3. Label spam as 1, ham as 0.
4. Assign each row a random fold id in $\{0,\dots,4\}$ and hold out one fold
   (~80/20 split).
5. Prepend a bias column so the intercept is learned as $w_0$.
6. Train and evaluate both models on the same split.

## Logistic regression

Predicts $\hat{y} = \sigma(w^\top x)$ and minimizes mean binary cross-entropy
with full-batch gradient descent:

$$
\nabla_w J = \frac{1}{n} X^\top (\hat{y} - y)
$$

Predictions are clipped to $[10^{-10},\ 1 - 10^{-10}]$ before the log so a
saturated sigmoid can't produce `nan`. Weights start at zero (safe because the
loss is convex) and training stops when the loss improves by less than
$10^{-4}$ between iterations, or at 3,000 iterations.

Regularization is switchable and excludes the bias weight:

| Mode | Penalty | Gradient contribution |
|------|---------|-----------------------|
| `l2` | $\frac{\lambda}{2} \lVert w_{1:} \rVert_2^2$ | $\lambda\, w_j$ |
| `l1` | $\lambda \lVert w_{1:} \rVert_1$ | $\lambda\, \operatorname{sign}(w_j)$ |

## Naive Bayes

Class priors are empirical frequencies; token likelihoods use Laplace
smoothing so unseen tokens never zero out a class:

$$
P(w \mid y) = \frac{\text{count}(w, y) + 1}{\sum_{w'} \text{count}(w', y) + |V|}
$$

Scoring happens in log space to avoid underflow, and is a single
`X @ log_likelihood` product per class followed by an `argmax`.

## Results

$d = 1000$, $\eta = 0.01$, $\lambda = 0.01$. Numbers vary slightly between
runs since the fold shuffle is unseeded.

| Model | Iterations | Train loss | Accuracy | Precision | Recall |
|-------|-----------:|-----------:|---------:|----------:|-------:|
| Logistic regression (L1) | 457 | 0.371 | 0.911 | 0.879 | 0.955 |
| Logistic regression (L2) | 683 | 0.278 | 0.944 | 0.922 | 0.971 |
| Naive Bayes | — | — | 0.953 | 0.933 | 0.976 |


- **Naive Bayes wins on every metric** with zero hyperparameters and a single
  pass over the data — the usual outcome for bag-of-words spam detection.
- **L1 stops earlier at a higher loss.** The kink at $w_j = 0$ makes weights
  oscillate under a fixed step size, tripping the loss-delta stop early.
- **Recall exceeds precision** for every model; raising the logistic
  regression threshold above 0.5 would trade some recall for precision.

## Repository layout

```
.
├── main.py                        # Full pipeline: loading, both models, evaluation
├── enron_spam_data.csv            # Enron spam/ham corpus (~52 MB)
└── pyproject.toml                 # uv project definition
```
