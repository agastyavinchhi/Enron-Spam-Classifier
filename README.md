# Spam Classification from Scratch: Logistic Regression & Naive Bayes in NumPy

Two text classifiers — **logistic regression** and **Naive Bayes** — built from
first principles on the Enron spam corpus (~33k emails), without using
high-level libraries like scikit-learn to implement them.

The point of this project is to do everything by hand. The loss function, the
gradient, the weight updates, the regularization penalties, the convergence
check, the class priors and smoothed likelihoods, the log-space scoring, and
the evaluation metrics are all written directly on NumPy arrays. No
`.fit()`, no `.predict()`, no black boxes — every line of model logic is
visible and derived from the math it implements.

- **Logistic regression** trained with batch gradient descent on the
  binary cross-entropy loss, with switchable L1 / L2 regularization and a
  convergence-based stopping rule.
- **Multinomial Naive Bayes** with Laplace smoothing, evaluated entirely in
  log space.
- **Evaluation** — accuracy, precision, and recall computed from the confusion
  counts.

## Contents

- [Quick start](#quick-start)
- [Pipeline](#pipeline)
- [Logistic regression](#logistic-regression)
- [Naive Bayes](#naive-bayes)
- [Evaluation](#evaluation)
- [Results](#results)
- [Design notes](#design-notes)
- [Repository layout](#repository-layout)

## Quick start

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

```bash
uv run python main.py
```

The script expects `enron_spam_data.csv` in the working directory. A full run
(both regularization modes for logistic regression, plus Naive Bayes) takes
roughly 15–20 seconds on a laptop.

## Pipeline

1. **Load** the CSV and drop rows whose `Message` field is empty.
2. **Vectorize** each message into a bag-of-words count vector over the
   1,000 most frequent tokens, giving a feature matrix
   $X \in \mathbb{N}^{n \times d}$ with $d = 1000$.
3. **Label** $y_i = 1$ for spam, $0$ for ham.
4. **Split** by assigning every row a random fold id in $\{0, \dots, 4\}$ and
   holding out one fold, yielding an ~80/20 train/test split. The same fold
   machinery generalizes to $k$-fold cross-validation if desired.
5. **Prepend a bias column** of ones so the intercept is learned as
   $w_0$ rather than tracked separately.
6. **Train and evaluate** both models on the same split.

## Logistic regression

### Model

For a feature vector $x \in \mathbb{R}^{d+1}$ (with $x_0 = 1$) and weights
$w \in \mathbb{R}^{d+1}$, the model predicts

$$
\hat{y} = \sigma(w^\top x) = \frac{1}{1 + e^{-w^\top x}}
$$

which is interpreted as $P(\text{spam} \mid x)$. `predict_probability`
computes this for the whole matrix at once as `1 / (1 + np.exp(-(X @ w)))`.

### Loss

Training minimizes the mean binary cross-entropy (negative log-likelihood of a
Bernoulli model):

$$
J(w) = -\frac{1}{n} \sum_{i=1}^{n}
\Big[ y_i \log \hat{y}_i + (1 - y_i) \log (1 - \hat{y}_i) \Big]
$$

`compute_loss` clips $\hat{y}$ to $[10^{-10},\ 1 - 10^{-10}]$ before taking
logs. Without this, a confidently wrong prediction saturates the sigmoid to
exactly 0 or 1 in floating point and the loss becomes `-inf`/`nan`, which
would also break the stopping criterion below.

### Gradient

The gradient of the cross-entropy with respect to $w$ has the compact closed
form

$$
\nabla_w J = \frac{1}{n} X^\top (\hat{y} - y)
$$

which follows from $\frac{d\sigma(z)}{dz} = \sigma(z)(1 - \sigma(z))$
cancelling against the $\frac{1}{\hat{y}(1-\hat{y})}$ that comes out of the
log terms. In code this is a single matrix product:

```python
gradient = X.T @ (predict_probability(X, weights) - y) / X.shape[0]
```

### Regularization

A penalty term is added to the objective and therefore to the gradient. The
bias weight $w_0$ is excluded from the penalty in both cases — shrinking the
intercept has no regularizing benefit and just biases the decision threshold.

| Mode | Penalty added to $J$ | Gradient contribution (for $j \ge 1$) |
|------|----------------------|----------------------------------------|
| `l2` | $\frac{\lambda}{2} \lVert w_{1:} \rVert_2^2$ | $\lambda\, w_j$ |
| `l1` | $\lambda \lVert w_{1:} \rVert_1$           | $\lambda\, \operatorname{sign}(w_j)$ |

L2 shrinks all weights proportionally toward zero (ridge). L1 applies a
constant-magnitude push toward zero regardless of weight size, so small
weights get driven to (or across) zero, producing a sparser model (lasso).
The `sign` subgradient is the standard choice at $w_j = 0$, where the L1 norm
is not differentiable.

### Update rule and stopping criterion

Each iteration performs full-batch gradient descent:

$$
w \leftarrow w - \eta \left( \nabla_w J + \nabla_w R \right)
$$

Weights are initialized to zero, which is a valid starting point for logistic
regression because the loss is convex — there is no symmetry-breaking concern
as there would be in a neural network.

Training stops when the improvement in loss between consecutive iterations
drops below a tolerance:

$$
\left| J(w^{(t-1)}) - J(w^{(t)}) \right| < \varepsilon, \qquad \varepsilon = 10^{-4}
$$

or when `max_iterations` (default 3000) is reached. This is preferred over a
fixed iteration count because the appropriate number of steps depends heavily
on the learning rate and regularization strength; the loss-delta rule adapts
automatically.

All three core functions (`predict_probability`, `compute_loss`,
`gradient_descent_step`) are fully vectorized. The only Python-level loop is
over iterations.

## Naive Bayes

### Model

Multinomial Naive Bayes treats each email as a bag of tokens drawn
independently from a class-conditional categorical distribution. By Bayes'
rule and the conditional-independence assumption,

$$
P(y \mid x) \;\propto\; P(y) \prod_{w \in V} P(w \mid y)^{x_w}
$$

where $x_w$ is the count of token $w$ in the email.

### Training with Laplace smoothing

Class priors are the empirical class frequencies. Token likelihoods are
count-based with add-one (Laplace) smoothing:

$$
P(w \mid y) = \frac{\text{count}(w, y) + 1}{\sum_{w' \in V} \text{count}(w', y) + |V|}
$$

The $+1$ in the numerator and $+|V|$ in the denominator keep the distribution
normalized while guaranteeing every token has non-zero probability under every
class. Without it, a single test-set token never seen in the training spam
would make $P(\text{spam} \mid x) = 0$ outright.

`train_naive_bayes` computes both class-conditional count vectors with a
single masked sum each (`X[y == 1].sum(axis=0)`), then takes the log of the
smoothed ratio.

### Prediction in log space

Multiplying hundreds of probabilities well below 1 underflows to zero in
`float64`. Taking logs turns the product into a sum:

$$
\hat{y} = \arg\max_{y \in \{\text{ham},\, \text{spam}\}}
\left[ \log P(y) + \sum_{w \in V} x_w \log P(w \mid y) \right]
$$

The inner sum is exactly a matrix-vector product between the count matrix and
the log-likelihood vector, so `predict_naive_bayes` scores every test email
for both classes with two `X @ log_likelihood` calls and an `argmax` across
the stacked columns. There is no per-sample loop.

## Evaluation

Spam is the positive class. From the confusion counts
$\text{TP}, \text{FP}, \text{FN}$:

$$
\text{Accuracy} = \frac{\#\{\hat{y}_i = y_i\}}{n}, \qquad
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \qquad
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
$$

Precision answers "of the emails flagged as spam, how many really were?" —
low precision means real mail lands in the spam folder. Recall answers "of the
actual spam, how much did we catch?" For a spam filter precision is usually
the metric to protect, since a false positive (a lost legitimate email) costs
more than a false negative (one spam message getting through).

## Results

Single run, $d = 1000$ features, $\eta = 0.01$, $\lambda = 0.01$,
$\varepsilon = 10^{-4}$. Numbers move slightly between runs because the fold
assignment is shuffled without a fixed seed.

| Model                    | Iterations to converge | Final train loss | Accuracy | Precision | Recall |
|--------------------------|-----------------------:|-----------------:|---------:|----------:|-------:|
| Logistic regression (L1) |                    456 |           0.371  |   0.917  |    0.889  |  0.958 |
| Logistic regression (L2) |                    685 |           0.278  |   0.946  |    0.923  |  0.976 |
| Naive Bayes              |                      — |               —  |   0.955  |    0.939  |  0.976 |

Observations:

- **Naive Bayes wins on every metric** despite being the simpler model. This is
  the classic result for bag-of-words spam detection: the conditional
  independence assumption is badly violated, but the decision boundary it
  induces is still good, and NB reaches it with zero hyperparameters and a
  single pass over the data.
- **L1 stops earlier and at a higher loss than L2.** The constant-magnitude
  L1 push means the loss surface has kinks at $w_j = 0$; with a fixed step
  size, weights near zero oscillate across the kink, so the loss-delta
  criterion trips sooner and the model lands at a worse optimum. L2's smooth
  penalty lets descent keep making steady progress.
- **Recall > precision for all models.** The classifiers are slightly
  trigger-happy. Raising the decision threshold above 0.5 for logistic
  regression would trade recall for precision.

## Design notes

- **Unscaled count features.** Token counts are used as-is, not normalized.
  The raw counts can be large for long emails, which makes $w^\top x$ large
  and saturates the sigmoid early in training. This is why the learning rate
  is a conservative 0.01 — anything much larger overflows `np.exp`. A
  drop-in alternative is binary presence (`X > 0`), which behaves more like
  the multivariate-Bernoulli setting and lets logistic regression tolerate a
  higher $\eta$.
- **Bias handled as a feature.** Prepending a column of ones to $X$ means the
  intercept is trained by the same matrix-math update as every other weight,
  with no special-case code path. The regularization step slices `[1:]` to
  leave it unpenalized.
- **Fold-based split.** Rather than a single `permutation` + slice, rows are
  assigned fold ids via `arange(n) % k` after shuffling. Holding out fold 1
  gives the 80/20 split; iterating the held-out fold over $0..k-1$ gives
  5-fold CV with no other changes.
- **Convex objective.** Logistic regression's cross-entropy is convex in $w$,
  and both penalties are convex, so batch gradient descent from zero converges
  to the global minimum for any sufficiently small learning rate. There's no
  need for random restarts or momentum here; the interesting hyperparameters
  are $\eta$, $\lambda$, and the feature budget $d$.

## Repository layout

```
.
├── main.py                 # Full pipeline: loading, both models, evaluation
├── classcode.py            # Earlier scaffold the final implementation grew out of
├── enron_spam_data.csv     # Enron spam/ham corpus (~52 MB)
└── pyproject.toml          # uv project definition
```

## Dataset

The Enron spam dataset combines legitimate email from the Enron corpus with
spam collected from several sources. Each row has a `Subject`, `Message`,
and `Spam/Ham` label. Only the message body is used for features here; the
subject line is an easy extension.
