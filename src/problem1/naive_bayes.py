"""
Problem (1a): Naive Bayes classifier with Parzen-window class-conditional densities.

Model
-----
Bayes' rule:  P(C_k | x) ∝ P(C_k) p(x | C_k).
Naive assumption: features are conditionally independent given the class,

    p(x | C_k) = Π_j p(x_j | C_k).

Continuous features: Gaussian Parzen window (kernel density estimate)

    p̂(x_j | C_k) = 1 / (n_k h_j) Σ_{i: y_i = k} K( (x_j - x_ij) / h_j ),
    K(u) = exp(-u² / 2) / sqrt(2π).

Classification: compute the log-posterior score of each class

    s_k(x) = log P(C_k) + Σ_j log p̂(x_j | C_k)

and choose the class with the largest score (equivalently P(C_1 | x) >= 0.5).

Bandwidth h (hyperparameter)
----------------------------
* a number: the same h for every feature. This is reasonable because the
  features are standardised (unit variance) before they reach the classifier;
* "silverman": a separate h_jk per feature and class from Silverman's rule of
  thumb, h = 0.9 min(σ, IQR / 1.34) n_k^(-1/5).

Special cases
-------------
* Constant features (a single value in the training data) carry no
  information about the class and are ignored.
* Binary features (exactly two values in the training data, e.g. the one-hot
  production-area indicators) are not continuous, so a kernel density makes
  little sense. Their class-conditional probability is estimated with
  Laplace-smoothed relative frequencies,
      P(x_j = v | C_k) = (count_k(v) + α) / (n_k + 2α),
  which is never zero, even for a value that never occurs in class k.
* Very small densities: all products are computed as sums of logarithms and
  kernel sums with logsumexp, so densities far below 1e-300 do not underflow
  to zero (log 0 = -inf).
"""

import numpy as np
from scipy.special import logsumexp

LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)


class ParzenNaiveBayes:
    """
    Naive Bayes with Gaussian Parzen windows (continuous features) and
    Laplace-smoothed frequencies (binary features).

    Parameters
    ----------
    bandwidth : float or "silverman"
        Kernel width h in standardised units, or Silverman's rule per feature.
    min_bandwidth : float
        Lower bound on every bandwidth (avoids h = 0 for nearly constant data).
    laplace_alpha : float
        Laplace smoothing constant α for binary features.
    """

    def __init__(self, bandwidth=0.5, min_bandwidth=1e-2, laplace_alpha=1.0):
        self.bandwidth = bandwidth
        self.min_bandwidth = min_bandwidth
        self.laplace_alpha = laplace_alpha

    # ------------------------------------------------------------------
    @staticmethod
    def _silverman(x):
        """Silverman's rule-of-thumb bandwidth for a 1-D sample x."""
        sigma = np.std(x)
        q75, q25 = np.percentile(x, [75, 25])
        spread = min(sigma, (q75 - q25) / 1.34)
        if spread <= 0:              # IQR can be 0 for skewed count data
            spread = sigma
        return 0.9 * spread * len(x) ** (-0.2)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        self.classes_ = np.array([0, 1])
        n_features = X.shape[1]

        # Class priors P(C_k): relative class frequencies in the training set.
        self.log_priors_ = np.log(np.array([np.mean(y == k) for k in self.classes_]))

        # Determine the type of every feature from the training data.
        n_unique = np.array([len(np.unique(X[:, j])) for j in range(n_features)])
        self.constant_ = np.where(n_unique == 1)[0]
        self.binary_ = np.where(n_unique == 2)[0]
        self.continuous_ = np.where(n_unique > 2)[0]

        # Binary features: the two possible values and their smoothed
        # log-probabilities per class, shape (2 classes, n_binary, 2 values).
        self.binary_values_ = np.array([np.unique(X[:, j]) for j in self.binary_])
        a = self.laplace_alpha
        self.binary_log_prob_ = np.zeros((2, len(self.binary_), 2))
        for k in self.classes_:
            Xk = X[y == k]
            for b, j in enumerate(self.binary_):
                counts = np.array([np.sum(Xk[:, j] == v) for v in self.binary_values_[b]])
                self.binary_log_prob_[k, b] = np.log((counts + a) / (len(Xk) + 2 * a))

        # Continuous features: a Parzen estimator is "lazy" - we store the
        # training samples of each class and one bandwidth per feature.
        self.samples_, self.bandwidths_ = [], []
        for k in self.classes_:
            Xk = X[y == k][:, self.continuous_]
            if self.bandwidth == "silverman":
                h = np.array([self._silverman(Xk[:, c]) for c in range(Xk.shape[1])])
            else:
                h = np.full(Xk.shape[1], float(self.bandwidth))
            self.samples_.append(Xk)
            self.bandwidths_.append(np.maximum(h, self.min_bandwidth))
        return self

    # ------------------------------------------------------------------
    def _log_parzen(self, X, k):
        """
        Σ_j log p̂(x_j | C_k) over the continuous features, for every row of X.

        Exact speed-up: most features take only a few distinct values, so the
        kernel sum is evaluated over the distinct training values u weighted by
        their counts n_u,  Σ_i K((x - x_i)/h) = Σ_u n_u K((x - u)/h),  and only
        once per distinct test value. This gives identical results and makes
        the bagged version (bootstrap samples contain duplicates) much faster.
        """
        Xk, h = self.samples_[k], self.bandwidths_[k]
        n_k = Xk.shape[0]
        out = np.zeros(X.shape[0])
        for c, j in enumerate(self.continuous_):
            train_vals, counts = np.unique(Xk[:, c], return_counts=True)
            test_vals, inverse = np.unique(X[:, j], return_inverse=True)
            u = (test_vals[:, None] - train_vals[None, :]) / h[c]
            log_k = -0.5 * u ** 2 - LOG_SQRT_2PI + np.log(counts)[None, :]
            # log[ 1/(n_k h) Σ_u n_u K(u) ]  (logsumexp avoids underflow)
            out += (logsumexp(log_k, axis=1) - np.log(n_k * h[c]))[inverse]
        return out

    def _log_binary(self, X, k):
        """Σ_j log P(x_j | C_k) over the binary features."""
        out = np.zeros(X.shape[0])
        for b, j in enumerate(self.binary_):
            # Index of the closest of the two training values (exact match in
            # practice, since validation/test data are transformed identically).
            v = self.binary_values_[b]
            idx = np.argmin(np.abs(X[:, j][:, None] - v[None, :]), axis=1)
            out += self.binary_log_prob_[k, b, idx]
        return out

    def log_posterior_scores(self, X):
        """Unnormalised log posteriors s_k(x), shape (n, 2)."""
        X = np.asarray(X, dtype=float)
        return np.column_stack([
            self.log_priors_[k] + self._log_parzen(X, k) + self._log_binary(X, k)
            for k in self.classes_])

    def predict_proba(self, X):
        """Posterior probabilities P(C_k | x), shape (n, 2)."""
        s = self.log_posterior_scores(X)
        return np.exp(s - logsumexp(s, axis=1, keepdims=True))

    def predict(self, X):
        """Class with the largest log-posterior score."""
        return np.argmax(self.log_posterior_scores(X), axis=1)
