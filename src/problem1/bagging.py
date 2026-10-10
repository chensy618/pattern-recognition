"""
Problems (1g)/(1h): Bagging (bootstrap aggregating) of the Parzen naive Bayes.

Algorithm
---------
1. Draw a bootstrap sample from the training set: N rows drawn uniformly
   *with replacement* (on average ~63.2% distinct rows, some repeated).
2. Train an independent ParzenNaiveBayes on this bootstrap sample (its own
   kernel centres and class priors).
3. Repeat B = 50 times.
4. Average the predicted probabilities of the B classifiers (soft voting):
       P_bag(y=1 | x) = (1/B) Σ_b P_b(y=1 | x).
5. Final class: 1 if P_bag(y=1 | x) >= 0.5.

The bootstrap samples are drawn only from the data passed to fit(), i.e. the
training set; validation and test rows are never part of any bootstrap sample.

Bagging mainly reduces the variance of the base classifier, so the gain is
largest for unstable models (e.g. a Parzen window with a small bandwidth).
"""

import numpy as np
from naive_bayes import ParzenNaiveBayes


class BaggedParzenNaiveBayes:
    """
    Parameters
    ----------
    n_estimators : int
        Number of bootstrap samples / base classifiers B.
    bandwidth, min_bandwidth, laplace_alpha, all_parzen :
        Passed on to every ParzenNaiveBayes.
    random_state : int
        Seed for drawing the bootstrap samples.
    """

    def __init__(self, n_estimators=50, bandwidth=0.5, min_bandwidth=1e-2,
                 laplace_alpha=1.0, all_parzen=True, random_state=0):
        self.n_estimators = n_estimators
        self.bandwidth = bandwidth
        self.min_bandwidth = min_bandwidth
        self.laplace_alpha = laplace_alpha
        self.all_parzen = all_parzen
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        n = X.shape[0]
        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []
        for _ in range(self.n_estimators):
            idx = rng.integers(0, n, size=n)          # bootstrap: N draws with replacement
            nb = ParzenNaiveBayes(bandwidth=self.bandwidth,
                                  min_bandwidth=self.min_bandwidth,
                                  laplace_alpha=self.laplace_alpha,
                                  all_parzen=self.all_parzen)
            self.estimators_.append(nb.fit(X[idx], y[idx]))
        return self

    def predict_proba(self, X):
        """Average of the members' posterior probabilities, shape (n, 2)."""
        return np.mean([est.predict_proba(X) for est in self.estimators_], axis=0)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
