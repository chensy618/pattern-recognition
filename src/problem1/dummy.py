"""
Problem (1d): Dummy baseline classifier.

Stratified random classifier
----------------------------
The dummy ignores the features completely and predicts at random

    ŷ ~ Bernoulli(p̂_train),

where p̂_train is the fraction of breachers (y = 1) in the training set, so it
flags sites at the same rate as breaches occur in the training data. This is
the "random inspection" policy the directorate would follow without a model.

Expected performance on the breach class: the predictions are independent of
the true labels, so
    E[precision] = π_test  (positive rate of the test set, ≈ 0.21),
    E[recall]    = p̂_train (≈ 0.18),
and F1 ≈ 0.20. A classifier that does not clearly beat these numbers has not
learned anything useful from the features.

The predictions are random, so the evaluation repeats the dummy with many
random states and reports mean and standard deviation (see experiment.evaluate_dummy).
"""

import numpy as np


class DummyClassifier:
    """
    Parameters
    ----------
    random_state : int
        Random state (seed) of the Bernoulli draws, for reproducibility.
    """

    def __init__(self, random_state=0):
        self.random_state = random_state

    def fit(self, X, y):
        # The only quantity learned is the positive rate of the training set.
        self.p_train_ = float(np.mean(np.asarray(y) == 1))
        return self

    def predict_proba(self, X):
        """Feature-blind probability P(y=1) = p̂_train for every row, shape (n, 2)."""
        n = np.asarray(X).shape[0]
        return np.column_stack([np.full(n, 1 - self.p_train_), np.full(n, self.p_train_)])

    def predict(self, X):
        """Random predictions ŷ ~ Bernoulli(p̂_train)."""
        n = np.asarray(X).shape[0]
        rng = np.random.default_rng(self.random_state)
        return (rng.random(n) < self.p_train_).astype(int)
