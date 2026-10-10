"""
Data loading and preprocessing for Problem 2 (water-quality reports).

Preprocessing (Appendix B)
--------------------------
1. Log transform every feature except pH:

       x'_j = x_j               if j = pH
       x'_j = log(1 + x_j)      otherwise      (np.log1p)

   The concentrations are strongly right-skewed in their reported units; the
   log transform makes them more symmetric and reduces the influence of a few
   very large values.

2. Standardise with the mean and standard deviation of the *reference* set
   (ddof = 0):

       z_j = (x'_j - μ_j) / σ_j

3. Apply the same transform and the same μ_j, σ_j to the query set.

All preprocessing parameters are estimated on X_reference only; X_query is
never used to fit anything.
"""

import os
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "data", "task2_water_reports.npz"))
RESULTS_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "results", "problem2"))


def load_data(path=DATA_PATH):
    """Load the archive and return it as a plain dictionary of NumPy arrays."""
    data = np.load(path)
    out = {
        "X_reference": data["X_reference"],                          # (915, 8) normal operation
        "X_query": data["X_query"],                                  # (246, 8) records to inspect
        "feature_names": list(data["feature_names_reference"]),      # column names
        "record_ids_query": data["record_ids_query"],                # ids, for reporting
    }
    # Both sets must have the same columns in the same order.
    assert list(data["feature_names_query"]) == out["feature_names"], "column mismatch"
    return out


class Preprocessor:
    """
    log(1 + x) transform (all features except pH) followed by z-scoring with
    the reference-set mean and standard deviation (ddof = 0).
    """

    def __init__(self, feature_names, no_log=("ph",)):
        self.feature_names = list(feature_names)
        # Boolean mask of the columns that are log-transformed.
        self.log_mask_ = np.array([name not in no_log for name in self.feature_names])

    def _log_transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        X[:, self.log_mask_] = np.log1p(X[:, self.log_mask_])
        return X

    def fit(self, X_reference):
        X_log = self._log_transform(X_reference)
        self.mean_ = X_log.mean(axis=0)
        self.std_ = X_log.std(axis=0, ddof=0)
        return self

    def transform(self, X):
        return (self._log_transform(X) - self.mean_) / self.std_

    def fit_transform(self, X_reference):
        return self.fit(X_reference).transform(X_reference)
