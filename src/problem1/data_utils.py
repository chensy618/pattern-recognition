"""
Data loading, group-aware splitting and preprocessing for Problem 1.

Pipeline
--------
1. Load the archive with np.load().
2. Split the *sites* (groups) into 70% training, 15% validation and 15% test
   sites. All anchor years of one site end up in the same set, so a model can
   never see a test site during training (a plain random split would leak
   information across anchor years of the same site).
3. Fill missing values with the training-set median of each column.
4. Standardise every column with the training-set mean and standard deviation.

The same split (same row indices) is used for X_registry and X_full, so the
experiments of (1e), (1f) and (1h) are directly comparable.

Important: preprocessing parameters are estimated on the training set only and
then applied unchanged to the validation and test sets. Standardising all data
before splitting would leak information from the test set into training.
"""

import os
import numpy as np

# Paths are resolved relative to this file so scripts can run from anywhere.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "data", "task1_lice_forecast.npz"))
RESULTS_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "results", "problem1"))

# Global random seed used for the split and all experiments (reproducibility).
SEED = 2026


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_data(path=DATA_PATH):
    """Load the archive and return it as a plain dictionary of NumPy arrays."""
    data = np.load(path)
    return {
        "X_registry": data["X_registry"],                    # facility + geography (28 cols)
        "X_full": data["X_full"],                            # registry + lice history (31 cols)
        "y": data["y"].astype(int),                          # breach_next label (0/1)
        "feature_names_registry": list(data["feature_names_registry"]),
        "feature_names_full": list(data["feature_names_full"]),
        "groups": data["groups"],                            # siteNumber of each row
        "anchor_year": data["anchor_year"],                  # anchor year Y of each row
    }


# ----------------------------------------------------------------------------
# Group-aware train / validation / test split
# ----------------------------------------------------------------------------
def group_train_val_test_split(groups, train_frac=0.70, val_frac=0.15, seed=SEED):
    """
    Split rows into training, validation and test sets *by site*.

    The unique site ids are shuffled; the first `train_frac` of the sites go to
    the training set, the next `val_frac` to the validation set and the rest to
    the test set. Fractions refer to sites, so the row fractions are only
    approximately equal to them (sites have different numbers of anchor years).

    Returns (train_idx, val_idx, test_idx) as arrays of row indices.
    """
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    sites = rng.permutation(np.unique(groups))
    n_train = int(round(train_frac * len(sites)))
    n_val = int(round(val_frac * len(sites)))

    train_sites = sites[:n_train]
    val_sites = sites[n_train:n_train + n_val]
    test_sites = sites[n_train + n_val:]

    train_idx = np.where(np.isin(groups, train_sites))[0]
    val_idx = np.where(np.isin(groups, val_sites))[0]
    test_idx = np.where(np.isin(groups, test_sites))[0]
    check_disjoint_sites(groups, train_idx, val_idx, test_idx)
    return train_idx, val_idx, test_idx


def check_disjoint_sites(groups, train_idx, val_idx, test_idx):
    """Verify that no site appears in more than one of the three sets."""
    s_tr, s_va, s_te = (set(groups[i]) for i in (train_idx, val_idx, test_idx))
    assert not (s_tr & s_va), "a site occurs in both training and validation set"
    assert not (s_tr & s_te), "a site occurs in both training and test set"
    assert not (s_va & s_te), "a site occurs in both validation and test set"
    assert len(train_idx) + len(val_idx) + len(test_idx) == len(groups), "rows lost"


# ----------------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------------
class Preprocessor:
    """
    Median imputation followed by z-score standardisation.

    fit(X_train) estimates, per column,
        * the median (ignoring NaNs)       -> used to fill missing values,
        * the mean and standard deviation  -> used for x' = (x - mean) / std.
    Columns with zero variance get std = 1 to avoid division by zero.

    Only productionUnitsPresent contains missing values (~11% of the rows).
    The median is used instead of the mean because this count variable is
    right-skewed, and the median is robust to large values.
    """

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.medians_ = np.nanmedian(X, axis=0)
        X_filled = self._impute(X)
        self.mean_ = X_filled.mean(axis=0)
        self.std_ = X_filled.std(axis=0)
        self.std_[self.std_ < 1e-12] = 1.0
        return self

    def _impute(self, X):
        """Replace every NaN by the training median of its column."""
        X = np.array(X, dtype=float, copy=True)
        rows, cols = np.where(np.isnan(X))
        X[rows, cols] = self.medians_[cols]
        return X

    def transform(self, X):
        return (self._impute(X) - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)
