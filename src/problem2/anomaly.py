"""
Problems (2c)/(2d): PCA reconstruction-based anomaly score and detection.

Anomaly score
-------------
1. Reconstruct every record from its first q principal components:
       X̂ = Z W_q^T + μ
2. Reconstruction residual (in the log-transformed, standardised space):
       R = X - X̂                                   (n × 8)
3. Standardise each feature's residual with the mean and standard deviation
   of the *reference* residuals of that feature:
       r̃_ij = (r_ij - μ_r,j) / σ_r,j
   so that residuals of features with different natural variability become
   comparable (a feature that PCA reconstructs badly even for normal data does
   not dominate the score).
4. Per-record score: sum of squared standardised residuals
       S(x_i) = Σ_j r̃_ij²
   S measures how far a record lies from the 2-D subspace that describes the
   normal (reference) data, i.e. how strongly it violates the correlation
   structure learned from the reference set.

Numerical protection: if a reference residual column has (almost) zero
standard deviation, it cannot be standardised; such columns are excluded from
the score (their indices are stored in `excluded_`).

Detection
---------
Threshold: τ = 95th percentile of the reference scores,
    np.quantile(reference_scores, 0.95)  (default linear interpolation).
A record is flagged if S(x) > τ (strictly greater; S = τ is not flagged).
By construction about 5% of the reference records exceed τ.
"""

import numpy as np


class PCAAnomalyScorer:
    """
    Parameters
    ----------
    pca : fitted PCA
        PCA model fitted on the preprocessed reference data.
    quantile : float
        Quantile of the reference scores used as detection threshold.
    eps : float
        Residual standard deviations below eps are treated as zero.
    """

    def __init__(self, pca, quantile=0.95, eps=1e-12):
        self.pca = pca
        self.quantile = quantile
        self.eps = eps

    def residuals(self, X):
        """R = X - X̂ with X̂ the reconstruction from the retained components."""
        X = np.asarray(X, dtype=float)
        X_hat = self.pca.inverse_transform(self.pca.transform(X))
        return X - X_hat

    def fit(self, X_reference):
        """Estimate the residual statistics and the threshold from the reference set."""
        R_ref = self.residuals(X_reference)
        self.res_mean_ = R_ref.mean(axis=0)
        self.res_std_ = R_ref.std(axis=0, ddof=0)
        # Columns whose residual cannot be standardised (zero spread).
        self.excluded_ = np.where(self.res_std_ < self.eps)[0]
        self.used_ = np.where(self.res_std_ >= self.eps)[0]

        self.reference_scores_ = self.score(X_reference)
        self.threshold_ = float(np.quantile(self.reference_scores_, self.quantile))
        return self

    def standardised_residuals(self, X):
        """r̃_ij = (r_ij - μ_r,j) / σ_r,j for the usable columns (others set to 0)."""
        R = self.residuals(X)
        R_std = np.zeros_like(R)
        R_std[:, self.used_] = (R[:, self.used_] - self.res_mean_[self.used_]) / self.res_std_[self.used_]
        return R_std

    def score(self, X):
        """Anomaly score S(x_i) = Σ_j r̃_ij²."""
        return np.sum(self.standardised_residuals(X) ** 2, axis=1)

    def predict(self, X):
        """1 = anomaly (S > τ, strictly), 0 = normal."""
        return (self.score(X) > self.threshold_).astype(int)
