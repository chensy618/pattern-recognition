"""
Problems (2a)/(2b): Principal component analysis, implemented with NumPy.

Algorithm
---------
Given the preprocessed reference data X (n × d, here d = 8):

1. Centre the data with its mean μ (stored explicitly, so projection and
   reconstruction use exactly the same centre):
       X_c = X - μ
2. Covariance matrix:
       C = X_c^T X_c / (n - 1)                      (d × d, symmetric)
3. Eigendecomposition  C v_i = λ_i v_i  with np.linalg.eigh, sorted so that
   λ_1 >= λ_2 >= ... >= λ_d. The eigenvectors v_i are orthonormal; they are
   the principal directions, and λ_i is the variance along v_i.
4. Keep the first q components  W_q = [v_1, ..., v_q]  (d × q).
5. Explained variance ratio of component i:
       EVR_i = λ_i / Σ_j λ_j
   and the cumulative ratio Σ_{i<=q} EVR_i.
6. Projection (no whitening):
       Z = (X - μ) W_q                               (n × q)
7. Reconstruction in the same (log-transformed, standardised) space:
       X̂ = Z W_q^T + μ

The model is fitted on X_reference only; X_query is projected with the same
μ and W_q.

Sign convention: an eigenvector is only defined up to its sign (v and -v are
both eigenvectors). To make the reported coefficients reproducible, each
component is flipped so that its largest-magnitude coefficient is positive.
A flip only mirrors the corresponding axis of the projection; the
reconstructions, residuals and anomaly scores are unchanged.
"""

import numpy as np


class PCA:
    """
    Parameters
    ----------
    n_components : int
        Number of principal components q to retain.
    """

    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]

        # 1. Centre the data (mean stored for projection / reconstruction).
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_

        # 2. Sample covariance matrix.
        C = Xc.T @ Xc / (n - 1)

        # 3. Eigendecomposition of the symmetric matrix C. eigh returns the
        #    eigenvalues in ascending order, so reverse to descending.
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

        # Sign convention: largest-|coefficient| entry of every component positive.
        signs = np.sign(eigenvectors[np.argmax(np.abs(eigenvectors), axis=0),
                                     np.arange(eigenvectors.shape[1])])
        eigenvectors = eigenvectors * signs

        self.covariance_ = C
        self.eigenvalues_ = eigenvalues                    # all d eigenvalues λ_i
        self.all_components_ = eigenvectors                # columns v_1 .. v_d
        # 4. Retained components W_q (d × q), one component per column.
        self.components_ = eigenvectors[:, :self.n_components]

        # 5. Explained variance ratios.
        self.explained_variance_ratio_ = eigenvalues / eigenvalues.sum()
        self.cumulative_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)
        return self

    def transform(self, X):
        """6. Projection Z = (X - μ) W_q."""
        return (np.asarray(X, dtype=float) - self.mean_) @ self.components_

    def inverse_transform(self, Z):
        """7. Reconstruction X̂ = Z W_q^T + μ."""
        return np.asarray(Z, dtype=float) @ self.components_.T + self.mean_

    def fit_transform(self, X):
        return self.fit(X).transform(X)
