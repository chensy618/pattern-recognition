"""
Problem (1b): Sum of least squares (linear) classifier.

Model
-----
Linear prediction  ŷ = X w + b,  with targets y_i ∈ {0, 1}.
The parameters minimise the (ridge-regularised) sum of squared errors

    J(w, b) = Σ_i (y_i - w^T x_i - b)^2 + λ ||w||^2 .

Writing X̃ = [X, 1] and w̃ = [w, b], setting the gradient to zero gives the
normal equations

    (X̃^T X̃ + λ I') w̃ = X̃^T y,

where I' is the identity with a zero in the bias position (the bias is not
penalised). For λ = 0 this is the classic w̃ = (X̃^T X̃)^(-1) X̃^T y. A small
λ > 0 keeps the system well conditioned: X̃^T X̃ is singular here because the
13 one-hot production-area columns always sum to one and are therefore
collinear with the bias column.

Classification
--------------
ŷ approximates P(y = 1 | x), so the natural decision rule is ŷ >= 0.5.
(ŷ is not a proper probability; it can be < 0 or > 1.)
"""

import numpy as np


class LeastSquaresClassifier:
    """
    Least-squares linear classifier with ridge penalty.

    Parameters
    ----------
    reg_lambda : float
        Ridge regularisation strength λ (hyperparameter).
    threshold : float
        Decision threshold on the linear output (0.5 for 0/1 targets).
    """

    def __init__(self, reg_lambda=1.0, threshold=0.5):
        self.reg_lambda = reg_lambda
        self.threshold = threshold

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        t = np.asarray(y, dtype=float)                       # targets 0 / 1
        Xa = np.hstack([X, np.ones((X.shape[0], 1))])        # append bias column

        if self.reg_lambda > 0:
            # Regularisation on every weight except the bias (last entry).
            reg = self.reg_lambda * np.eye(Xa.shape[1])
            reg[-1, -1] = 0.0
            w_full = np.linalg.solve(Xa.T @ Xa + reg, Xa.T @ t)
        else:
            # λ = 0: X^T X is singular (collinear one-hot columns), so use the
            # Moore-Penrose pseudo-inverse (minimum-norm least-squares solution).
            w_full = np.linalg.pinv(Xa) @ t
        self.w_, self.b_ = w_full[:-1], w_full[-1]
        return self

    def decision_function(self, X):
        """Linear output ŷ = X w + b."""
        return np.asarray(X, dtype=float) @ self.w_ + self.b_

    def predict(self, X):
        return (self.decision_function(X) >= self.threshold).astype(int)
