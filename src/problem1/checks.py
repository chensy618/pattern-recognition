"""
Sanity checks of the implementations on small synthetic data, run before the
real experiments:  python checks.py

1. Group split: no site in two sets, no rows lost.
2. Preprocessing: statistics come from the training rows only.
3. Parzen density: each 1-D class-conditional density integrates to 1, and the
   classifier separates two well-separated Gaussian classes.
4. Least squares: matches the closed-form solution w = (X^T X)^(-1) X^T y.
5. MLP: backpropagation gradients agree with finite differences, and the
   network learns a non-linear (XOR-like) problem.
6. Metrics: precision / recall / F1 / accuracy on a hand-computed example.
"""

import numpy as np

from data_utils import load_data, group_train_val_test_split, Preprocessor
from naive_bayes import ParzenNaiveBayes, LOG_SQRT_2PI
from least_squares import LeastSquaresClassifier
from mlp import MLPClassifier
from metrics import precision_recall_f1, accuracy
from scipy.special import logsumexp

rng = np.random.default_rng(0)


def check_split():
    data = load_data()
    tr, va, te = group_train_val_test_split(data["groups"])   # asserts internally
    print(f"[ok] split: {len(tr)}/{len(va)}/{len(te)} rows, sites disjoint")


def check_preprocessing():
    X = rng.normal(5.0, 2.0, size=(100, 3))
    X[::7, 1] = np.nan
    tr, te = np.arange(70), np.arange(70, 100)
    prep = Preprocessor().fit(X[tr])
    Z = prep.transform(X[tr])
    assert not np.isnan(prep.transform(X)).any()
    assert np.allclose(Z.mean(axis=0), 0) and np.allclose(Z.std(axis=0), 1)
    assert np.allclose(prep.medians_, np.nanmedian(X[tr], axis=0))
    print("[ok] preprocessing: NaNs filled, train data has mean 0 / std 1")


def check_parzen():
    # Two Gaussian classes in 2-D, means (-2, -2) and (2, 2).
    X = np.vstack([rng.normal(-2, 1, (200, 2)), rng.normal(2, 1, (200, 2))])
    y = np.r_[np.zeros(200), np.ones(200)].astype(int)
    for bw in (0.3, "silverman"):
        nb = ParzenNaiveBayes(bandwidth=bw).fit(X, y)
        # Numerical integral of p̂(x_0 | class 0) over a fine grid.
        grid = np.linspace(-10, 10, 4001)
        Xk, h = nb.samples_[0], nb.bandwidths_[0][0]
        u = (grid[:, None] - Xk[None, :, 0]) / h
        dens = np.exp(logsumexp(-0.5 * u ** 2 - LOG_SQRT_2PI, axis=1)) / (len(Xk) * h)
        integral = np.sum(dens) * (grid[1] - grid[0])
        assert abs(integral - 1) < 1e-3, integral
        acc = np.mean(nb.predict(X) == y)
        assert acc > 0.95 and np.allclose(nb.predict_proba(X).sum(axis=1), 1)
        print(f"[ok] parzen (h={bw}): density integrates to {integral:.4f}, accuracy {acc:.3f}")


def check_least_squares():
    X = rng.normal(size=(50, 4))
    y = (X @ np.array([1.0, -2.0, 0.5, 0.0]) + 0.3 > 0).astype(int)
    ls = LeastSquaresClassifier(reg_lambda=0.0).fit(X, y)
    Xa = np.hstack([X, np.ones((50, 1))])
    w_closed = np.linalg.inv(Xa.T @ Xa) @ Xa.T @ y
    assert np.allclose(np.r_[ls.w_, ls.b_], w_closed)
    print(f"[ok] least squares: matches (X^T X)^-1 X^T y, train accuracy "
          f"{np.mean(ls.predict(X) == y):.3f}")


def check_mlp():
    # Finite-difference gradient check on a tiny network, for both activations.
    X = rng.normal(size=(15, 4))
    y = (rng.random(15) < 0.4).astype(float)
    for act in ("relu", "tanh"):
        m = MLPClassifier(hidden_sizes=(5, 3), alpha=0.01, activation=act)
        m._init_params(4, rng)
        acts, zs = m._forward(X)
        gW, gb = m._backward(acts, zs, y)
        eps, worst = 1e-6, 0.0
        for params, grads in ((m.W_, gW), (m.b_, gb)):
            for P, G in zip(params, grads):
                for idx in np.ndindex(P.shape):
                    old = P[idx]
                    P[idx] = old + eps; lp = m.loss(X, y)
                    P[idx] = old - eps; lm = m.loss(X, y)
                    P[idx] = old
                    worst = max(worst, abs((lp - lm) / (2 * eps) - G[idx]))
        assert worst < 1e-6, worst
        print(f"[ok] mlp ({act}): max |numerical - backprop gradient| = {worst:.1e}")

    # XOR-like problem that a linear classifier cannot solve.
    X = rng.uniform(-1, 1, size=(400, 2))
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    m = MLPClassifier(hidden_sizes=(16,), learning_rate=0.05, n_epochs=200,
                      random_state=1).fit(X, y)
    acc = np.mean(m.predict(X) == y)
    assert acc > 0.9, acc
    print(f"[ok] mlp: XOR accuracy {acc:.3f}, loss {m.loss_curve_[0]:.3f} -> {m.loss_curve_[-1]:.3f}")


def check_metrics():
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 1])
    y_pred = np.array([1, 0, 1, 1, 0, 0, 0, 0])
    p, r, f = precision_recall_f1(y_true, y_pred)       # TP=2, FP=1, FN=2
    assert np.isclose(p, 2 / 3) and np.isclose(r, 0.5) and np.isclose(f, 4 / 7)
    assert np.isclose(accuracy(y_true, y_pred), 5 / 8)  # TP + TN = 2 + 3
    print("[ok] metrics: P = 2/3, R = 1/2, F1 = 4/7, accuracy = 5/8")


if __name__ == "__main__":
    check_split()
    check_preprocessing()
    check_parzen()
    check_least_squares()
    check_mlp()
    check_metrics()
    print("all checks passed")
