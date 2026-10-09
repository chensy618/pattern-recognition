"""
Problem (1c): Multilayer perceptron (MLP), implemented from scratch in NumPy.

Architecture (default)
----------------------
    d -> 32 -> 16 -> 1,    d = 31 (X_full) or 28 (X_registry)

    z^(l) = a^(l-1) W^(l) + b^(l),   a^(l) = ReLU(z^(l)) = max(0, z^(l))   (hidden layers)
    ŷ = σ(z^(L+1)) = 1 / (1 + exp(-z^(L+1)))                                (output)

Rows are samples, so a^(l) has shape (n, units). ŷ is interpreted as
P(breach | x).

Loss: binary cross-entropy plus an L2 penalty on the weights,

    E = -(1/N) Σ_i [ y_i log ŷ_i + (1 - y_i) log(1 - ŷ_i) ] + (α/2) Σ_l ||W^(l)||².

Backpropagation
---------------
For a sigmoid output with cross-entropy the output error is δ^(L+1) = ŷ - y.
It is propagated backwards with

    δ^(l) = (δ^(l+1) W^(l+1)^T) ⊙ ReLU'(z^(l)),   ReLU'(z) = 1[z > 0],
    ∂E/∂W^(l) = a^(l-1)^T δ^(l) / N + α W^(l),   ∂E/∂b^(l) = mean_i δ^(l).

Gradient update: mini-batch gradient descent with momentum,

    Δθ(t) = μ Δθ(t-1) - γ ∂E/∂θ,     θ(t+1) = θ(t) + Δθ(t).

Initialisation: He initialisation for ReLU (W ~ N(0, 2 / fan_in)) or Xavier
uniform for tanh; biases start at zero. Random initial weights break the
symmetry between hidden units.

Hyperparameters: hidden layers, learning rate γ, number of epochs, batch size
(tuned); momentum μ, L2 strength α and activation (fixed by default).
"""

import numpy as np


def _sigmoid(z):
    """Numerically stable logistic sigmoid."""
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


class MLPClassifier:
    """
    Binary MLP classifier trained with backpropagation and momentum.

    Parameters
    ----------
    hidden_sizes : tuple of int
        Number of units in each hidden layer, e.g. (32, 16).
    learning_rate : float
        Step size γ.
    n_epochs : int
        Number of passes over the training data.
    batch_size : int
        Mini-batch size.
    momentum : float
        Momentum coefficient μ (0 = plain gradient descent).
    alpha : float
        L2 weight-decay strength.
    activation : {"relu", "tanh"}
        Hidden-layer activation function.
    random_state : int
        Seed for weight initialisation and mini-batch shuffling.
    """

    def __init__(self, hidden_sizes=(32, 16), learning_rate=0.01, n_epochs=100,
                 batch_size=64, momentum=0.9, alpha=1e-4, activation="relu", random_state=0):
        self.hidden_sizes = tuple(hidden_sizes)
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.momentum = momentum
        self.alpha = alpha
        self.activation = activation
        self.random_state = random_state

    # ------------------------------------------------------------------
    def _init_params(self, n_in, rng):
        """He (ReLU) or Xavier-uniform (tanh) weights, zero biases."""
        sizes = [n_in, *self.hidden_sizes, 1]
        self.W_, self.b_ = [], []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            if self.activation == "relu":
                W = rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))
            else:
                limit = np.sqrt(6.0 / (fan_in + fan_out))
                W = rng.uniform(-limit, limit, size=(fan_in, fan_out))
            self.W_.append(W)
            self.b_.append(np.zeros(fan_out))

    def _act(self, z):
        return np.maximum(z, 0.0) if self.activation == "relu" else np.tanh(z)

    def _act_grad(self, z, a):
        """f'(z); for tanh expressed through a = tanh(z)."""
        return (z > 0).astype(float) if self.activation == "relu" else 1.0 - a ** 2

    def _forward(self, X):
        """
        Forward pass. Returns the activations a^(0..L) (a^(0) = X) and the
        pre-activations z^(1..L+1), which are needed by backpropagation.
        """
        activations, pre_acts = [X], []
        a = X
        for l, (W, b) in enumerate(zip(self.W_, self.b_)):
            z = a @ W + b
            pre_acts.append(z)
            if l < len(self.W_) - 1:            # hidden layer
                a = self._act(z)
                activations.append(a)
        return activations, pre_acts

    def _backward(self, activations, pre_acts, y):
        """Gradients of the loss E w.r.t. every W^(l) and b^(l)."""
        n = y.shape[0]
        y_hat = _sigmoid(pre_acts[-1][:, 0])
        delta = (y_hat - y)[:, None]            # output error, shape (n, 1)

        grads_W = [None] * len(self.W_)
        grads_b = [None] * len(self.b_)
        for l in reversed(range(len(self.W_))):
            grads_W[l] = activations[l].T @ delta / n + self.alpha * self.W_[l]
            grads_b[l] = delta.mean(axis=0)
            if l > 0:
                # Propagate the error through W^(l) and the activation of the
                # hidden layer below (its output is activations[l]).
                delta = (delta @ self.W_[l].T) * self._act_grad(pre_acts[l - 1],
                                                                activations[l])
        return grads_W, grads_b

    def loss(self, X, y):
        """Cross-entropy + L2 penalty on a data set (used for monitoring)."""
        p = np.clip(self.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
        bce = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        return bce + 0.5 * self.alpha * sum(np.sum(W ** 2) for W in self.W_)

    # ------------------------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = X.shape[0]
        rng = np.random.default_rng(self.random_state)
        self._init_params(X.shape[1], rng)

        # Momentum state: previous update Δθ for every parameter array.
        velocity = [np.zeros_like(p) for p in self.W_ + self.b_]
        self.loss_curve_ = []
        for _ in range(self.n_epochs):
            order = rng.permutation(n)                     # reshuffle every epoch
            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]
                acts, zs = self._forward(X[idx])
                gW, gb = self._backward(acts, zs, y[idx])
                # Δθ = μ Δθ_prev - γ g ;  θ += Δθ  (in-place on W_ / b_)
                for k, (p, g) in enumerate(zip(self.W_ + self.b_, gW + gb)):
                    velocity[k] = self.momentum * velocity[k] - self.learning_rate * g
                    p += velocity[k]
            self.loss_curve_.append(self.loss(X, y))
        return self

    def predict_proba(self, X):
        """[P(y=0 | x), P(y=1 | x)], shape (n, 2)."""
        _, zs = self._forward(np.asarray(X, dtype=float))
        p1 = _sigmoid(zs[-1][:, 0])
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
