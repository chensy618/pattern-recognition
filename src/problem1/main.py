"""
Problem 1 entry point: trains, tunes and tests the classifiers.

    python main.py                          # all classifiers
    python main.py naive_bayes              # only the Parzen naive Bayes
    python main.py least_squares mlp        # several classifiers

Classifier names (same as the file names):
    dummy          stratified random baseline      (1d)  -> used in 1e, 1f, 1h
    naive_bayes    naive Bayes + Parzen windows    (1a)  -> used in 1e, 1f, 1h
    least_squares  least-squares classifier        (1b)  -> used in 1e, 1f, 1h
    mlp            multilayer perceptron           (1c)  -> used in 1e, 1f, 1h
    bagging        bagged naive Bayes, B = 50      (1g)  -> used in 1h only

Each selected classifier is run in every experiment it belongs to:
(1e)  registry features only (X_registry, 28 features) -> results/registry_results.csv
(1f)  all features (X_full, 31 features)               -> results/full_results.csv
(1h)  all features, incl. the bagged naive Bayes       -> results/bagging_results.csv

Running a subset of classifiers only replaces their own rows in the CSV
files; the rows of the other classifiers are kept.

All experiments use the same group-aware 70/15/15 train/validation/test
split. The split, the random seed, the hyperparameter grids, the selected
hyperparameters and the validation score of every setting are saved to
results/run_config.json.
"""

import os
import sys
import json
import numpy as np

from data_utils import load_data, group_train_val_test_split, RESULTS_DIR, SEED
from experiment import param_grid, run_experiment, save_csv, N_DUMMY_REPEATS
from naive_bayes import ParzenNaiveBayes
from least_squares import LeastSquaresClassifier
from mlp import MLPClassifier
from bagging import BaggedParzenNaiveBayes

# ----------------------------------------------------------------------------
# Hyperparameter grids (selected on the validation set)
# ----------------------------------------------------------------------------
# Naive Bayes: kernel bandwidth h (standardised units) or Silverman's rule.
NB_GRID = param_grid(bandwidth=[0.1, 0.25, 0.5, 1.0, "silverman"])

# Least squares: ridge strength λ.
LS_GRID = param_grid(reg_lambda=[1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0])

# MLP: hidden layers, learning rate, epochs and batch size. Momentum (0.9),
# L2 strength (1e-4) and ReLU activation are fixed.
MLP_GRID = param_grid(hidden_sizes=[(32, 16), (16,)],
                      learning_rate=[1e-3, 1e-2],
                      n_epochs=[50, 100],
                      batch_size=[32, 64])

# Bagged naive Bayes: B = 50 bootstrap samples, bandwidth tuned as for NB.
N_BAGS = 50
BAG_GRID = NB_GRID


def make_mlp(**params):
    return MLPClassifier(momentum=0.9, alpha=1e-4, activation="relu",
                         random_state=SEED, **params)


def make_bagged(**params):
    return BaggedParzenNaiveBayes(n_estimators=N_BAGS, random_state=SEED, **params)


# ----------------------------------------------------------------------------
# Classifiers and experiments
# ----------------------------------------------------------------------------
# command-line name -> (display name, factory, grid, experiments it is part of)
# The dummy has no factory/grid; it is evaluated by experiment.evaluate_dummy.
CLASSIFIERS = {
    "dummy": ("Dummy (stratified)", None, None, ("e", "f", "h")),
    "naive_bayes": ("Naive Bayes (Parzen)", ParzenNaiveBayes, NB_GRID, ("e", "f", "h")),
    "least_squares": ("Least Squares", LeastSquaresClassifier, LS_GRID, ("e", "f", "h")),
    "mlp": ("MLP", make_mlp, MLP_GRID, ("e", "f", "h")),
    "bagging": ("Bagged NB (Parzen)", make_bagged, BAG_GRID, ("h",)),
}

# experiment key -> (title, feature matrix, output file)
EXPERIMENTS = {
    "e": ("(1e) Registry features", "X_registry", "registry_results.csv"),
    "f": ("(1f) Full features", "X_full", "full_results.csv"),
    "h": ("(1h) Bagging comparison", "X_full", "bagging_results.csv"),
}

# Row order of the result tables.
MODEL_ORDER = [display for display, *_ in CLASSIFIERS.values()]


def main(selected):
    data = load_data()
    y, groups = data["y"], data["groups"]

    # One split for all experiments (same rows for X_registry and X_full).
    split = group_train_val_test_split(groups, train_frac=0.70, val_frac=0.15, seed=SEED)
    split_info = {}
    for label, idx in zip(("train", "validation", "test"), split):
        split_info[label] = {"sites": int(len(np.unique(groups[idx]))), "rows": int(len(idx)),
                             "positive_rate": float(y[idx].mean())}
        print(f"{label:10s}: {split_info[label]['sites']:3d} sites, "
              f"{split_info[label]['rows']:4d} rows, positive rate "
              f"{split_info[label]['positive_rate']:.3f}")

    # Previous run configuration, so validation tables of classifiers that are
    # not run this time are kept.
    config_path = os.path.join(RESULTS_DIR, "run_config.json")
    validation = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            validation = json.load(f).get("validation", {})

    for key, (title, X_key, filename) in EXPERIMENTS.items():
        # Classifiers selected on the command line that belong to this experiment.
        names = [n for n in CLASSIFIERS if n in selected and key in CLASSIFIERS[n][3]]
        if not names:
            continue
        models = {CLASSIFIERS[n][0]: (CLASSIFIERS[n][1], CLASSIFIERS[n][2])
                  for n in names if n != "dummy"}
        rows, tables = run_experiment(title, data[X_key], y, split, models,
                                      include_dummy="dummy" in names)
        print("saved", save_csv(rows, filename, MODEL_ORDER))
        validation.setdefault(title, {}).update(tables)

    config = {"seed": SEED, "split": split_info, "n_bags": N_BAGS,
              "dummy_random_states": f"{SEED}-{SEED + N_DUMMY_REPEATS - 1}",
              "grids": {"naive_bayes": NB_GRID, "least_squares": LS_GRID,
                        "mlp": MLP_GRID, "bagging": BAG_GRID},
              "validation": validation}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    print("saved", config_path)


if __name__ == "__main__":
    args = sys.argv[1:] or list(CLASSIFIERS)
    unknown = [a for a in args if a not in CLASSIFIERS]
    if unknown:
        sys.exit(f"unknown classifier(s): {', '.join(unknown)}\n"
                 f"choose from: {', '.join(CLASSIFIERS)}")
    main(set(args))
