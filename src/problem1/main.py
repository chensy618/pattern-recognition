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
(1e)  registry features only (X_registry, 28 features) -> registry_results.csv
(1f)  all features (X_full, 31 features)               -> full_results.csv
(1h)  all features, incl. the bagged naive Bayes       -> bagging_results.csv

For every experiment the validation precision, recall, F1 and accuracy of
every hyperparameter setting (default rule and tuned threshold) are written to
registry_validation.csv / full_validation.csv / bagging_validation.csv; the
column `selected` marks the chosen setting.

Running a subset of classifiers only replaces their own rows in the CSV
files; the rows of the other classifiers are kept.

Split (parameter SPLIT_METHOD below):
    "holdout"  one group-aware 70/15/15 train/validation/test split.
    "kfold"    group k-fold over sites (K_FOLDS folds): each fold
                       is the test set once; the remaining sites are split by
                       site into training and validation (70 : 15). Metrics are
                       reported as mean ± std over the folds, in files with the
                       suffix _kfold (e.g. full_results_kfold.csv).
    "holdout_cv"  train / test only, by site (CV_TEST_FRAC of the sites,
                       default 15%, in the test set). Hyperparameters and
                       thresholds are selected with a group k-fold
                       (INNER_FOLDS folds) inside the training set; the final
                       models are refitted on the whole training set. Files
                       with the suffix _cv.

All experiments use the same split(s). The split, the random seed, the
hyperparameter grids, the selected hyperparameters and the validation score
of every setting are saved to results/problem1/run_config.json
(run_config_kfold.json / run_config_cv.json).
"""

import os
import sys
import json
import numpy as np

from data_utils import (load_data, group_train_val_test_split, group_kfold_splits,
                        RESULTS_DIR, SEED)
from experiment import (param_grid, run_experiment, run_experiment_cv, aggregate_folds,
                        format_table, save_csv, save_validation_csv, N_DUMMY_REPEATS)
from naive_bayes import ParzenNaiveBayes
from least_squares import LeastSquaresClassifier
from mlp import MLPClassifier
from bagging import BaggedParzenNaiveBayes

# ----------------------------------------------------------------------------
# Data split: "holdout" (one 70/15/15 split), "kfold" (group k-fold) or
# "holdout_cv" (train/test split, cross-validation inside the training set)
# ----------------------------------------------------------------------------
SPLIT_METHOD = "holdout"
K_FOLDS = 5             # number of folds, only used when SPLIT_METHOD = "kfold"
INNER_FOLDS = 5         # inner CV folds, only used when SPLIT_METHOD = "holdout_cv"
CV_TEST_FRAC = 0.15     # test fraction of the sites, only used when SPLIT_METHOD = "holdout_cv"

SUFFIXES = {"holdout": "", "kfold": "_kfold", "holdout_cv": "_cv"}

# ----------------------------------------------------------------------------
# Hyperparameter grids (selected on the validation set)
# ----------------------------------------------------------------------------
# Naive Bayes: kernel bandwidth h (standardised units) or Silverman's rule.
NB_GRID = param_grid(bandwidth=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, "silverman"])

# Least squares: ridge strength λ.
LS_GRID = param_grid(reg_lambda=[1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5])

# MLP: hidden layers, learning rate, epochs and batch size. Momentum (0.9),
# L2 strength (1e-4) and ReLU activation are fixed.
MLP_GRID = param_grid(hidden_sizes=[(16,), (32, 16), (64, 32)],
                      learning_rate=[1e-3, 3e-3, 1e-2],
                      n_epochs=[50, 100, 200],
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


def split_summary(groups, y, split):
    """Number of sites, rows and positive rate of each part of the split."""
    labels = ("train", "validation", "test") if len(split) == 3 else ("train", "test")
    return {label: {"sites": int(len(np.unique(groups[idx]))), "rows": int(len(idx)),
                    "positive_rate": float(y[idx].mean())}
            for label, idx in zip(labels, split)}


def print_split(info, prefix=""):
    for label, v in info.items():
        print(f"{prefix}{label:10s}: {v['sites']:3d} sites, {v['rows']:4d} rows, "
              f"positive rate {v['positive_rate']:.3f}")


def main(selected, split_method=SPLIT_METHOD, k=K_FOLDS, inner_k=INNER_FOLDS):
    data = load_data()
    y, groups = data["y"], data["groups"]

    # The same split(s) for all experiments (same rows for X_registry and X_full).
    if split_method not in SUFFIXES:
        raise ValueError(f"SPLIT_METHOD must be one of {list(SUFFIXES)}, not {split_method!r}")
    suffix = SUFFIXES[split_method]
    if split_method == "holdout":
        splits = [group_train_val_test_split(groups, train_frac=0.70, val_frac=0.15, seed=SEED)]
    elif split_method == "holdout_cv":
        # Train / test only (no validation set): (1 - CV_TEST_FRAC) / CV_TEST_FRAC by site.
        tr, _, te = group_train_val_test_split(groups, train_frac=1 - CV_TEST_FRAC,
                                               val_frac=0.0, seed=SEED)
        splits = [(tr, te)]
    else:
        splits = group_kfold_splits(groups, k=k, seed=SEED)
    split_info = [split_summary(groups, y, sp) for sp in splits]
    for i, info in enumerate(split_info):
        if split_method == "kfold":
            print(f"fold {i + 1}/{k}")
        print_split(info, prefix="  " if split_method == "kfold" else "")

    # Previous run configuration, so validation tables of classifiers that are
    # not run this time are kept.
    config_path = os.path.join(RESULTS_DIR, f"run_config{suffix}.json")
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

        fold_rows, fold_tables = [], []
        for i, sp in enumerate(splits):
            fold_title = title if len(splits) == 1 else f"{title}, fold {i + 1}/{k}"
            if split_method == "holdout_cv":
                rows, tables = run_experiment_cv(fold_title, data[X_key], y, groups, sp, models,
                                                 inner_k=inner_k, include_dummy="dummy" in names)
            else:
                rows, tables = run_experiment(fold_title, data[X_key], y, sp, models,
                                              include_dummy="dummy" in names,
                                              verbose=len(splits) == 1)
            fold_rows.append(rows)
            fold_tables.append(tables)

        if len(splits) == 1:
            rows, tables = fold_rows[0], fold_tables[0]
        else:
            rows = aggregate_folds(fold_rows)
            # Validation tables per model: one list of settings per fold.
            tables = {m: [t[m] for t in fold_tables] for m in fold_tables[0]}
            print(f"\n=== {title}: mean ± std over {k} folds ===")
            print(format_table(rows))

        base, ext = os.path.splitext(filename)
        print("saved", save_csv(rows, f"{base}{suffix}{ext}", MODEL_ORDER))
        # Validation metrics of every hyperparameter setting.
        print("saved", save_validation_csv(tables, f"{base.replace('_results', '_validation')}"
                                                   f"{suffix}{ext}", MODEL_ORDER))
        validation.setdefault(title, {}).update(tables)

    config = {"seed": SEED, "split_method": split_method,
              "k": k if split_method == "kfold" else None,
              "inner_folds": inner_k if split_method == "holdout_cv" else None,
              "cv_test_frac": CV_TEST_FRAC if split_method == "holdout_cv" else None,
              "split": split_info if split_method == "kfold" else split_info[0],
              "n_bags": N_BAGS,
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
