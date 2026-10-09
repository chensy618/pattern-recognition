"""
Shared experiment logic for Problems (1e), (1f) and (1h).

Every classifier implements the same interface

    model.fit(X_train, y_train)
    y_pred = model.predict(X)
    p      = model.predict_proba(X)        (if the model outputs probabilities)
    s      = model.decision_function(X)    (least squares: linear output)

so that one function can train, tune and test all of them.

Protocol
--------
1. Preprocessing (median imputation + standardisation) is fitted on the
   training set and applied to validation and test sets.
2. For every hyperparameter setting the model is trained on the training set
   and scored on the validation set.
3. The decision threshold is treated as an additional hyperparameter: for each
   setting the threshold that maximises the validation F1 is found. The setting
   (and threshold) with the highest validation F1 is selected.
4. The selected model is evaluated once on the test set, with
       "default" - the model's own rule (argmax posterior / ŷ >= 0.5), and
       "tuned"   - the threshold selected on the validation set.
   Reporting both shows the effect of the class imbalance (19% breachers) on
   the default rules.
"""

import os
import csv
import json
import itertools
import numpy as np

from data_utils import Preprocessor, RESULTS_DIR, SEED
from metrics import precision_recall_f1, best_f1_threshold
from dummy import DummyClassifier

N_DUMMY_REPEATS = 100     # repetitions of the random dummy classifier


def param_grid(**axes):
    """Cartesian product of hyperparameter values -> list of dicts."""
    keys = list(axes)
    return [dict(zip(keys, values)) for values in itertools.product(*axes.values())]


def positive_score(model, X):
    """Continuous score for the positive class: P(y=1|x) or the linear output."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def prepare_data(X, split):
    """Fit the preprocessing on the training rows and transform all three sets."""
    tr, va, te = split
    prep = Preprocessor().fit(X[tr])
    return prep.transform(X[tr]), prep.transform(X[va]), prep.transform(X[te])


# ----------------------------------------------------------------------------
# Learned classifiers
# ----------------------------------------------------------------------------
def select_and_evaluate(name, factory, grid, X_tr, y_tr, X_va, y_va, X_te, y_te,
                        verbose=True):
    """
    Hyperparameter + threshold selection on the validation set, then a single
    evaluation on the test set. Inputs are already preprocessed.
    Returns (result rows, validation table of all settings).
    """
    table, best = [], None
    for params in grid:
        model = factory(**params).fit(X_tr, y_tr)
        s_va = positive_score(model, X_va)
        f1_default = precision_recall_f1(y_va, model.predict(X_va))[2]
        thr, f1_tuned = best_f1_threshold(y_va, s_va)
        table.append({"params": params, "val_f1_default": f1_default,
                      "val_f1_tuned": f1_tuned, "threshold": thr})
        if best is None or f1_tuned > best[0]:
            best = (f1_tuned, params, thr, model)

    _, params, thr, model = best
    s_te = positive_score(model, X_te)
    rows = []
    for rule, y_pred in (("default", model.predict(X_te)),
                         ("tuned", (s_te >= thr).astype(int))):
        p, r, f = precision_recall_f1(y_te, y_pred)
        rows.append({"model": name, "rule": rule, "precision": p, "recall": r,
                     "f1": f, "precision_std": "", "recall_std": "", "f1_std": "",
                     "hyperparameters": json.dumps(params),
                     "threshold": thr if rule == "tuned" else "default"})
    if verbose:
        print(f"  {name:22s} selected {params}, threshold {thr:.4g} "
              f"(val F1 {best[0]:.3f})")
    return rows, table


# ----------------------------------------------------------------------------
# Dummy baseline
# ----------------------------------------------------------------------------
def evaluate_dummy(X_tr_raw, y_tr, X_te_raw, y_te, n_repeats=N_DUMMY_REPEATS,
                   random_state=SEED):
    """
    Stratified random dummy, repeated with `n_repeats` random states
    (random_state, random_state + 1, ..., random_state + n_repeats - 1)
    because its predictions are random. Returns one row with mean and standard deviation.
    """
    scores = []
    for rs in range(random_state, random_state + n_repeats):
        dummy = DummyClassifier(random_state=rs).fit(X_tr_raw, y_tr)
        scores.append(precision_recall_f1(y_te, dummy.predict(X_te_raw)))
    mean, std = np.mean(scores, axis=0), np.std(scores, axis=0)
    return {"model": "Dummy (stratified)", "rule": f"mean of {n_repeats} runs",
            "precision": mean[0], "recall": mean[1], "f1": mean[2],
            "precision_std": std[0], "recall_std": std[1], "f1_std": std[2],
            "hyperparameters": json.dumps({"random_state": f"{random_state}-{random_state + n_repeats - 1}"}),
            "threshold": ""}


# ----------------------------------------------------------------------------
# Full experiment
# ----------------------------------------------------------------------------
def run_experiment(title, X, y, split, models, include_dummy=True, verbose=True):
    """
    Train, tune and test every model in `models` (name -> (factory, grid)) and,
    if `include_dummy`, evaluate the dummy baseline, all on the same split.
    Returns (rows, validation_tables).
    """
    tr, va, te = split
    if verbose:
        print(f"\n=== {title} ({X.shape[1]} features) ===")
    X_tr, X_va, X_te = prepare_data(X, split)

    rows, tables = [], {}
    if include_dummy:
        rows.append(evaluate_dummy(X[tr], y[tr], X[te], y[te]))

    for name, (factory, grid) in models.items():
        model_rows, tables[name] = select_and_evaluate(
            name, factory, grid, X_tr, y[tr], X_va, y[va], X_te, y[te], verbose)
        rows.extend(model_rows)

    if verbose:
        print(format_table(rows))
    return rows, tables


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------
CSV_FIELDS = ["model", "rule", "precision", "recall", "f1", "precision_std",
              "recall_std", "f1_std", "hyperparameters", "threshold"]


def _fmt(v):
    return f"{v:.4f}" if isinstance(v, (float, np.floating)) else v


def save_csv(rows, filename, model_order):
    """
    Write the result rows to results/<filename>.

    If the file already exists, the rows of models that were not run this time
    are kept, so running a single classifier only updates its own rows. Rows
    of models that are no longer in `model_order` are dropped. Rows are
    ordered by `model_order` (list of model names).
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    new_models = {r["model"] for r in rows}
    kept = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            kept = [r for r in csv.DictReader(f)
                    if r["model"] not in new_models and r["model"] in model_order]
    rank = {m: i for i, m in enumerate(model_order)}
    # Stable sort: the default/tuned order within a model is preserved.
    all_rows = sorted(kept + rows, key=lambda r: rank.get(r["model"], len(rank)))

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: _fmt(row.get(k, "")) for k in CSV_FIELDS})
    return path


def format_table(rows):
    """Markdown table (Model | Rule | Precision | Recall | F1) for the report."""
    lines = ["| Model | Rule | Precision | Recall | F1 |", "|---|---|---|---|---|"]
    for r in rows:
        cells = []
        for k in ("precision", "recall", "f1"):
            std = r[k + "_std"]
            cells.append(f"{r[k]:.3f} ± {std:.3f}" if std != "" else f"{r[k]:.3f}")
        lines.append(f"| {r['model']} | {r['rule']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
