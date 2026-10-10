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

The protocol is applied to one split (single hold-out) or to each of the k
splits of a group k-fold; in the latter case aggregate_folds() reports the
mean and standard deviation of every metric over the folds.

Option C (run_experiment_cv): only a train / test split; steps 2-3 use a
group k-fold inside the training set (pooled inner-validation scores), and the
selected model is refitted on the whole training set before testing.
"""

import os
import csv
import json
import itertools
import numpy as np

from data_utils import Preprocessor, group_kfold, RESULTS_DIR, SEED
from metrics import precision_recall_f1, accuracy, best_f1_threshold
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
def validation_entry(params, y_val, default_pred, scores):
    """
    Validation metrics of one hyperparameter setting: precision, recall, F1 and
    accuracy with the default rule and with the F1-optimal threshold.
    """
    thr, _ = best_f1_threshold(y_val, scores)
    entry = {"params": params, "threshold": thr}
    for rule, y_pred in (("default", default_pred), ("tuned", (scores >= thr).astype(int))):
        p, r, f = precision_recall_f1(y_val, y_pred)
        entry.update({f"val_precision_{rule}": p, f"val_recall_{rule}": r,
                      f"val_f1_{rule}": f, f"val_accuracy_{rule}": accuracy(y_val, y_pred)})
    return entry

def select_and_evaluate(name, factory, grid, X_tr, y_tr, X_va, y_va, X_te, y_te,
                        verbose=True):
    """
    Hyperparameter + threshold selection on the validation set, then a single
    evaluation on the test set. Inputs are already preprocessed.
    Returns (result rows, validation table of all settings).
    """
    table, best = [], None
    for i, params in enumerate(grid):
        model = factory(**params).fit(X_tr, y_tr)
        entry = validation_entry(params, y_va, model.predict(X_va), positive_score(model, X_va))
        table.append(entry)
        if best is None or entry["val_f1_tuned"] > best[0]:
            best = (entry["val_f1_tuned"], params, entry["threshold"], model, i)

    _, params, thr, model, i_best = best
    for i, entry in enumerate(table):
        entry["selected"] = i == i_best
    if verbose:
        print(f"  {name:22s} selected {params}, threshold {thr:.4g} "
              f"(val F1 {best[0]:.3f})")
    return test_rows(name, model, params, thr, X_te, y_te), table


def test_rows(name, model, params, thr, X_te, y_te):
    """Test-set metrics of a fitted model with the default rule and threshold `thr`."""
    s_te = positive_score(model, X_te)
    rows = []
    for rule, y_pred in (("default", model.predict(X_te)),
                         ("tuned", (s_te >= thr).astype(int))):
        p, r, f = precision_recall_f1(y_te, y_pred)
        rows.append({"model": name, "rule": rule, "precision": p, "recall": r,
                     "f1": f, "accuracy": accuracy(y_te, y_pred),
                     "precision_std": "", "recall_std": "", "f1_std": "", "accuracy_std": "",
                     "hyperparameters": json.dumps(params),
                     "threshold": thr if rule == "tuned" else "default"})
    return rows


def select_and_evaluate_cv(name, factory, grid, inner, X_tr, y_tr, X_te, y_te, verbose=True):
    """
    Option C: hyperparameter + threshold selection by group k-fold *inside*
    the training set, then refit on the whole training set and test once.

    `inner` is a list of preprocessed inner folds (X_fit, y_fit, X_val, y_val).
    For every setting the validation scores of all inner folds are pooled
    before computing F1 and the F1-optimal threshold. X_tr / X_te are the
    full training and test sets, preprocessed with statistics of X_tr.
    Returns (result rows, validation table of all settings).
    """
    table, best = [], None
    for params in grid:
        scores, labels, default_preds = [], [], []
        for X_fit, y_fit, X_val, y_val in inner:
            model = factory(**params).fit(X_fit, y_fit)
            scores.append(positive_score(model, X_val))
            default_preds.append(model.predict(X_val))
            labels.append(y_val)
        entry = validation_entry(params, np.concatenate(labels), np.concatenate(default_preds),
                                 np.concatenate(scores))
        table.append(entry)
        if best is None or entry["val_f1_tuned"] > best[0]:
            best = (entry["val_f1_tuned"], params, entry["threshold"], len(table) - 1)

    f1_val, params, thr, i_best = best
    for i, entry in enumerate(table):
        entry["selected"] = i == i_best
    # Refit on the whole training set with the selected hyperparameters; the
    # threshold selected on the pooled inner-validation scores is reused.
    model = factory(**params).fit(X_tr, y_tr)
    if verbose:
        print(f"  {name:22s} selected {params}, threshold {thr:.4g} "
              f"(inner-CV F1 {f1_val:.3f})")
    return test_rows(name, model, params, thr, X_te, y_te), table


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
        y_pred = dummy.predict(X_te_raw)
        scores.append([*precision_recall_f1(y_te, y_pred), accuracy(y_te, y_pred)])
    mean, std = np.mean(scores, axis=0), np.std(scores, axis=0)
    return {"model": "Dummy (stratified)", "rule": f"mean of {n_repeats} runs",
            "precision": mean[0], "recall": mean[1], "f1": mean[2], "accuracy": mean[3],
            "precision_std": std[0], "recall_std": std[1], "f1_std": std[2],
            "accuracy_std": std[3],
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


def run_experiment_cv(title, X, y, groups, split, models, inner_k=5, include_dummy=True,
                      verbose=True):
    """
    Option C ("holdout_cv"): `split` = (train_idx, test_idx). Hyperparameters
    and thresholds are selected with a group k-fold (`inner_k` folds) inside
    the training set; the final models are refitted on the whole training set.
    Returns (rows, validation_tables).
    """
    tr, te = split
    if verbose:
        print(f"\n=== {title} ({X.shape[1]} features, {inner_k}-fold CV in training set) ===")

    # Inner folds, preprocessed once (statistics of each inner fitting part).
    inner = []
    for fit_pos, val_pos in group_kfold(groups[tr], inner_k, seed=SEED):
        fit_idx, val_idx = tr[fit_pos], tr[val_pos]
        prep = Preprocessor().fit(X[fit_idx])
        inner.append((prep.transform(X[fit_idx]), y[fit_idx],
                      prep.transform(X[val_idx]), y[val_idx]))
    # Final preprocessing on the whole training set.
    prep = Preprocessor().fit(X[tr])
    X_tr, X_te = prep.transform(X[tr]), prep.transform(X[te])

    rows, tables = [], {}
    if include_dummy:
        rows.append(evaluate_dummy(X[tr], y[tr], X[te], y[te]))
    for name, (factory, grid) in models.items():
        model_rows, tables[name] = select_and_evaluate_cv(
            name, factory, grid, inner, X_tr, y[tr], X_te, y[te], verbose)
        rows.extend(model_rows)

    if verbose:
        print(format_table(rows))
    return rows, tables


METRIC_KEYS = ("precision", "recall", "f1", "accuracy")


def aggregate_folds(fold_rows):
    """
    Combine the result rows of k folds (list of k row lists) into one row per
    (model, rule): mean and standard deviation of every metric over the folds.
    The hyperparameters and thresholds selected in each fold are kept as lists.
    """
    groups = {}
    for rows in fold_rows:
        for r in rows:
            groups.setdefault((r["model"], r["rule"]), []).append(r)

    out = []
    for (model, rule), rs in groups.items():
        row = {"model": model, "rule": rule if not model.startswith("Dummy")
               else f"{rule}, mean of {len(rs)} folds"}
        for k in METRIC_KEYS:
            values = np.array([r[k] for r in rs], dtype=float)
            row[k], row[k + "_std"] = values.mean(), values.std()
        row["hyperparameters"] = json.dumps(
            [json.loads(r["hyperparameters"]) if r["hyperparameters"] else None for r in rs])
        row["threshold"] = json.dumps([r["threshold"] for r in rs], default=float)
        out.append(row)
    return out


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------
CSV_FIELDS = ["model", "rule", "precision", "recall", "f1", "accuracy", "precision_std",
              "recall_std", "f1_std", "accuracy_std", "hyperparameters", "threshold"]


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


VALIDATION_FIELDS = (["model", "fold", "hyperparameters", "threshold", "selected"] +
                     [f"val_{m}_{rule}" for rule in ("default", "tuned") for m in METRIC_KEYS])


def save_validation_csv(tables, filename, model_order):
    """
    Validation metrics of every hyperparameter setting -> results/<filename>.

    `tables` maps model -> list of settings (one split) or model -> list of
    folds, each a list of settings (k-fold). Rows of models that were not run
    this time are kept, as in save_csv.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    rows = []
    for model, entries in tables.items():
        folds = entries if entries and isinstance(entries[0], list) else [entries]
        for k, fold in enumerate(folds, start=1):
            for e in fold:
                row = {f: e.get(f, "") for f in VALIDATION_FIELDS}
                row.update({"model": model, "fold": k if len(folds) > 1 else "",
                            "hyperparameters": json.dumps(e["params"]),
                            "selected": int(e.get("selected", False))})
                rows.append(row)

    kept = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            kept = [r for r in csv.DictReader(f)
                    if r["model"] not in tables and r["model"] in model_order]
    rank = {m: i for i, m in enumerate(model_order)}
    all_rows = sorted(kept + rows, key=lambda r: rank.get(r["model"], len(rank)))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=VALIDATION_FIELDS)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: _fmt(row.get(k, "")) for k in VALIDATION_FIELDS})
    return path


def format_table(rows):
    """Markdown table (Model | Rule | Precision | Recall | F1 | Accuracy) for the report."""
    lines = ["| Model | Rule | Precision | Recall | F1 | Accuracy |", "|---|---|---|---|---|---|"]
    for r in rows:
        cells = []
        for k in ("precision", "recall", "f1", "accuracy"):
            std = r[k + "_std"]
            cells.append(f"{r[k]:.3f} ± {std:.3f}" if std != "" else f"{r[k]:.3f}")
        lines.append(f"| {r['model']} | {r['rule']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
