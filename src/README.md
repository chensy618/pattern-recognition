# src

Code for the FYS-3012 home exam. 

## Problem 1 (`problem1/`)

| File | Problem | Content |
|---|---|---|
| `naive_bayes.py` | 1a | Naive Bayes with Parzen windows |
| `least_squares.py` | 1b | Least-squares classifier |
| `mlp.py` | 1c | Multilayer perceptron |
| `dummy.py` | 1d | Stratified random baseline |
| `bagging.py` | 1g | Bagged naive Bayes |
| `data_utils.py` | | Data loading, split by site, preprocessing |
| `metrics.py` | | Precision, recall, F1, accuracy |
| `experiment.py` | | Training, tuning and testing |
| `main.py` | 1e, 1f, 1h | Runs the experiments, hyperparameter grids |
| `checks.py` | | Sanity checks on synthetic data |

```bash
cd problem1
python main.py                    # all classifiers
python main.py naive_bayes mlp    # selected classifiers
python checks.py                  # sanity checks
```

Names: `dummy`, `naive_bayes`, `least_squares`, `mlp`, `bagging`.

Split: set `SPLIT_METHOD` in `main.py` to `"holdout"` (one 70/15/15 split by site), `"kfold"` (group k-fold, `K_FOLDS` folds) or `"holdout_cv"` (train/test split by site with `CV_TEST_FRAC` = 15% test, hyperparameters chosen by `INNER_FOLDS`-fold CV inside the training set).

Results are written to `results/problem1/`: test metrics (`registry_results.csv`, `full_results.csv`, `bagging_results.csv`), validation metrics of every hyperparameter setting (`*_validation.csv`) and `run_config.json`; with the suffix `_kfold` or `_cv` for the other splits.

## Problem 2 (`problem2/`)

| File | Problem | Content |
|---|---|---|
| `data_utils.py` | | Data loading, log(1 + x) transform, standardisation |
| `pca.py` | 2a, 2b | PCA (covariance, eigendecomposition, projection, reconstruction) |
| `anomaly.py` | 2c, 2d | Reconstruction-based anomaly score, threshold |
| `visualization.py` | 2b | PC1–PC2 scatter plot |
| `main.py` | 2b–2d | Runs all steps |

```bash
cd problem2
python main.py
```

Results are written to `results/problem2/` (`explained_variance.csv`, `pca_projection.png`, `top10_anomalies.csv`, `summary.txt`).
