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
| `metrics.py` | | Precision, recall, F1 |
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

Results are written to `results/problem1/` (`registry_results.csv`, `full_results.csv`, `bagging_results.csv`, `run_config.json`).

## Problem 2 (`problem2/`)

Not implemented yet. Results will go to `results/problem2/`.
