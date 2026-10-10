"""
Problem 2 entry point:  python main.py

Steps
-----
1. Load X_reference (915 × 8) and X_query (246 × 8).
2. Preprocess: log(1 + x) for all features except pH, then standardise with
   the reference mean / standard deviation (ddof = 0).
3. Fit PCA (2 components) on the reference set; project both sets.     (2b)
4. Reconstruct, compute residuals and the anomaly score S(x).          (2c)
5. Threshold τ = 95th percentile of the reference scores; flag query
   records with S > τ and list the ten highest-scoring query records.  (2d)

Every fitted quantity (preprocessing, PCA, residual statistics, threshold)
comes from X_reference only.

Outputs (results/problem2/):
    explained_variance.csv   eigenvalue and (cumulative) explained variance per component
    pca_projection.png       reference and query data in the PC1-PC2 plane
    top10_anomalies.csv      ten highest-scoring query records
    summary.txt              all numbers reported in (2b) and (2d)
"""

import os
import csv
import numpy as np

from data_utils import load_data, Preprocessor, RESULTS_DIR
from pca import PCA
from anomaly import PCAAnomalyScorer
from visualization import plot_pca_projection

N_COMPONENTS = 2
QUANTILE = 0.95


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    lines = []                                       # collected for summary.txt

    def report(text=""):
        print(text)
        lines.append(text)

    # ---- Step 1: data ------------------------------------------------------
    data = load_data()
    X_ref_raw, X_qry_raw = data["X_reference"], data["X_query"]
    names, record_ids = data["feature_names"], data["record_ids_query"]
    report(f"X_reference: {X_ref_raw.shape}, X_query: {X_qry_raw.shape}")
    report(f"features: {', '.join(names)}")

    # ---- Step 2: preprocessing (fitted on the reference set) --------------
    prep = Preprocessor(names).fit(X_ref_raw)
    X_ref = prep.transform(X_ref_raw)
    X_qry = prep.transform(X_qry_raw)

    # ---- Step 3: PCA (2b) --------------------------------------------------
    pca = PCA(n_components=N_COMPONENTS).fit(X_ref)
    Z_ref, Z_qry = pca.transform(X_ref), pca.transform(X_qry)
    evr, cum = pca.explained_variance_ratio_, pca.cumulative_variance_ratio_

    report("\n(2b) Explained variance")
    for i in range(N_COMPONENTS):
        report(f"  PC{i + 1}: {evr[i]:.4f} ({100 * evr[i]:.2f}%)   cumulative: "
               f"{cum[i]:.4f} ({100 * cum[i]:.2f}%)")

    with open(os.path.join(RESULTS_DIR, "explained_variance.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["component", "eigenvalue", "explained_variance_ratio", "cumulative_ratio"])
        for i, (lam, r, c) in enumerate(zip(pca.eigenvalues_, evr, cum)):
            w.writerow([f"PC{i + 1}", f"{lam:.6f}", f"{r:.6f}", f"{c:.6f}"])

    # Three features with the largest absolute coefficients in PC1.
    pc1 = pca.components_[:, 0]
    top3 = np.argsort(np.abs(pc1))[::-1][:3]
    report("\n(2b) PC1: three largest absolute coefficients")
    for j in top3:
        report(f"  {names[j]:12s} {pc1[j]:+.4f}")
    report("  all PC1 coefficients: " +
           ", ".join(f"{n} {c:+.3f}" for n, c in zip(names, pc1)))

    plot_pca_projection(Z_ref, Z_qry, evr, os.path.join(RESULTS_DIR, "pca_projection.png"))

    # ---- Step 4 + 5: anomaly score and detection (2c, 2d) -----------------
    scorer = PCAAnomalyScorer(pca, quantile=QUANTILE).fit(X_ref)
    if len(scorer.excluded_):
        report(f"\nresidual columns excluded (zero std): "
               f"{[names[j] for j in scorer.excluded_]}")
    query_scores = scorer.score(X_qry)
    flagged = query_scores > scorer.threshold_          # strictly greater

    report(f"\n(2d) Threshold tau = {QUANTILE:.0%} quantile of the reference scores = "
           f"{scorer.threshold_:.4f}")
    report(f"  reference records above tau: {np.sum(scorer.reference_scores_ > scorer.threshold_)}"
           f" / {len(X_ref)}")
    report(f"  flagged query records: {int(flagged.sum())} / {len(X_qry)}")

    top10 = np.argsort(query_scores)[::-1][:10]
    report("\n(2d) Ten highest-scoring query records")
    report(f"  {'rank':>4s}  {'record_id':14s} {'score':>9s}  flagged")
    for rank, i in enumerate(top10, start=1):
        report(f"  {rank:4d}  {record_ids[i]:14s} {query_scores[i]:9.3f}  {bool(flagged[i])}")

    with open(os.path.join(RESULTS_DIR, "top10_anomalies.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "record_id", "score", "flagged"])
        for rank, i in enumerate(top10, start=1):
            w.writerow([rank, record_ids[i], f"{query_scores[i]:.6f}", int(flagged[i])])

    with open(os.path.join(RESULTS_DIR, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\nsaved results to", RESULTS_DIR)


if __name__ == "__main__":
    main()
