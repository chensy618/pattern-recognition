"""
Problem (2b): 2-D PCA scatter plot of the reference and query sets.
"""

import matplotlib
matplotlib.use("Agg")                       # render to file, no display needed
import matplotlib.pyplot as plt

BLUE, ORANGE = "#2a78d6", "#eb6834"
TEXT, TEXT_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def plot_pca_projection(Z_reference, Z_query, explained_variance_ratio, path):
    """
    Scatter plot of both sets in the space of the first two principal
    components (fitted on the reference set). Saved as PNG to `path`.
    """
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(Z_reference[:, 0], Z_reference[:, 1], s=14, color=BLUE, alpha=0.55,
               edgecolors="none", label=f"Reference (n = {len(Z_reference)})")
    ax.scatter(Z_query[:, 0], Z_query[:, 1], s=14, color=ORANGE, alpha=0.75,
               edgecolors="none", label=f"Query (n = {len(Z_query)})")
    ax.set_xlabel(f"PC1 ({100 * explained_variance_ratio[0]:.1f}% of variance)", color=TEXT)
    ax.set_ylabel(f"PC2 ({100 * explained_variance_ratio[1]:.1f}% of variance)", color=TEXT)
    ax.set_title("Reference and query data in the first two principal components",
                 color=TEXT, loc="left", fontsize=10)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=TEXT_2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
