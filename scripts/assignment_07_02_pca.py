# ============================================================
# Assignment 7: PCA and Cluster Analysis of Wine Data
#
# assignment_07_02_pca.py
#
# Purpose:
#   Standardize the wine data, conduct PCA, determine the
#   number of components required to explain at least 80%
#   of total variance, and save report-ready artifacts.
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# Load wine data
# ------------------------------------------------------------

wine = pd.read_csv(
    DATA_DIR / "wine-clustering.csv"
)


# ------------------------------------------------------------
# Standardize variables
# ------------------------------------------------------------

scaler = StandardScaler()

wine_scaled_array = scaler.fit_transform(wine)

wine_scaled = pd.DataFrame(
    wine_scaled_array,
    columns=wine.columns,
    index=wine.index
)


# ------------------------------------------------------------
# Save standardized data
# ------------------------------------------------------------

wine_scaled.to_csv(
    TABLE_DIR / "wine_standardized_data.csv",
    index=False
)


# ------------------------------------------------------------
# Conduct PCA
#
# PCA is initially performed with all components so that
# cumulative explained variance can be examined.
# ------------------------------------------------------------

pca = PCA()

pca_scores_array = pca.fit_transform(
    wine_scaled
)


# ------------------------------------------------------------
# PCA explained variance table
# ------------------------------------------------------------

explained_variance = pd.DataFrame(
    {
        "Principal_Component": [
            f"PC{i + 1}"
            for i in range(len(pca.explained_variance_))
        ],
        "Eigenvalue": pca.explained_variance_,
        "Proportion_Variance": pca.explained_variance_ratio_,
        "Cumulative_Variance": np.cumsum(
            pca.explained_variance_ratio_
        )
    }
)

explained_variance[
    "Percentage_Variance"
] = (
    explained_variance[
        "Proportion_Variance"
    ] * 100
)

explained_variance[
    "Cumulative_Percentage"
] = (
    explained_variance[
        "Cumulative_Variance"
    ] * 100
)


explained_variance.to_csv(
    TABLE_DIR / "pca_explained_variance.csv",
    index=False
)


# ------------------------------------------------------------
# Determine number of PCs required for >= 80% variance
# ------------------------------------------------------------

n_components_80 = (
    np.argmax(
        explained_variance[
            "Cumulative_Variance"
        ].values >= 0.80
    )
    + 1
)


variance_at_80 = (
    explained_variance.loc[
        n_components_80 - 1,
        "Cumulative_Variance"
    ]
)


# ------------------------------------------------------------
# Print PCA summary
# ------------------------------------------------------------

print("PCA ANALYSIS")
print("=" * 60)

print(
    f"Original variables: {wine.shape[1]}"
)

print(
    f"Observations: {wine.shape[0]}"
)

print(
    f"\nPrincipal components required for "
    f"at least 80% variance: {n_components_80}"
)

print(
    f"Variance explained by {n_components_80} "
    f"components: {variance_at_80:.4%}"
)

print("\nEXPLAINED VARIANCE")
print("=" * 60)

print(
    explained_variance[
        [
            "Principal_Component",
            "Eigenvalue",
            "Percentage_Variance",
            "Cumulative_Percentage"
        ]
    ].round(4)
)


# ------------------------------------------------------------
# PCA loadings
# ------------------------------------------------------------

loadings = pd.DataFrame(
    pca.components_.T,
    index=wine.columns,
    columns=[
        f"PC{i + 1}"
        for i in range(len(wine.columns))
    ]
)

loadings.index.name = "Variable"

loadings.to_csv(
    TABLE_DIR / "pca_loadings_all_components.csv"
)


# ------------------------------------------------------------
# Retained loadings
# ------------------------------------------------------------

retained_loadings = loadings.iloc[
    :,
    :n_components_80
]

retained_loadings.to_csv(
    TABLE_DIR / "pca_loadings_retained.csv"
)


# ------------------------------------------------------------
# PCA scores
# ------------------------------------------------------------

pca_scores = pd.DataFrame(
    pca_scores_array,
    columns=[
        f"PC{i + 1}"
        for i in range(len(pca_scores_array[0]))
    ]
)

pca_scores_retained = pca_scores.iloc[
    :,
    :n_components_80
]

pca_scores_retained.to_csv(
    TABLE_DIR / "pca_scores_retained.csv",
    index=False
)


# ------------------------------------------------------------
# Print loadings for retained components
# ------------------------------------------------------------

print("\nRETAINED PCA LOADINGS")
print("=" * 60)

print(
    retained_loadings.round(4)
)


# ------------------------------------------------------------
# Cumulative variance plot
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)

components = np.arange(
    1,
    len(explained_variance) + 1
)

cumulative_variance = (
    explained_variance[
        "Cumulative_Variance"
    ]
)

plt.plot(
    components,
    cumulative_variance,
    marker="o",
    linewidth=2
)

plt.axhline(
    y=0.80,
    color="red",
    linestyle="--",
    linewidth=1.5,
    label="80% variance"
)

plt.axvline(
    x=n_components_80,
    color="green",
    linestyle="--",
    linewidth=1.5,
    label=(
        f"{n_components_80} components"
    )
)

plt.xlabel(
    "Number of Principal Components"
)

plt.ylabel(
    "Cumulative Proportion of Variance Explained"
)

plt.title(
    "PCA Cumulative Variance Explained"
)

plt.xticks(components)

plt.ylim(
    0,
    1.05
)

plt.grid(
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "pca_cumulative_variance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# Scree plot
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    components,
    explained_variance[
        "Proportion_Variance"
    ],
    marker="o",
    linewidth=2
)

plt.xlabel(
    "Principal Component"
)

plt.ylabel(
    "Proportion of Variance Explained"
)

plt.title(
    "PCA Scree Plot"
)

plt.xticks(components)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "pca_scree_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# Final confirmation
# ------------------------------------------------------------

print("\nPCA ARTIFACTS SAVED")
print("=" * 60)

print(
    "Tables:"
)

for artifact in sorted(
    TABLE_DIR.glob("pca_*.csv")
):
    print(
        f"  {artifact.name}"
    )

print(
    "\nFigures:"
)

for artifact in sorted(
    FIGURE_DIR.glob("pca_*.png")
):
    print(
        f"  {artifact.name}"
    )

print(
    "\nassignment_07_02_pca.py "
    "completed successfully."
)
