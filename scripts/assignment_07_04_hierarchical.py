# ============================================================
# Assignment 7: PCA and Cluster Analysis of Wine Data
#
# assignment_07_04_hierarchical.py
#
# Purpose:
#   Conduct hierarchical clustering using complete linkage
#   and Euclidean distance. Compare clustering using the
#   original unscaled variables with clustering after
#   standardization.
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import (
    linkage,
    dendrogram,
    fcluster
)
from scipy.spatial.distance import pdist


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

wine_scaled = pd.DataFrame(
    scaler.fit_transform(wine),
    columns=wine.columns
)


# ============================================================
# PART 1: UN SCALED DATA
# ============================================================

print("HIERARCHICAL CLUSTERING")
print("=" * 60)

print("\nUNSCALED DATA")
print("-" * 60)


# ------------------------------------------------------------
# Euclidean distances
# ------------------------------------------------------------

distance_unscaled = pdist(
    wine,
    metric="euclidean"
)


print(
    f"Number of pairwise distances: "
    f"{len(distance_unscaled):,}"
)

print(
    f"Minimum distance: "
    f"{distance_unscaled.min():.4f}"
)

print(
    f"Maximum distance: "
    f"{distance_unscaled.max():.4f}"
)


# ------------------------------------------------------------
# Complete linkage
# ------------------------------------------------------------

Z_unscaled = linkage(
    distance_unscaled,
    method="complete"
)


# ------------------------------------------------------------
# Dendrogram: unscaled
# ------------------------------------------------------------

plt.figure(
    figsize=(16, 8)
)

dendrogram(
    Z_unscaled,
    no_labels=True,
    color_threshold=None
)

plt.title(
    "Hierarchical Clustering: Complete Linkage, Unscaled Data"
)

plt.xlabel(
    "Wine Observation"
)

plt.ylabel(
    "Euclidean Distance"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "hierarchical_unscaled_dendrogram.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PART 2: STANDARDIZED DATA
# ============================================================

print("\nSTANDARDIZED DATA")
print("-" * 60)


# ------------------------------------------------------------
# Euclidean distances
# ------------------------------------------------------------

distance_scaled = pdist(
    wine_scaled,
    metric="euclidean"
)


print(
    f"Number of pairwise distances: "
    f"{len(distance_scaled):,}"
)

print(
    f"Minimum distance: "
    f"{distance_scaled.min():.4f}"
)

print(
    f"Maximum distance: "
    f"{distance_scaled.max():.4f}"
)


# ------------------------------------------------------------
# Complete linkage
# ------------------------------------------------------------

Z_scaled = linkage(
    distance_scaled,
    method="complete"
)


# ------------------------------------------------------------
# Dendrogram: standardized
# ------------------------------------------------------------

plt.figure(
    figsize=(16, 8)
)

dendrogram(
    Z_scaled,
    no_labels=True,
    color_threshold=None
)

plt.title(
    "Hierarchical Clustering: Complete Linkage, Standardized Data"
)

plt.xlabel(
    "Wine Observation"
)

plt.ylabel(
    "Euclidean Distance"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "hierarchical_scaled_dendrogram.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PART 3: COMPARE THREE-CLUSTER SOLUTIONS
# ============================================================

# We use three clusters so that the hierarchical solution
# can be compared directly with the selected k-means solution.
# ============================================================

UNSCALED_K = 3
SCALED_K = 3


unscaled_labels = fcluster(
    Z_unscaled,
    t=UNSCALED_K,
    criterion="maxclust"
)

scaled_labels = fcluster(
    Z_scaled,
    t=SCALED_K,
    criterion="maxclust"
)


# ------------------------------------------------------------
# Save cluster assignments
# ------------------------------------------------------------

hierarchical_results = pd.DataFrame(
    {
        "Observation": np.arange(
            1,
            len(wine) + 1
        ),
        "Hierarchical_Unscaled_Cluster":
            unscaled_labels,
        "Hierarchical_Scaled_Cluster":
            scaled_labels
    }
)

hierarchical_results.to_csv(
    TABLE_DIR / "hierarchical_cluster_assignments.csv",
    index=False
)


# ------------------------------------------------------------
# Cluster sizes
# ------------------------------------------------------------

unscaled_sizes = (
    pd.Series(
        unscaled_labels
    )
    .value_counts()
    .sort_index()
)

scaled_sizes = (
    pd.Series(
        scaled_labels
    )
    .value_counts()
    .sort_index()
)


cluster_size_comparison = pd.DataFrame(
    {
        "Cluster": [1, 2, 3],
        "Unscaled_Count": [
            int(
                unscaled_sizes.get(
                    cluster,
                    0
                )
            )
            for cluster in [1, 2, 3]
        ],
        "Scaled_Count": [
            int(
                scaled_sizes.get(
                    cluster,
                    0
                )
            )
            for cluster in [1, 2, 3]
        ]
    }
)


cluster_size_comparison[
    "Unscaled_Percentage"
] = (
    cluster_size_comparison[
        "Unscaled_Count"
    ]
    / len(wine)
    * 100
)


cluster_size_comparison[
    "Scaled_Percentage"
] = (
    cluster_size_comparison[
        "Scaled_Count"
    ]
    / len(wine)
    * 100
)


cluster_size_comparison.to_csv(
    TABLE_DIR / "hierarchical_cluster_size_comparison.csv",
    index=False
)


# ------------------------------------------------------------
# Console: cluster sizes
# ------------------------------------------------------------

print("\nTHREE-CLUSTER SOLUTION")
print("=" * 60)

print(
    cluster_size_comparison.round(2).to_string(
        index=False
    )
)


# ============================================================
# PART 4: SAVE LINKAGE MATRICES
# ============================================================

pd.DataFrame(
    Z_unscaled,
    columns=[
        "Cluster_1",
        "Cluster_2",
        "Distance",
        "Cluster_Size"
    ]
).to_csv(
    TABLE_DIR / "hierarchical_linkage_unscaled.csv",
    index=False
)


pd.DataFrame(
    Z_scaled,
    columns=[
        "Cluster_1",
        "Cluster_2",
        "Distance",
        "Cluster_Size"
    ]
).to_csv(
    TABLE_DIR / "hierarchical_linkage_scaled.csv",
    index=False
)


# ------------------------------------------------------------
# Final confirmation
# ------------------------------------------------------------

print("\nHIERARCHICAL ARTIFACTS SAVED")
print("=" * 60)

print("Tables:")

for artifact in sorted(
    TABLE_DIR.glob("hierarchical_*.csv")
):

    print(
        f"  {artifact.name}"
    )

print("\nFigures:")

for artifact in sorted(
    FIGURE_DIR.glob("hierarchical_*.png")
):

    print(
        f"  {artifact.name}"
    )


print(
    "\nassignment_07_04_hierarchical.py "
    "completed successfully."
)

