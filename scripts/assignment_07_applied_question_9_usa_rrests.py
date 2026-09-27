# ============================================================
# Assignment 7
#
# assignment_07_applied_question_9_usa_rrests.py
#
# Purpose:
#   Applied Question #9 from ISLR Python.
#
#   (a) Complete-linkage hierarchical clustering using
#       Euclidean distance.
#   (b) Three-cluster solution.
#   (c) Repeat after standardizing variables.
#   (d) Compare the effect of scaling.
# ============================================================

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import (
    linkage,
    dendrogram,
    fcluster
)

from scipy.spatial.distance import pdist

from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------
# Paths
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
# Load data
# ------------------------------------------------------------

usarrests = pd.read_csv(
    DATA_DIR / "USArrests.csv",
    index_col=0
)

print("USARRESTS HIERARCHICAL CLUSTERING")
print("=" * 60)

print(
    f"States: {usarrests.shape[0]}"
)

print(
    f"Variables: {usarrests.shape[1]}"
)

print(
    "\nVariables:"
)

print(
    usarrests.columns.tolist()
)


# ============================================================
# PART A: UNSCALED
# ============================================================

print("\nUNSCALED COMPLETE-LINKAGE CLUSTERING")
print("-" * 60)

distances_unscaled = pdist(
    usarrests,
    metric="euclidean"
)

hc_unscaled = linkage(
    distances_unscaled,
    method="complete"
)


# ------------------------------------------------------------
# Dendrogram
# ------------------------------------------------------------

plt.figure(
    figsize=(16, 9)
)

dendrogram(
    hc_unscaled,
    labels=usarrests.index,
    leaf_rotation=90
)

plt.title(
    "USArrests: Complete Linkage, Unscaled Variables"
)

plt.xlabel("State")

plt.ylabel(
    "Euclidean Distance"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "usarrests_hierarchical_unscaled.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PART B: THREE UNSCALED CLUSTERS
# ============================================================

clusters_unscaled = fcluster(
    hc_unscaled,
    t=3,
    criterion="maxclust"
)

unscaled_results = pd.DataFrame(
    {
        "State": usarrests.index,
        "Cluster": clusters_unscaled
    }
)

unscaled_results.to_csv(
    TABLE_DIR / "usarrests_unscaled_clusters.csv",
    index=False
)


print("\nTHREE UNSCALED CLUSTERS")
print("-" * 60)

for cluster in sorted(
    unscaled_results["Cluster"].unique()
):

    states = (
        unscaled_results.loc[
            unscaled_results["Cluster"] == cluster,
            "State"
        ]
        .tolist()
    )

    print(
        f"Cluster {cluster} "
        f"({len(states)} states):"
    )

    print(
        ", ".join(states)
    )


# ============================================================
# PART C: STANDARDIZED
# ============================================================

print("\nSTANDARDIZED COMPLETE-LINKAGE CLUSTERING")
print("-" * 60)

scaler = StandardScaler()

usarrests_scaled = pd.DataFrame(
    scaler.fit_transform(usarrests),
    index=usarrests.index,
    columns=usarrests.columns
)

distances_scaled = pdist(
    usarrests_scaled,
    metric="euclidean"
)

hc_scaled = linkage(
    distances_scaled,
    method="complete"
)


# ------------------------------------------------------------
# Standardized dendrogram
# ------------------------------------------------------------

plt.figure(
    figsize=(16, 9)
)

dendrogram(
    hc_scaled,
    labels=usarrests.index,
    leaf_rotation=90
)

plt.title(
    "USArrests: Complete Linkage, Standardized Variables"
)

plt.xlabel("State")

plt.ylabel(
    "Euclidean Distance"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "usarrests_hierarchical_scaled.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# THREE STANDARDIZED CLUSTERS
# ============================================================

clusters_scaled = fcluster(
    hc_scaled,
    t=3,
    criterion="maxclust"
)

scaled_results = pd.DataFrame(
    {
        "State": usarrests.index,
        "Cluster": clusters_scaled
    }
)

scaled_results.to_csv(
    TABLE_DIR / "usarrests_scaled_clusters.csv",
    index=False
)


print("\nTHREE STANDARDIZED CLUSTERS")
print("-" * 60)

for cluster in sorted(
    scaled_results["Cluster"].unique()
):

    states = (
        scaled_results.loc[
            scaled_results["Cluster"] == cluster,
            "State"
        ]
        .tolist()
    )

    print(
        f"Cluster {cluster} "
        f"({len(states)} states):"
    )

    print(
        ", ".join(states)
    )


# ============================================================
# CLUSTER SIZE COMPARISON
# ============================================================

unscaled_sizes = (
    unscaled_results[
        "Cluster"
    ]
    .value_counts()
    .sort_index()
)

scaled_sizes = (
    scaled_results[
        "Cluster"
    ]
    .value_counts()
    .sort_index()
)

size_comparison = pd.DataFrame(
    {
        "Cluster": [1, 2, 3],
        "Unscaled_Count": [
            unscaled_sizes.get(
                i,
                0
            )
            for i in [1, 2, 3]
        ],
        "Scaled_Count": [
            scaled_sizes.get(
                i,
                0
            )
            for i in [1, 2, 3]
        ]
    }
)

size_comparison.to_csv(
    TABLE_DIR / "usarrests_cluster_size_comparison.csv",
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\nUSARRESTS ARTIFACTS SAVED")
print("=" * 60)

print("Tables:")

for artifact in sorted(
    TABLE_DIR.glob(
        "usarrests_*.csv"
    )
):

    print(
        f"  {artifact.name}"
    )

print("\nFigures:")

for artifact in sorted(
    FIGURE_DIR.glob(
        "usarrests_*.png"
    )
):

    print(
        f"  {artifact.name}"
    )


print(
    "\nassignment_07_applied_question_9_usa_rrests.py "
    "completed successfully."
)

