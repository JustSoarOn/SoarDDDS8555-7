# ============================================================
# Assignment 7: PCA and Cluster Analysis of Wine Data
#
# assignment_07_05_cluster_profiles.py
#
# Purpose:
#   Describe the original wine characteristics of the final
#   k-means clusters and compare the clustering methods.
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

TABLE_DIR = OUTPUT_DIR / "tables"

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

wine = pd.read_csv(
    DATA_DIR / "wine-clustering.csv"
)

pca_data = pd.read_csv(
    TABLE_DIR / "pca_scores_retained.csv"
)

hierarchical = pd.read_csv(
    TABLE_DIR / "hierarchical_cluster_assignments.csv"
)


# ============================================================
# FINAL K-MEANS CLUSTERING
# ============================================================

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=20
)

kmeans_labels = (
    kmeans.fit_predict(
        pca_data
    )
    + 1
)


# ------------------------------------------------------------
# Add k-means clusters to original data
# ------------------------------------------------------------

wine_kmeans = wine.copy()

wine_kmeans[
    "KMeans_Cluster"
] = kmeans_labels


# ------------------------------------------------------------
# Original-variable cluster means
# ------------------------------------------------------------

cluster_means = (
    wine_kmeans
    .groupby(
        "KMeans_Cluster"
    )[wine.columns]
    .mean()
)

cluster_means.to_csv(
    TABLE_DIR / "kmeans_cluster_means_original_variables.csv"
)


# ------------------------------------------------------------
# Overall means
# ------------------------------------------------------------

overall_means = (
    wine.mean()
)


# ------------------------------------------------------------
# Difference from overall mean
# ------------------------------------------------------------

cluster_mean_difference = (
    cluster_means
    - overall_means
)

cluster_mean_difference.to_csv(
    TABLE_DIR / "kmeans_cluster_mean_difference.csv"
)


# ------------------------------------------------------------
# Standardized cluster profiles
#
# This expresses each cluster mean relative to the overall
# sample mean in standard-deviation units.
# ------------------------------------------------------------

cluster_profiles_z = (
    cluster_mean_difference
    / wine.std()
)

cluster_profiles_z.to_csv(
    TABLE_DIR / "kmeans_cluster_profiles_z.csv"
)


# ============================================================
# PCA CLUSTER PROFILES
# ============================================================

pca_cluster_means = (
    pca_data.assign(
        KMeans_Cluster=kmeans_labels
    )
    .groupby(
        "KMeans_Cluster"
    )
    .mean()
)

pca_cluster_means.to_csv(
    TABLE_DIR / "kmeans_cluster_means_pca.csv"
)


# ============================================================
# COMPARE K-MEANS WITH HIERARCHICAL CLUSTERING
# ============================================================

scaled_hierarchical = (
    hierarchical[
        "Hierarchical_Scaled_Cluster"
    ]
    .values
)


unscaled_hierarchical = (
    hierarchical[
        "Hierarchical_Unscaled_Cluster"
    ]
    .values
)


# ------------------------------------------------------------
# Adjusted Rand Index
#
# ARI accounts for arbitrary cluster labels.
# Values near 1 indicate strong agreement; values near 0
# indicate agreement comparable to random assignment.
# ------------------------------------------------------------

ari_scaled = adjusted_rand_score(
    kmeans_labels,
    scaled_hierarchical
)

ari_unscaled = adjusted_rand_score(
    kmeans_labels,
    unscaled_hierarchical
)


method_comparison = pd.DataFrame(
    {
        "Comparison": [
            "K-means vs standardized hierarchical",
            "K-means vs unscaled hierarchical"
        ],
        "Adjusted_Rand_Index": [
            ari_scaled,
            ari_unscaled
        ]
    }
)

method_comparison.to_csv(
    TABLE_DIR / "clustering_method_comparison.csv",
    index=False
)


# ============================================================
# CONSOLE OUTPUT
# ============================================================

print("K-MEANS CLUSTER PROFILES")
print("=" * 60)

print(
    cluster_means.round(3).to_string()
)


print("\nSTANDARDIZED CLUSTER PROFILES")
print("=" * 60)

print(
    cluster_profiles_z.round(3).to_string()
)


print("\nPCA CLUSTER PROFILES")
print("=" * 60)

print(
    pca_cluster_means.round(3).to_string()
)


print("\nCLUSTERING METHOD COMPARISON")
print("=" * 60)

print(
    method_comparison.round(4).to_string(
        index=False
    )
)


# ------------------------------------------------------------
# Final confirmation
# ------------------------------------------------------------

print(
    "\nassignment_07_05_cluster_profiles.py "
    "completed successfully."
)
