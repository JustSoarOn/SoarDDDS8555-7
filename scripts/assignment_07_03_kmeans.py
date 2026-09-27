# ============================================================
# Assignment 7: PCA and Cluster Analysis of Wine Data
#
# assignment_07_03_kmeans.py
#
# Purpose:
#   Evaluate k-means clustering using the five PCA components
#   that explain at least 80% of the standardized wine data
#   variance.
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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
# Load retained PCA scores
# ------------------------------------------------------------

pca_data = pd.read_csv(
    TABLE_DIR / "pca_scores_retained.csv"
)


# ------------------------------------------------------------
# Determine number of PCA components
# ------------------------------------------------------------

n_components = pca_data.shape[1]

print("K-MEANS CLUSTERING")
print("=" * 60)

print(
    f"Observations: {pca_data.shape[0]}"
)

print(
    f"PCA components used: {n_components}"
)


# ------------------------------------------------------------
# Evaluate k = 2 through 10
# ------------------------------------------------------------

k_values = range(2, 11)

inertia_values = []

silhouette_values = []

cluster_results = []


for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    labels = kmeans.fit_predict(
        pca_data
    )

    inertia = kmeans.inertia_

    silhouette = silhouette_score(
        pca_data,
        labels
    )

    inertia_values.append(
        inertia
    )

    silhouette_values.append(
        silhouette
    )

    cluster_results.append(
        {
            "k": k,
            "Inertia": inertia,
            "Silhouette_Score": silhouette
        }
    )


# ------------------------------------------------------------
# Results table
# ------------------------------------------------------------

k_results = pd.DataFrame(
    cluster_results
)

k_results.to_csv(
    TABLE_DIR / "kmeans_k_evaluation.csv",
    index=False
)


print("\nK-MEANS EVALUATION")
print("=" * 60)

print(
    k_results.round(4).to_string(
        index=False
    )
)


# ------------------------------------------------------------
# Identify highest silhouette score
# ------------------------------------------------------------

best_silhouette_row = (
    k_results.loc[
        k_results["Silhouette_Score"].idxmax()
    ]
)

best_silhouette_k = int(
    best_silhouette_row["k"]
)

best_silhouette = (
    best_silhouette_row[
        "Silhouette_Score"
    ]
)


print(
    f"\nHighest silhouette score:"
)

print(
    f"k = {best_silhouette_k}"
)

print(
    f"Silhouette score = "
    f"{best_silhouette:.4f}"
)


# ------------------------------------------------------------
# Elbow plot
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    k_values,
    inertia_values,
    marker="o",
    linewidth=2
)

plt.xlabel(
    "Number of Clusters (k)"
)

plt.ylabel(
    "Within-Cluster Sum of Squares (Inertia)"
)

plt.title(
    "Elbow Method for K-Means Clustering"
)

plt.xticks(
    list(k_values)
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "kmeans_elbow_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# Silhouette plot
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    k_values,
    silhouette_values,
    marker="o",
    linewidth=2,
    color="darkgreen"
)

plt.xlabel(
    "Number of Clusters (k)"
)

plt.ylabel(
    "Average Silhouette Score"
)

plt.title(
    "Silhouette Scores for K-Means Clustering"
)

plt.xticks(
    list(k_values)
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "kmeans_silhouette_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# Fit candidate solutions
# ------------------------------------------------------------

candidate_ks = [3, 4, 5]

cluster_summary = []


for k in candidate_ks:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    labels = kmeans.fit_predict(
        pca_data
    )

    counts = (
        pd.Series(labels + 1)
        .value_counts()
        .sort_index()
    )

    row = {
        "k": k,
        "Silhouette_Score":
            silhouette_score(
                pca_data,
                labels
            )
    }

    for cluster_number, count in counts.items():

        row[
            f"Cluster_{cluster_number}_Size"
        ] = int(count)

    cluster_summary.append(
        row
    )


cluster_summary = pd.DataFrame(
    cluster_summary
)

cluster_summary.to_csv(
    TABLE_DIR / "kmeans_candidate_cluster_sizes.csv",
    index=False
)


# ------------------------------------------------------------
# Final confirmation
# ------------------------------------------------------------

print("\nK-MEANS ARTIFACTS SAVED")
print("=" * 60)

print("Tables:")

for artifact in sorted(
    TABLE_DIR.glob("kmeans_*.csv")
):

    print(
        f"  {artifact.name}"
    )

print("\nFigures:")

for artifact in sorted(
    FIGURE_DIR.glob("kmeans_*.png")
):

    print(
        f"  {artifact.name}"
    )


# ------------------------------------------------------------
# Final k-means model: k = 3
# ------------------------------------------------------------

FINAL_K = 3

final_kmeans = KMeans(
    n_clusters=FINAL_K,
    random_state=42,
    n_init=20
)

final_labels = final_kmeans.fit_predict(
    pca_data
)

final_clustered_data = pca_data.copy()

final_clustered_data[
    "KMeans_Cluster"
] = final_labels + 1


# ------------------------------------------------------------
# Save final cluster assignments
# ------------------------------------------------------------

final_clustered_data.to_csv(
    TABLE_DIR / "kmeans_final_cluster_assignments.csv",
    index=False
)


# ------------------------------------------------------------
# Final cluster sizes
# ------------------------------------------------------------

final_cluster_sizes = (
    final_clustered_data[
        "KMeans_Cluster"
    ]
    .value_counts()
    .sort_index()
    .rename("Count")
    .reset_index()
)

final_cluster_sizes.columns = [
    "Cluster",
    "Count"
]

final_cluster_sizes[
    "Percentage"
] = (
    final_cluster_sizes["Count"]
    / len(final_clustered_data)
    * 100
)

final_cluster_sizes.to_csv(
    TABLE_DIR / "kmeans_final_cluster_sizes.csv",
    index=False
)


# ------------------------------------------------------------
# Visualize final clusters using PC1 and PC2
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 7)
)

sns.scatterplot(
    data=final_clustered_data,
    x="PC1",
    y="PC2",
    hue="KMeans_Cluster",
    palette="Set1",
    s=70
)

plt.xlabel(
    "Principal Component 1"
)

plt.ylabel(
    "Principal Component 2"
)

plt.title(
    "K-Means Clustering of Wine Observations (k = 3)"
)

plt.legend(
    title="Cluster"
)

plt.grid(
    alpha=0.2
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "kmeans_final_clusters_pc1_pc2.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "\nassignment_07_03_kmeans.py "
    "completed successfully."
)
