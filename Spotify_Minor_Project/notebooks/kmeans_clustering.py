# Part 5b - Final K-Means model, cluster plot, save model
# Run from inside the Spotify_Minor_Project folder:  python notebooks/kmeans_clustering.py
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

df = pd.read_csv("dataset/spotify_cleaned.csv")
FEATURES = ["danceability", "energy", "loudness", "speechiness", "acousticness",
            "instrumentalness", "liveness", "valence", "tempo"]

# K chosen from kmeans_find_k.py results (see explanation):
# among K = 4..10, K = 6 has the best silhouette score (0.1575) and the elbow curve starts to flatten after it.
K = 6

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[FEATURES])

kmeans = KMeans(n_clusters=K, n_init=10, random_state=42)
df["cluster"] = kmeans.fit_predict(X_scaled)
print("Songs per cluster:\n", df["cluster"].value_counts().sort_index())

# Save model and scaler (the scaler is needed to prepare new songs the same way)
joblib.dump(kmeans, "model/kmeans_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
print("\nSaved model/kmeans_model.pkl and model/scaler.pkl")

# Cluster visualization: squeeze 9 features into 2 dimensions with PCA (only for drawing)
pca = PCA(n_components=2, random_state=42)
points = pca.fit_transform(X_scaled)
print("\nVariance explained by the 2 plot axes:", (pca.explained_variance_ratio_.sum() * 100).round(1), "%")

plot_df = pd.DataFrame({"PC1": points[:, 0], "PC2": points[:, 1], "cluster": df["cluster"]}).sample(8000, random_state=42)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="cluster", palette="tab10", s=12, alpha=0.6)
plt.title("K-Means Clusters (K=6) shown in 2D using PCA")
plt.xlabel("PCA component 1"); plt.ylabel("PCA component 2")
plt.tight_layout(); plt.savefig("visualizations/cluster_visualization.png", dpi=150); plt.close()
