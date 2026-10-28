# Part 5a - Find the best K (Elbow + Silhouette)
# Run from inside the Spotify_Minor_Project folder:  python notebooks/kmeans_find_k.py
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("dataset/spotify_cleaned.csv")
FEATURES = ["danceability", "energy", "loudness", "speechiness", "acousticness",
            "instrumentalness", "liveness", "valence", "tempo"]

# Standardize: every feature gets mean 0 and standard deviation 1
X_scaled = StandardScaler().fit_transform(df[FEATURES])

k_values = list(range(2, 11))
inertia_list, silhouette_list = [], []
for k in k_values:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
    inertia_list.append(km.inertia_)
    # silhouette on 10,000 random songs to keep it fast
    sil = silhouette_score(X_scaled, km.labels_, sample_size=10000, random_state=42)
    silhouette_list.append(sil)
    print(f"K={k:2d}  inertia={km.inertia_:10.1f}  silhouette={sil:.4f}")

# Elbow graph
plt.figure(figsize=(6, 4))
plt.plot(k_values, inertia_list, marker="o")
plt.title("Elbow Method"); plt.xlabel("Number of clusters (K)"); plt.ylabel("Inertia (within-cluster distance)")
plt.xticks(k_values); plt.grid(True)
plt.tight_layout(); plt.savefig("visualizations/elbow_graph.png", dpi=150); plt.close()

# Silhouette graph
plt.figure(figsize=(6, 4))
plt.plot(k_values, silhouette_list, marker="o", color="green")
plt.title("Silhouette Score for Different K"); plt.xlabel("Number of clusters (K)"); plt.ylabel("Silhouette score")
plt.xticks(k_values); plt.grid(True)
plt.tight_layout(); plt.savefig("visualizations/silhouette_graph.png", dpi=150); plt.close()
