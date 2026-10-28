# Part 6 - Cluster Analysis
# Run from inside the Spotify_Minor_Project folder:  python notebooks/cluster_analysis.py
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.width", 200)
df = pd.read_csv("dataset/spotify_cleaned.csv")
FEATURES = ["danceability", "energy", "loudness", "speechiness", "acousticness",
            "instrumentalness", "liveness", "valence", "tempo"]

# 1. Add cluster number to each song (using the saved scaler and model from Part 5)
scaler = joblib.load("model/scaler.pkl")
kmeans = joblib.load("model/kmeans_model.pkl")
df["cluster"] = kmeans.predict(scaler.transform(df[FEATURES]))
df.to_csv("dataset/spotify_clustered.csv", index=False)

# 2. Count songs in each cluster
print("Songs per cluster:\n", df["cluster"].value_counts().sort_index())

# 3. Average audio features per cluster (real units) + overall average for comparison
profile = df.groupby("cluster")[FEATURES].mean()
print("\nAverage features per cluster:\n", profile.round(3))
print("\nOverall average:\n", df[FEATURES].mean().round(3).to_dict())

# Same table as 'how many standard deviations above/below the overall average' (easier to compare)
z_profile = (profile - df[FEATURES].mean()) / df[FEATURES].std()
print("\nCluster profile in standard deviations (+ = above average, - = below):\n", z_profile.round(2))

plt.figure(figsize=(9, 5))
sns.heatmap(z_profile, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Cluster Profiles (standard deviations from overall average)")
plt.xlabel("Audio feature"); plt.ylabel("Cluster")
plt.tight_layout(); plt.savefig("visualizations/cluster_profiles.png", dpi=150); plt.close()

# 5. Cluster vs Playlist Genre (each row = one cluster, values = % of that cluster's songs)
genre_pct = pd.crosstab(df["cluster"], df["playlist_genre"], normalize="index") * 100
print("\nCluster vs Genre (% of each cluster):\n", genre_pct.round(1))
print("\nOverall genre share (%):\n", (df["playlist_genre"].value_counts(normalize=True) * 100).round(1).to_dict())

plt.figure(figsize=(8, 5))
sns.heatmap(genre_pct, annot=True, fmt=".1f", cmap="YlGnBu")
plt.title("Cluster vs Playlist Genre (% of songs in each cluster)")
plt.xlabel("Playlist genre"); plt.ylabel("Cluster")
plt.tight_layout(); plt.savefig("visualizations/cluster_vs_genre.png", dpi=150); plt.close()

# 6. Cluster vs Playlist Name (449 names, so we use the 10 playlists with the most songs)
top_names = df["playlist_name"].value_counts().head(10).index
name_pct = pd.crosstab(df[df["playlist_name"].isin(top_names)]["playlist_name"],
                       df[df["playlist_name"].isin(top_names)]["cluster"], normalize="index") * 100
name_pct = name_pct.loc[top_names]
print("\nCluster vs top-10 Playlist Names (% of each playlist's songs):\n", name_pct.round(1))

plt.figure(figsize=(9, 6))
sns.heatmap(name_pct, annot=True, fmt=".1f", cmap="YlGnBu")
plt.title("Top 10 Playlists vs Cluster (% of playlist's songs)")
plt.xlabel("Cluster"); plt.ylabel("Playlist name")
plt.tight_layout(); plt.savefig("visualizations/cluster_vs_playlist_name.png", dpi=150); plt.close()

# Extra: top 3 playlist names inside each cluster
print("\nMost common playlist names inside each cluster:")
for c in sorted(df["cluster"].unique()):
    print(c, df[df["cluster"] == c]["playlist_name"].value_counts().head(3).to_dict())
