# Part 3 - Data Analysis and Visualization
# Run from inside the Spotify_Minor_Project folder:  python notebooks/analysis.py
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # save images without opening windows
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
df = pd.read_csv("dataset/spotify_cleaned.csv")
OUT = "visualizations/"


def save(name):
    plt.tight_layout()
    plt.savefig(OUT + name, dpi=150)
    plt.close()


# 1. Playlist genre distribution
order = df["playlist_genre"].value_counts().index
plt.figure(figsize=(7, 4))
sns.countplot(data=df, x="playlist_genre", order=order, color="steelblue")
plt.title("Playlist Genre Distribution"); plt.xlabel("Playlist genre"); plt.ylabel("Number of songs")
save("01_genre_distribution.png")

# 2. Playlist subgenre distribution
order = df["playlist_subgenre"].value_counts().index
plt.figure(figsize=(8, 7))
sns.countplot(data=df, y="playlist_subgenre", order=order, color="steelblue")
plt.title("Playlist Subgenre Distribution"); plt.xlabel("Number of songs"); plt.ylabel("Playlist subgenre")
save("02_subgenre_distribution.png")

# 3-8. Histograms of single features
hist_plots = [("track_popularity", "03_popularity_distribution.png"),
              ("danceability", "04_danceability_distribution.png"),
              ("energy", "05_energy_distribution.png"),
              ("loudness", "06_loudness_distribution.png"),
              ("acousticness", "07_acousticness_distribution.png"),
              ("valence", "08_valence_distribution.png")]
for col, filename in hist_plots:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], bins=30, color="steelblue")
    plt.title(f"{col.replace('_', ' ').title()} Distribution"); plt.xlabel(col); plt.ylabel("Number of songs")
    save(filename)

# 9. Energy vs Danceability (sample of 5000 songs so the plot is not too crowded)
sample = df.sample(5000, random_state=42)
plt.figure(figsize=(6, 5))
sns.scatterplot(data=sample, x="energy", y="danceability", alpha=0.4, s=15)
plt.title("Energy vs Danceability (5000 random songs)")
save("09_energy_vs_danceability.png")

# 10. Genre vs Energy
plt.figure(figsize=(7, 4))
sns.boxplot(data=df, x="playlist_genre", y="energy", hue="playlist_genre", legend=False)
plt.title("Playlist Genre vs Energy"); plt.xlabel("Playlist genre")
save("10_genre_vs_energy.png")

# 11. Genre vs Danceability
plt.figure(figsize=(7, 4))
sns.boxplot(data=df, x="playlist_genre", y="danceability", hue="playlist_genre", legend=False)
plt.title("Playlist Genre vs Danceability"); plt.xlabel("Playlist genre")
save("11_genre_vs_danceability.png")

# ---- Numbers used for the observations ----
print("Songs:", len(df))
print("\nGenre counts:\n", df["playlist_genre"].value_counts())
print("\nTop 3 / bottom 3 subgenres:\n", df["playlist_subgenre"].value_counts().iloc[[0, 1, 2, -3, -2, -1]])
print("\nMean / median of features:")
print(df[["track_popularity", "danceability", "energy", "loudness", "acousticness", "valence"]].agg(["mean", "median"]).T.round(3))
print("\nSongs with popularity 0:", (df["track_popularity"] == 0).sum())
print("Songs with acousticness < 0.2:", round((df["acousticness"] < 0.2).mean() * 100, 1), "%")
print("Songs with loudness > -10:", round((df["loudness"] > -10).mean() * 100, 1), "%")
print("Energy vs danceability correlation:", round(df["energy"].corr(df["danceability"]), 3))
print("\nMedian energy by genre:\n", df.groupby("playlist_genre")["energy"].median().sort_values().round(3))
print("\nMedian danceability by genre:\n", df.groupby("playlist_genre")["danceability"].median().sort_values().round(3))
print("\nZero-value check -> tempo==0:", (df["tempo"] == 0).sum(), "| danceability==0:", (df["danceability"] == 0).sum())
