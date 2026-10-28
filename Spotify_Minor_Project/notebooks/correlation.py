# Part 4 - Correlation Matrix
# Run from inside the Spotify_Minor_Project folder:  python notebooks/correlation.py
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("dataset/spotify_cleaned.csv")

AUDIO_FEATURES = ["danceability", "energy", "loudness", "speechiness", "acousticness",
                  "instrumentalness", "liveness", "valence", "tempo"]

corr = df[AUDIO_FEATURES].corr()

plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True)
plt.title("Correlation Matrix of Spotify Audio Features")
plt.tight_layout()
plt.savefig("visualizations/correlation_matrix.png", dpi=150)
plt.close()

# List every pair of features once, then sort by correlation value
pair_list = []
for i in range(len(AUDIO_FEATURES)):
    for j in range(i + 1, len(AUDIO_FEATURES)):
        a, b = AUDIO_FEATURES[i], AUDIO_FEATURES[j]
        pair_list.append((a + " vs " + b, corr.loc[a, b]))
pairs = pd.Series(dict(pair_list)).sort_values()

print("Strongest NEGATIVE pairs:\n", pairs.head(4).round(3))
print("\nStrongest POSITIVE pairs:\n", pairs.tail(4).round(3))
print("\nWeakest pairs (closest to 0):\n", pairs.abs().sort_values().head(4).round(3))
print("\nPairs with |corr| >= 0.5:", int((pairs.abs() >= 0.5).sum()), "of", len(pairs))
