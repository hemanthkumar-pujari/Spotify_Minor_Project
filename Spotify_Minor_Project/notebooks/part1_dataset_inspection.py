# Part 1 - Dataset Inspection
# Run from inside the Spotify_Minor_Project folder:  python notebooks/part1_dataset_inspection.py
import pandas as pd

df = pd.read_csv("dataset/spotify_dataset.csv")

print("1. Rows    :", df.shape[0])
print("2. Columns :", df.shape[1])
print("\n3. Column names:\n", list(df.columns))
print("\n4. Data types:\n", df.dtypes)

print("\n5. Missing values (only columns that have any):")
missing = df.isnull().sum()
print(missing[missing > 0])

print("\n6. Duplicate rows (full row identical):", df.duplicated().sum())
print("   Duplicate track_id values            :", df["track_id"].duplicated().sum())

numerical = df.select_dtypes(include="number").columns.tolist()
categorical = df.select_dtypes(exclude="number").columns.tolist()
print("\n7. Numerical columns  :", numerical)
print("8. Categorical columns:", categorical)

audio_features = ["danceability", "energy", "loudness", "speechiness", "acousticness",
                  "instrumentalness", "liveness", "valence", "tempo"]
print("\n9. Spotify audio features used:", audio_features)
print(df[audio_features].describe().T[["min", "mean", "max"]])

print("\n10. playlist_genre:", df["playlist_genre"].nunique(), "unique values")
print(df["playlist_genre"].value_counts())

print("\n11. playlist_name:", df["playlist_name"].nunique(), "unique values")
print(df["playlist_name"].value_counts().head(5))
