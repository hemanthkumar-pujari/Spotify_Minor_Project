# Part 2 - Data Preprocessing
# Run from inside the Spotify_Minor_Project folder:  python notebooks/preprocessing.py
import pandas as pd

RAW_PATH = "dataset/spotify_dataset.csv"
CLEAN_PATH = "dataset/spotify_cleaned.csv"

AUDIO_FEATURES = ["danceability", "energy", "loudness", "speechiness", "acousticness",
                  "instrumentalness", "liveness", "valence", "tempo"]

# Columns we keep (the rest, like album ids and release date, are not needed)
KEEP_COLUMNS = ["track_id", "track_name", "track_artist", "track_popularity",
                "playlist_name", "playlist_genre", "playlist_subgenre"] + AUDIO_FEATURES


def preprocess():
    # 1. Load CSV
    df = pd.read_csv(RAW_PATH)
    print("1. Loaded:", df.shape)

    # 2-3. Check and handle missing values
    print("\n2. Missing values before cleaning:")
    missing = df.isnull().sum()
    print(missing[missing > 0])
    # Only 5 rows have no song name/artist. We cannot recommend a song with no name,
    # so we drop these rows (very few, so no data is really lost).
    df = df.dropna(subset=["track_name", "track_artist"])
    print("3. After dropping rows with missing name/artist:", df.shape)

    # 4-5. Check and remove duplicates
    print("\n4. Duplicate track_id before removing:", df["track_id"].duplicated().sum())
    df = df.drop_duplicates(subset="track_id", keep="first")
    # Same song name + artist can also appear with different track_ids (e.g. album vs single)
    print("   Same track_name + artist (different id):",
          df.duplicated(subset=["track_name", "track_artist"]).sum())
    df = df.drop_duplicates(subset=["track_name", "track_artist"], keep="first")
    print("5. After removing duplicates:", df.shape)

    # 6. Check numerical data
    print("\n6. Numeric check on audio features:")
    print("   Non-numeric columns among features:",
          [c for c in AUDIO_FEATURES if not pd.api.types.is_numeric_dtype(df[c])])
    print("   Missing in audio features:", int(df[AUDIO_FEATURES].isnull().sum().sum()))
    print(df[AUDIO_FEATURES].describe().T[["min", "mean", "max"]])

    # 7-8. Select relevant columns and prepare cleaned dataset
    df = df[KEEP_COLUMNS].reset_index(drop=True)
    print("\n7. Columns kept:", list(df.columns))

    # 9. Save
    df.to_csv(CLEAN_PATH, index=False)
    print("\n9. Saved", CLEAN_PATH, "with shape", df.shape)
    print("\nGenre counts after cleaning:\n", df["playlist_genre"].value_counts())
    return df


if __name__ == "__main__":
    preprocess()
