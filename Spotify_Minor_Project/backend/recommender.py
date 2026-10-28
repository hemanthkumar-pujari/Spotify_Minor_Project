# Part 7 - Content-based Music Recommender
# Idea: songs with similar audio features sound similar, so we recommend the songs
# whose audio-feature "profile" is closest (cosine similarity) to the selected song.
import os
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Paths are built from this file's location, so it works from any folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "spotify_cleaned.csv")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")

FEATURES = ["danceability", "energy", "loudness", "speechiness", "acousticness",
            "instrumentalness", "liveness", "valence", "tempo"]

# Load everything once, when the file is first imported
df = pd.read_csv(DATA_PATH)
scaler = joblib.load(SCALER_PATH)                 # same scaler used for K-Means
X = scaler.transform(df[FEATURES])                # scaled audio features of every song
df["name_lower"] = df["track_name"].str.lower().str.strip()


def find_song(song_name):
    """Return the row index of the song, or None if it is not in the dataset."""
    name = song_name.lower().strip()
    matches = df[df["name_lower"] == name]
    if matches.empty:                              # no exact match -> try 'contains'
        matches = df[df["name_lower"].str.contains(name, regex=False)]
    if matches.empty:
        return None
    # if several songs share the name (different artists), take the most popular one
    return matches["track_popularity"].idxmax()


def search_songs(query, limit=10):
    """Songs whose name contains the text typed by the user (used by the search box)."""
    query = query.lower().strip()
    if query == "":
        return []
    matches = df[df["name_lower"].str.contains(query, regex=False)]
    matches = matches.sort_values("track_popularity", ascending=False).head(limit)
    return [{"song": r.track_name, "artist": r.track_artist} for r in matches.itertuples()]


def recommend_song(song_name, n=5):
    """Return a table of the n most similar songs. Raises ValueError for an unknown song."""
    idx = find_song(song_name)
    if idx is None:
        raise ValueError(f"Song '{song_name}' was not found in the dataset. Please try another name.")

    # cosine similarity between the selected song and every song
    scores = cosine_similarity(X[idx].reshape(1, -1), X)[0]

    result = df.copy()
    result["Similarity Score"] = scores
    result = result.drop(index=idx)                # do not recommend the song itself
    result = result.sort_values("Similarity Score", ascending=False).head(n)

    result = result.rename(columns={"track_name": "Song Name", "track_artist": "Artist",
                                    "playlist_genre": "Genre"})
    result["Similarity Score"] = result["Similarity Score"].round(4)
    return result[["Song Name", "Artist", "Genre", "Similarity Score"]].reset_index(drop=True)


if __name__ == "__main__":
    # quick test: pick a real song from the dataset and show its recommendations
    sample_song = df.sample(1, random_state=7).iloc[0]
    print("Selected song:", sample_song["track_name"], "-", sample_song["track_artist"],
          "| genre:", sample_song["playlist_genre"], "\n")
    print(recommend_song(sample_song["track_name"], n=5))
    print("\nInvalid name test:")
    try:
        recommend_song("this song does not exist 12345")
    except ValueError as error:
        print(error)
