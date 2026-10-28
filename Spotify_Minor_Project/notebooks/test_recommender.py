# Part 7 - Test and evaluate the recommender
# Run from inside the Spotify_Minor_Project folder:  python notebooks/test_recommender.py
import sys
sys.path.append("backend")
import pandas as pd
from recommender import df, recommend_song, search_songs, find_song

pd.set_option("display.width", 200)

# 1. Show recommendations for the most popular song of 3 different genres (taken from the data, not typed by hand)
for genre in ["rock", "rap", "latin"]:
    song = df[df["playlist_genre"] == genre].sort_values("track_popularity", ascending=False).iloc[0]
    print(f"\n=== Selected: {song['track_name']} - {song['track_artist']} (genre: {genre}) ===")
    print(recommend_song(song["track_name"], n=5).to_string())

# 2. Upper/lower case and partial names still work, invalid names give a clear message
song_name = df.iloc[0]["track_name"]
found = df.loc[find_song(song_name.upper()), "track_name"]
print(f"\nTyped '{song_name.upper()}' -> matched song: '{found}'")
print("search_songs('love') ->", search_songs("love", limit=3))
try:
    recommend_song("zzzz not a song")
except ValueError as e:
    print("Invalid name ->", e)

# 3. Simple check: do recommended songs share the genre of the selected song?
test_songs = df.sample(300, random_state=42)
same_genre = 0
total = 0
for idx, row in test_songs.iterrows():
    recs = recommend_song(row["track_name"], n=5)
    same_genre += (recs["Genre"] == row["playlist_genre"]).sum()
    total += len(recs)
print(f"\nGenre check on 300 random songs: {same_genre / total * 100:.1f}% of recommended songs have the same genre as the selected song")

# Baseline: chance of matching genre if we picked songs completely at random
share = df["playlist_genre"].value_counts(normalize=True)
print(f"Random-pick baseline: {(share ** 2).sum() * 100:.1f}%")
