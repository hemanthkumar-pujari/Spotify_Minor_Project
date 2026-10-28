# Part 8 - Flask application
# Run from inside the Spotify_Minor_Project folder:  python backend/app.py
# Then open http://127.0.0.1:5000 in the browser.
import os
from flask import Flask, jsonify, request, send_from_directory
from recommender import recommend_song, search_songs

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")


# Home page: sends frontend/index.html (style.css and script.js are served automatically)
@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")


# Search box: /search?q=love  ->  list of matching songs
@app.route("/search")
def search():
    query = request.args.get("q", "")
    return jsonify(search_songs(query, limit=10))


# Recommend button: /recommend?song=<song name>  ->  top 5 similar songs
@app.route("/recommend")
def recommend():
    song = request.args.get("song", "").strip()
    if song == "":
        return jsonify({"error": "Please enter a song name."}), 400
    try:
        result = recommend_song(song, n=5)
    except ValueError as error:                 # raised by recommender.py for unknown songs
        return jsonify({"error": str(error)}), 404
    return jsonify({"selected_song": song, "recommendations": result.to_dict(orient="records")})


if __name__ == "__main__":
    app.run(debug=True)
