# Spotify Songs' Genre Segmentation and Music Recommendation System

B.Tech CSE Minor Project

## About the project
This project analyses a Spotify songs dataset (32,833 rows, 23 columns), groups songs into clusters using their audio features (K-Means), compares the clusters with playlist genres and playlist names, and recommends similar songs using cosine similarity. A small Flask web app lets a user search a song and see 5 recommendations.

## Technology used
Python 3.11, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Joblib, Flask, HTML, CSS, JavaScript.

## Folder structure
```
Spotify_Minor_Project/
├── dataset/
│   ├── spotify_dataset.csv        original data
│   ├── spotify_cleaned.csv        after preprocessing (26,229 songs)
│   └── spotify_clustered.csv      cleaned data + cluster number
├── notebooks/
│   ├── spotify_analysis.ipynb     whole project in one notebook
│   └── *.py                       the same steps as separate scripts
├── visualizations/                all graphs and web page screenshots
├── model/
│   ├── kmeans_model.pkl           trained K-Means model (K = 6)
│   └── scaler.pkl                 StandardScaler used before clustering
├── backend/
│   ├── app.py                     Flask application
│   └── recommender.py             recommendation model
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── report/                        project report (Word)
├── requirements.txt
└── README.md
```

## How to run
Run every command from inside the `Spotify_Minor_Project` folder.

1. Install the libraries (once):
   `pip install -r requirements.txt`
2. Run the steps in order (or open `notebooks/spotify_analysis.ipynb` and use *Run All*):
   ```
   python notebooks/part1_dataset_inspection.py
   python notebooks/preprocessing.py
   python notebooks/analysis.py
   python notebooks/correlation.py
   python notebooks/kmeans_find_k.py
   python notebooks/kmeans_clustering.py
   python notebooks/cluster_analysis.py
   python notebooks/test_recommender.py
   ```
3. Start the web app:
   `python backend/app.py`
   then open http://127.0.0.1:5000 in the browser.

The model files in `model/` must exist before the web app starts. They are created by `kmeans_clustering.py`.

## Main results (from the actual dataset)
- Cleaning: 32,833 rows reduced to 26,229 unique songs (5 rows with missing names removed, duplicates removed).
- Strongest feature correlations: energy and loudness (+0.69), energy and acousticness (-0.55).
- K-Means: K = 6 chosen using the Elbow method and Silhouette score (0.1575 at K = 6). The scores are low, which means the clusters overlap.
- One cluster with very high instrumentalness is 56.3% EDM, and one with very high speechiness is 57.5% rap. Other clusters mix several genres.
- Recommender: in a test on 300 random songs, 36.7% of recommended songs had the same genre as the selected song, compared with 16.9% by random chance.

## Limitations
- Only 9 audio features are used. Lyrics, language and artist are not considered.
- Each song keeps only the first playlist it appeared in, because duplicates were removed.
- Different versions of the same song (for example remasters) can still appear as recommendations.
- The saved `.pkl` files depend on the scikit-learn version in `requirements.txt`. If you see a version warning, re-run `python notebooks/kmeans_clustering.py`.
