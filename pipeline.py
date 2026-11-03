import pandas as pd
import datetime
import urllib.request, zipfile
from prefect import task, flow

@task(retries=2, retry_delay_seconds=5)
def download_data():
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    urllib.request.urlretrieve(url, "ml.zip")
    with zipfile.ZipFile("ml.zip") as z:
        z.extractall()
    movies = pd.read_csv("ml-latest-small/movies.csv")
    ratings = pd.read_csv("ml-latest-small/ratings.csv")
    return movies, ratings

@task
def clean_data(movies, ratings):
    movies = movies.copy()
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)\s*$', expand=False)
    movies['title'] = movies['title'].str.replace(r'\s*\(\d{4}\)\s*$', '', regex=True)
    movies = movies.dropna(subset=['year'])
    movies = movies[movies['genres'] != '(no genres listed)'].copy()
    movies = movies.drop_duplicates()
    movies['year'] = movies['year'].astype(int)
    return ratings.merge(movies, on='movieId')

@task
def recommend_movies(merged, favorites, n=5):
    counts = merged.groupby('movieId')['rating'].count()
    good_ids = counts[counts >= 50].index
    filtered = merged[merged['movieId'].isin(good_ids)]
    matrix = filtered.pivot_table(index='userId', columns='title', values='rating')

    rows = []
    for movie in favorites:
        if movie not in matrix.columns:
            print("Skipping:", movie)
            continue
        sim = matrix.corrwith(matrix[movie]).dropna().drop(movie, errors='ignore')
        for title, score in sim.sort_values(ascending=False).head(n).items():
            rows.append({
                'date': str(datetime.date.today()),
                'liked_movie': movie,
                'recommended': title,
                'similarity': round(score, 3)
            })
    return pd.DataFrame(rows)

@task
def save_results(df, path="daily_recommendations.csv"):
    df.to_csv(path, index=False)
    return len(df)

@flow(name="Movie Recommendation Pipeline", log_prints=True)
def movie_pipeline(favorites=["Toy Story", "Matrix, The", "Titanic"]):
    movies, ratings = download_data()
    merged = clean_data(movies, ratings)
    recs = recommend_movies(merged, favorites)
    saved = save_results(recs)
    print(f"Pipeline finished. Saved {saved} recommendations.")

if __name__ == "__main__":
    movie_pipeline()
