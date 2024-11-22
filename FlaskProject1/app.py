from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load dataset and preprocess it
def load_data():
    dataset_path = r"C:\Users\latha\Downloads\IMDB-Movie-Dataset(2023-1951).csv"
    try:
        movies = pd.read_csv(dataset_path)
        print(f"Columns in dataset: {movies.columns}")  # Debug: print column names

        # Using 'genre' and 'movie_name' for combined features as 'rating' column does not exist
        movies['combined_features'] = movies['genre'] + " " + movies['movie_name']
        print(f"Dataset loaded successfully: {movies.head()}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    return movies


def preprocess_data(movies):
    if movies is None:
        raise ValueError("Dataset is not loaded properly.")

    if 'combined_features' not in movies.columns:
        raise ValueError("The 'combined_features' column is missing.")

    movies['combined_features'] = movies['combined_features'].fillna('')
    vectorizer = CountVectorizer(stop_words='english')
    feature_matrix = vectorizer.fit_transform(movies['combined_features'])

    try:
        similarity = cosine_similarity(feature_matrix)
        print("Similarity matrix computed successfully.")
    except Exception as e:
        print(f"Error computing similarity matrix: {e}")
        return None

    return similarity


# Function to recommend movies based on genre
def recommend_movies(genre, movies, similarity, top_n=10):
    filtered_movies = movies[movies['genre'].str.contains(genre, case=False, na=False)]

    if filtered_movies.empty:
        return "No movies found for the genre!"

    top_movies = filtered_movies.sort_values('year', ascending=False).head(top_n)
    # Ensure the key is 'title' for JavaScript to process it correctly
    return top_movies[['movie_name', 'genre']].rename(columns={'movie_name': 'title'}).to_dict(orient='records')


# Initialize Flask app
app = Flask(__name__)

# Load and preprocess the data
movies = load_data()

# Check if movies is None to prevent further errors
if movies is not None:
    similarity = preprocess_data(movies)
else:
    similarity = None


@app.route('/')
def home():
    return render_template('MathProg.html')  # Render the HTML template


@app.route('/recommend', methods=['GET'])
def recommend():
    genre = request.args.get('genre')

    if not genre:
        return jsonify({"error": "Please provide a genre"}), 400

    try:
        if similarity is None:
            return jsonify({"error": "Dataset or similarity matrix not loaded properly."}), 500

        recommendations = recommend_movies(genre, movies, similarity)

        if isinstance(recommendations, str):
            return jsonify({"error": recommendations}), 404

        return jsonify(recommendations)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
