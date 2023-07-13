import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

movies_df = pd.read_csv('movies.csv', sep='::', engine='python', header=None, names=['movieId', 'title', 'genre'], encoding='latin-1')
ratings_df = pd.read_csv('ratings.csv', sep='::', engine='python', header=None, names=['userId', 'movieId', 'rating', 'timestamp'], encoding='latin-1')

print("Cargado")

mlb = MultiLabelBinarizer()
genre_encoded = mlb.fit_transform(movies_df['genre'].str.split('|'))
genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_)
movies_df = pd.concat([movies_df, genre_df], axis=1)

print("Procesado")

features = mlb.classes_
X = movies_df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

print("Clusterizado")

merged_df = pd.merge(ratings_df, movies_df, on='movieId')
X_train, X_test, y_train, y_test = train_test_split(merged_df[features], merged_df['rating'], test_size=0.2, random_state=42)

model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', random_state=42)
model.fit(X_train, y_train)

print("Entrenado")

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Error cuadrático medio (MSE): {mse}')

def recommend_movies(genre):
    if genre in mlb.classes_:
        genre_movies = movies_df[movies_df[genre] == 1]
        user_movie_features = genre_movies[features]
        user_movie_ratings = model.predict(user_movie_features)
        recommendations = genre_movies.assign(rating=user_movie_ratings).sort_values('rating', ascending=False)

        recommended_movies = recommendations['title'].head(5).to_string(index=False)
        print(f"Las siguientes películas del género '{genre}' te podrían interesar:\n{recommended_movies}")
    else:
        print("No se encontraron películas con el género especificado.")

while True:
    genre = input("Ingresa el género de película (o escribe 'stop' para salir): ")
    if genre.lower() == 'stop':
        break
    recommend_movies(genre)
