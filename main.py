import random
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from datetime import datetime
import matplotlib.pyplot as plt

movies_df = pd.read_csv('movies.csv', sep='::', engine='python', header=None, names=['movieId', 'title', 'genre'], encoding='latin-1')
ratings_df = pd.read_csv('ratings.csv', sep='::', engine='python', header=None, names=['userId', 'movieId', 'rating', 'timestamp'], encoding='latin-1')
desired_rows = 100000
num_rows_original = ratings_df.shape[0]
if desired_rows<num_rows_original:

    rows_to_keep = random.sample(range(num_rows_original), desired_rows)

    df_reduced = ratings_df.iloc[rows_to_keep]

    # Guarda el DataFrame reducido en un nuevo archivo CSV
    df_reduced.to_csv("ratingedit.csv", index=False)


print("Cargado")

mlb = MultiLabelBinarizer()
genre_encoded = mlb.fit_transform(movies_df['genre'].str.split('|'))

genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_)

movies_df = pd.concat([movies_df, genre_df], axis=1)

print("Preprocesado")

features = mlb.classes_
X = movies_df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

dbscan = DBSCAN(eps=0.5, min_samples=5)
clusterspredi = dbscan.fit_predict(X_scaled)

#Contador de clusters

num_clusters = len(set(clusterspredi)) - (1 if -1 in clusterspredi else 0)
print(f"Número de clusters creados: {num_clusters}")

print("Clusterizado")

ClusterInicial = datetime.now()

merged_df = pd.merge(df_reduced, movies_df, on='movieId')
X_train, X_test, y_train, y_test = train_test_split(merged_df[features], merged_df['rating'], test_size=0.2, random_state=42)

model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', random_state=42)
model.fit(X_train, y_train)
ClusterFinal = datetime.now()
clustertime = ClusterFinal - ClusterInicial
segundos = clustertime.seconds
print("Entrenado en "+ str(segundos)+ " segundos.")

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Error cuadrático medio (MSE): {mse}')

all_movie_features = movies_df[features]
all_movie_ratings = model.predict(all_movie_features)

# Crea un gráfico de dispersión para visualizar las calificaciones predichas
plt.scatter(all_movie_ratings, movies_df['movieId'], alpha=0.5)
plt.xlabel('Calificaciones predichas')
plt.ylabel('ID de película')
plt.title('Rendimiento del modelo: Calificaciones predichas')
plt.show()

def recommend_movies(genre):
    if genre in mlb.classes_:
        genre_movies = movies_df[movies_df[genre] == 1]

        user_movie_features = genre_movies[features]
        user_movie_ratings = model.predict(user_movie_features)

        recommendations = genre_movies.assign(rating=user_movie_ratings).sort_values('rating', ascending=False)

        recommended_movies = recommendations['title'].head(10).to_string(index=False)
        print(f"Las siguientes películas de '{genre}' te podrían interesar:\n{recommended_movies}")
    else:
        print("No se encontraron películas con ese género")


while True:
    genre = input("Ingresa un género de película (o escribe 'stop' para finalizar): ")
    if genre.lower() == 'stop':
        break
    recommend_movies(genre)
