import pandas as pd
import opendatasets as od
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')


def clean_text(text_column):
    """
    Limpa os dados em texto para que a análise ocorra de forma coesa e eficiente.

    Parametros:
        - Coluna que aprensenta informações textuais sobre o filme.

    """
    # Converte para letras minúsculas.
    text_column = text_column.str.lower()

    # Remove pontuações.
    text_column = text_column.str.replace('[^\w\s]', '')

    # Remove números.
    text_column = text_column.replace('\d+', '', regex=True)

    # Remove 'stopwords' (palavra que é removida antes ou após o processamento de um texto em linguagem natural).
    stop_words = stopwords.words('english')
    # Preenche valores identificados como nulo (NaN) para que possa ocorrer análises.
    text_column = text_column.fillna('').apply(
        lambda x: ' '.join([word for word in x.split() if word not in stop_words])
    )

    return text_column


def find_similar_movies(data, movie_index, top_n=5):
    """
    Encontra e 'imprime' os títulos dos filmes mais semelhantes ao filme dado.

    Parameters:
    - data: DataFrame (formato disponibilizado pela biblioteca pandas) a ser analisado.
    - movie_index: Index do filme a ser comparado.
    - top_n: Número de filmes mais semelhantes ao filme de entrada.
    """
    # Inicia CountVectorizer.
    cv = CountVectorizer(max_features=1000, stop_words='english')

    # Vetoriza a coluna 'tags', que no caso do presente estudo é o que possue informações relevantes sobre o filme.
    vectorized_data = cv.fit_transform(data['tags'])

    # Calcula a distância coseno entre os vetores.
    similarity_matrix = cosine_similarity(vectorized_data)

    # Calcula a similaridade do score obtido para o filme especificado em 'movie_index'.
    similarity_scores = sorted(list(enumerate(similarity_matrix[movie_index])), reverse=True,
                               key=lambda vector: vector[1])

    # Imprime os títulos dos filmes mais semelhantes.
    print(f"Top {top_n} movies similar to {data.iloc[movie_index].title}:\n")
    for i in similarity_scores[1:top_n + 1]:  # Skip the first one since it will be the movie itself
        print(data.iloc[i[0]].title)


def main():
    # Carrega os dados
    movies = pd.read_csv("movies_metadata.csv")
    movies = movies[["id", "title", "overview", "genres"]]

    # Cria coluna tags, a partir da união das colunas 'overview' e 'genres'.
    movies['tags'] = movies["overview"].fillna('') + ' ' + movies["genres"].fillna('')
    data = movies.drop(columns=["overview", "genres"]).head(5000)

    # Utiliza a função 'clean_text' para limpar as informações textuais da coluna 'tags'.
    data['tags'] = clean_text(data['tags'])

    # Encontra e imprime os 5 filmes mais semelhantes ao filme cudo index=10.
    find_similar_movies(data, movie_index=10, top_n=5)


if __name__ == "__main__":
    main()