import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


pandas.set_option('display.max_columns', None)

def get_weighted_rating(df, min_votes, average_rating_all_movies):
    average_rating = df["vote_average"]
    num_votes = df["vote_count"]
    weighted_rating = (num_votes / (num_votes + min_votes)) * average_rating + (min_votes / (num_votes + min_votes)) * average_rating_all_movies
    return weighted_rating

if __name__ == '__main__':
    # Popularity Based Filtering
    movies = pandas.read_csv("movies.csv")
    credits = pandas.read_csv("credits.csv")
    ratings = pandas.read_csv("ratings.csv")


    min_votes = movies["vote_count"].quantile(0.9)
    average_rating_all_movie = movies["vote_average"].mean()
    movies_filtered = movies.copy().loc[movies["vote_count"] >= min_votes]

    movies_filtered["weighted_rating"] = movies.apply(get_weighted_rating, axis=1, args=(min_votes, average_rating_all_movie))

    movies_filtered.sort_values("weighted_rating", ascending=False)
    # print(movies_filtered.head(10))

    # Content Based Filtering
    movies_small = pandas.read_csv("movies_small.csv", sep=";")
    print(movies_small.head())

    tfidf = TfidfVectorizer(stop_words="english")
    movies_small["overview"] = movies_small["overview"].fillna("")
    tfidf_matrix = tfidf.fit_transform(movies_small["overview"].values.astype('U'))

    print(pandas.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out()))
    similarity_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
    print(similarity_matrix)

