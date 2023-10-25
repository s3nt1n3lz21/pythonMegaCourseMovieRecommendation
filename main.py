import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Dataset, Reader, SVD, model_selection

pandas.set_option('display.max_columns', None)

def get_weighted_rating(df, min_votes, average_rating_all_movies):
    average_rating = df["vote_average"]
    num_votes = df["vote_count"]
    weighted_rating = (num_votes / (num_votes + min_votes)) * average_rating + (min_votes / (num_votes + min_votes)) * average_rating_all_movies
    return weighted_rating

def similar_movies(similarity_matrix, movie_index):
    scores = list(enumerate(similarity_matrix[movie_index]))
    print(scores)
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    print(sorted_scores)
    movies = [item[0] for item in sorted_scores[1:]]
    return movies

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
    movies_small = pandas.read_csv("movies.csv")
    # print(movies_small.head())

    tfidf = TfidfVectorizer(stop_words="english")
    movies_small["overview"] = movies_small["overview"].fillna("")
    tfidf_matrix = tfidf.fit_transform(movies_small["overview"].values.astype('U'))

    # print(pandas.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out()))
    similarity_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
    print(similarity_matrix)
    movie_title = "John Carter"
    movie_details = movies_small.loc[movies_small['title'] == movie_title]
    movie_index = movie_details.index[0]
    print(movie_index)
    movies_index_similar_by_overview = similar_movies(similarity_matrix, movie_index)
    print(movies_index_similar_by_overview)
    # print(movies_small.iterrows())
    movies_similar_by_overview = movies_small["title"].iloc[movies_index_similar_by_overview]
    print(movies_similar_by_overview)

    # Collaborative Filtering
    # Predict what rating a user would give a movie they have not seen yet based on similar peoples ratings
    print(ratings.head())
    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

    trainset = dataset.build_full_trainset()
    svd = SVD()
    svd.fit(trainset)

    user_id = 15
    movie_id = 1956
    predicted_rating = svd.predict(user_id, movie_id)
    print(predicted_rating.est)

    # Validate the model, check the models prediction against the actual whole dataset
    model_errors = model_selection.cross_validate(svd, dataset, measures=["RMSE", "MAE"])
    print(model_errors)