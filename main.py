from Utils.data_manager import DataManager
from Utils.evaluation import Evaluation
from recommender_system import RecommenderSystem
import os

if __name__ == '__main__':

    # set data path
    cur_path = os.getcwd()
    path_users = os.path.join(cur_path, "Data/ml-1m/users.dat")
    path_movies = os.path.join(cur_path, "Data/ml-1m/movies.dat")
    path_ratings = os.path.join(cur_path, "Data/ml-1m/ratings.dat")

    # get the data
    data = DataManager(path_users=path_users, path_movies=path_movies, path_ratings=path_ratings, train_test_split=0.1,
                       max_users=1000, max_movies=1000)

    # predict the test data
    recommender_system = RecommenderSystem(data, n_neighbors=10, method_name="item_based")
    # # or use this "user_based" method:
    # recommender_system = RecommenderSystem(data, n_neighbors=100, method_name="user_based")
    prediction = recommender_system.predict(data.df_ratings_test)

    # evaluate our results
    true = data.df_ratings_test["Rating"].tolist()
    score = Evaluation.mean_manhattan_distance(prediction=prediction, true=true)
    print(score)

    # evaluate if everything was the mean value of true = 3.6
    score = Evaluation.mean_manhattan_distance(prediction=[3.6]*len(true), true=true)
    print(score)

