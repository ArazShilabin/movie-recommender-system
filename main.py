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
    data = DataManager(path_users=path_users, path_movies=path_movies, path_ratings=path_ratings, train_test_split=0.05,
                       max_users=1000, max_movies=1000)

    # build models based on train data (rs = recommender_system)
    rs_user_based = RecommenderSystem(data, n_neighbors=50, method_name="user_based")
    rs_item_based = RecommenderSystem(data, n_neighbors=50, method_name="item_based")
    rs_content_based = RecommenderSystem(data, n_neighbors=50, method_name="content_based")

    # predict the test data using the built models
    prediction_user_based = rs_user_based.predict(data.df_ratings_test)
    prediction_item_based = rs_item_based.predict(data.df_ratings_test)
    prediction_content_based = rs_content_based.predict(data.df_ratings_test)

    # evaluate our results
    true_labels = data.df_ratings_test["Rating"].tolist()

    score_user_based = Evaluation.mean_manhattan_distance(prediction=prediction_user_based, true_labels=true_labels)
    score_item_based = Evaluation.mean_manhattan_distance(prediction=prediction_item_based, true_labels=true_labels)
    score_content_based = Evaluation.mean_manhattan_distance(prediction=prediction_content_based,
                                                             true_labels=true_labels)

    # print the results
    print(f"Our classifiers results:")
    print(f"Loss in 'user_based' recommender system = {score_user_based}")
    print(f"Loss in 'item_based' recommender system = {score_item_based}")
    print(f"Loss in 'content_based' recommender system = {score_content_based}", end="\n\n")

    # evaluate if our predicted scores were the mean value of true_labels ~= 3.56
    print(f'The mean value of scores are: {sum(true_labels) / len(true_labels)}.')
    prediction_greedy = [sum(true_labels) / len(true_labels)] * len(true_labels)
    greedy_score = Evaluation.mean_manhattan_distance(prediction=prediction_greedy, true_labels=true_labels)
    print(f"Loss in 'greedy' recommender system = {greedy_score}")
