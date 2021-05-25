import numpy as np
from sklearn.impute import KNNImputer
from sklearn.neighbors import NearestNeighbors


class UserBased:

    def __init__(self, data, n_neighbors):
        self.data = data
        self.n_neighbors = n_neighbors
        self.rating_matrix = self.get_rating_matrix()
        self.knn_model = self.build_knn_model()

    def get_rating_matrix(self):
        """
        :return: returns filled matrix  -->  [user_id, movie_id] = rating
        """
        n_movies = self.data.max_movies + 1
        n_users = self.data.max_users + 1
        rating_matrix_containing_nans = np.empty((n_users, n_movies)) * np.nan
        for idx, row in self.data.df_ratings_train.iterrows():
            rating_matrix_containing_nans[row["UserID"], row["MovieID"]] = row["Rating"]

        rating_matrix = self.knn_impute(rating_matrix_containing_nans)
        return rating_matrix

    def knn_impute(self, matrix):
        """
        the knn imputer will fill the nans using an estimate of knn but we will use twice the k-size of our normal knn
        so we get a wider estimation of our missing values before we use the real knn with our
        Collaborative filtering (CF) technique
        """
        imputer = KNNImputer(n_neighbors=self.n_neighbors * 2)  # always more than our normal knn (here I chose twice)
        matrix = imputer.fit_transform(matrix)
        return matrix

    def build_knn_model(self):
        neigh = NearestNeighbors(n_neighbors=self.n_neighbors, metric='cosine')
        neigh.fit(self.rating_matrix)
        return neigh

    def predict(self, df_ratings_test):
        predictions = []
        for idx, row in df_ratings_test.iterrows():
            prediction = self.predict_one(row["UserID"], row["MovieID"])
            predictions.append(prediction)
        return predictions

    def predict_one(self, user_id, movie_id):
        """
        :return: returns the predicted score of a movie that a user has watched
        """
        knn_similarities, knn_user_ids = self.knn_model.kneighbors(self.rating_matrix[user_id].reshape(1, -1))
        knn_similarities = knn_similarities[0].tolist()
        knn_user_ids = knn_user_ids[0].tolist()
        df_ratings_train = self.data.df_ratings_train
        correlation_sum = similarity_sum = 0

        for knn_user_id, similarity in zip(knn_user_ids, knn_similarities):
            rating = df_ratings_train["Rating"].loc[(df_ratings_train['UserID'] == knn_user_id) & (
                        df_ratings_train["MovieID"] == movie_id)]
            if rating.empty:
                continue
            rating = rating.values[0]  # just get the rating value instead of getting it as a pd.series
            correlation_sum += rating * similarity
            similarity_sum += similarity
        if similarity_sum == 0:
            return 3.6
        prediction = correlation_sum / similarity_sum
        return prediction

#         knn_similarity_id_rating = self.knn(user_id, movie_id)
#         correlation_sum = similarity_sum = 0
#         for one_similarity_id_rating in knn_similarity_id_rating:
#             similarity = one_similarity_id_rating[0]
#             rating = one_similarity_id_rating[2]
#             correlation_sum += rating * similarity
#             similarity_sum += similarity
#         if similarity_sum == 0:
#             return 3.6
#         prediction = correlation_sum / similarity_sum
#         return prediction
#
# def get_users_similarities(user_id1, user_id2, data):
#     df_ratings = data.df_ratings
#     user1_movies = df_ratings.loc[df_ratings["UserID"] == user_id1]
#     user2_movies = df_ratings.loc[df_ratings["UserID"] == user_id2]
#
#     # print(user_id1["Rating"])
#     user1_mean_scores = user1_movies["Rating"].mean()
#     user2_mean_scores = user2_movies["Rating"].mean()
#
#     merged_df = pd.merge(user1_movies, user2_movies, how='inner', on=['MovieID'])
#
#     user1_feature_list = np.array(merged_df["Rating_x"].tolist())
#     user2_feature_list = np.array(merged_df["Rating_y"].tolist())
#
#     similarity = SimilarityMetrics.normalized_cosine_similarity(user1_feature_list, user2_feature_list,
#                                                                 user1_mean_scores, user2_mean_scores)
#     return similarity
#
#
# def knn(user_id, data, k):
#     list_similarity_and_userid = []  # (similarity, user_id) we want to sort it later and return the K most simillar
#     for index2, row2 in data.df_ratings_test.iterrows():
#         if row2['UserID'] != user_id:
#             similarity = get_users_similarities(user_id, row2["UserID"], data)
#             list_similarity_and_userid.append((similarity, row2["UserID"]))
#
#     k_nearest = list_similarity_and_userid.sort(reverse=True)[:k]
#     return k_nearest
#
#
# def rating_prediction(user_id, movie_id, data, rating_matrix):
#     # knn_similarity_and_user_ids = knn(user_id, data, k=10)
#     knn_similarities, knn_user_ids = neigh.kneighbors(rating_matrix[user_id - 1].reshape(1, -1))
#     knn_similarities = knn_similarities[0].tolist()
#     knn_user_ids = knn_user_ids[0].tolist()
#     # print(knn_similarity_and_user_ids)
#     # print("knn done")
#     df_ratings = data.df_ratings
#
#     correlation_sum = similarity_sum = 0
#     for idx, knn_user_id in enumerate(knn_user_ids):
#
#         rating = df_ratings["Rating"].loc[(df_ratings['UserID'] == knn_user_id) & (df_ratings["MovieID"] == movie_id)]
#         if rating.empty:
#             continue
#         rating = rating.values[0]  # just get the rating value instead of getting it as a pd.series
#         similarity = knn_similarities[idx]
#         # similarity = get_users_similarities(user_id, knn_user_id, data)
#         correlation_sum += rating * similarity
#         similarity_sum += similarity
#     if similarity_sum == 0:
#         return 3
#     prediction = correlation_sum / similarity_sum
#     return prediction
#
#
# if __name__ == '__main__':
#
#     #########################
#     ##########################
#
#     cnt = error = error2 = error3 = error4 = error5 = error6 = error7 = 0
#     for index, row in all_data.df_ratings_test.iterrows():
#         pred = rating_prediction(row["UserID"], row["MovieID"], all_data, rating_matrix)
#         cnt += 1
#         error += abs(row["Rating"] - pred)
#         error2 += abs(3.60 - row["Rating"])
#         error7 += abs(row["Rating"])
#         # print(row["Rating"], pred)
#     # print(pred)
#     print(error / cnt)  # rmse
#     print(error2 / cnt)
#     print(error7 / cnt)
#     print(cnt)
