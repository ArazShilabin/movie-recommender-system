from Algorithms.item_based import ItemBased
from Algorithms.user_based import UserBased


class RecommenderSystem:
    def __init__(self, data, n_neighbors, method_name="item_based"):
        """
        :param n_neighbors: number of neighbors to consider
        :param method_name: it should be either "item_based" or "user_based"
        """
        self.data = data
        self.n_neighbors = n_neighbors
        self.method_name = method_name
        self.model = self.get_model()

    def get_model(self):
        if self.method_name == "item_based":
            return ItemBased(data=self.data, n_neighbors=self.n_neighbors)
        elif self.method_name == "user_based":
            return UserBased(data=self.data, n_neighbors=self.n_neighbors)
        else:
            raise Exception(f"method_name=\"{self.method_name}\" is not defined")

    def predict(self, df_ratings_test):
        if self.method_name == "item_based" or self.method_name == "user_based":
            return self.model.predict(df_ratings_test)
        else:
            raise Exception(f"method_name=\"{self.method_name}\" is not defined")
