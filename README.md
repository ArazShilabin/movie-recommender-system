# movie-recommender-system


## Introduction
A recommender system, or a recommendation system, is a subclass of information filtering systems that seeks to predict the "rating" or "preference" a user would give to an item. Here we have built a program to do the same and predict user ratings (scaling from 1 to 5) on different movies using the user's previous ratings.



## Dataset
you can get the dataset from this [website](https://grouplens.org/datasets/movielens/). Here the "MovieLens 1M Dataset" has been used to test the algorithms. Just download and unzip it in the Data folder. 



## Algorithms :hugs:

- **Item-Based**: here we use item-based collaborative filtering (CF) and KNN, to predict the score the user will give to a movie depending on how he has rated other movies with similar genres.
 
 - **User-Based**: in this method, the program starts imputing the missing values using knn-imputer then tries finding users with similar movie tastes (users who have rated movies similarly to our user) using KNN. In the end, predicts the new movie depending on how the other similar users have predicted the same movie.



