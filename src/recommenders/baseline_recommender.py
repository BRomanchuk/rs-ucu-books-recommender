from src.recommenders.base import BaseRecommender
from src.evaluation import abs_loss

import numpy as np
import pandas as pd


class BaselineRecommender(BaseRecommender):
    def __init__(self):
        super().__init__()

    def fit(self, items, users, ratings):
        self.ratings = ratings
    
    def predict(self, users, items=None):
        # Group the ratings by item
        grouped_rating = self.ratings.groupby('ISBN')['Rating']
        # Count the number of ratings for each item
        number_of_ratings = grouped_rating.count()
        # Calculate the average rating for each item
        avg_items_ratings = grouped_rating.mean()
        # Filter the items with less than 10 ratings
        filtered_avg_items_ratings = avg_items_ratings[number_of_ratings > 10]
        # Get the 10 items with the highest average rating
        best_items = np.array(filtered_avg_items_ratings.sort_values(ascending=False).head(10).index)
        # Return the same 10 items for all users
        return [best_items for _ in users]
    
    def eval(self, gt_ratings, pred_ratings):
        pass
    