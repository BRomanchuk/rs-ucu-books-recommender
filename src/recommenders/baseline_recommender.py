from src.recommenders.base import BaseRecommender
from src.evaluation import abs_loss

import numpy as np
import pandas as pd

def list_minus(a, b):
    return [x for x in a if x not in b]


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
        filtered_avg_items_ratings = avg_items_ratings[number_of_ratings > 100]
        # Get the 10 items with the highest average rating
        best_items = np.array(filtered_avg_items_ratings.sort_values(ascending=False).index)
        # Get the 10 best items for each user
        recommendations = []
        for user in users:
            user_mask = self.ratings['User-ID'] == user
            user_films = self.ratings[user_mask]['ISBN'].values
            user_best_items = list_minus(best_items, user_films)[:10]
            recommendations.append(user_best_items)
        return recommendations
    
    def eval(self, gt_ratings, pred_ratings):
        pass
    