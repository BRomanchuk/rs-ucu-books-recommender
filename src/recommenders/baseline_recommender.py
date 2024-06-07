from src.entities.ratings import Ratings
from src.evaluation import abs_loss

import pandas as pd


class BaselineRecommender:
    def __init__(self):
        pass

    def fit(self, items, users, ratings: Ratings):
        self.ratings = ratings
    
    def predict(self, users, items):
        users = self.ratings['User-ID'].unique()
        items = self.ratings['ISBN'].unique()
        predictions = []
        for item in items:
            item_rating = self.ratings.item_avg(item)
            for user in users:
                predictions.append({
                    "User-ID": user,
                    "ISBN": item,
                    "Rating": item_rating
                })
        predictions = pd.DataFrame(predictions)
        return predictions
    
    def eval(self, gt_ratings, pred_ratings):
        return abs_loss(gt_ratings, pred_ratings)
    