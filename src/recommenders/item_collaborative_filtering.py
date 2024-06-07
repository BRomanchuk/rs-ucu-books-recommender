from src.recommenders.base import BaseRecommender

class ItemRecommender(BaseRecommender):
    def __init__(self):
        super().__init__()

    def fit(self, items, users, ratings):
        pass

    def predict(self, users, items):
        pass

    def eval(self, gt_ratings, pred_ratings):
        pass