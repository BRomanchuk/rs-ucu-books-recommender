class BaseRecommender:
    def __init__(self):
        pass
    
    def fit(self, items, users, ratings):
        pass

    def predict(self, users, items):
        pass

    def eval(self, gt_ratings, pred_ratings):
        pass