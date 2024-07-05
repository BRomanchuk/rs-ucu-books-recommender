from src.recommenders.base import BaseRecommender

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class DNNRecommender (BaseRecommender):
    def __init__(self):
        super.__init__(self)
    
    def fit(self, items, users, ratings):
        # Normalize the user-item matrix
        normalized_matrix = user_item_matrix.copy()
        self.means = np.array([normalized_matrix[i].data.mean() for i in range(normalized_matrix.shape[0])])
        normalized_matrix.data -= np.repeat(self.means, np.diff(normalized_matrix.indptr))

        pass

    def predict(self, users, items):
        pass

    def eval(self, gt_ratings, pred_ratings):
        pass