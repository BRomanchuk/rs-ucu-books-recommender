from src.recommenders.base import BaseRecommender
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

class UserRecommender(BaseRecommender):
    def __init__(self):
        super().__init__()
        self.n_neighbors = 30
        self.n_recomm = 5

    def fit(self, items, users, ratings):
        books = items.reset_index() # add index as a column
        isbn_mapping = {category: idx for idx, category in enumerate(books['ISBN'])}
        
        ratings = ratings.copy()
        ratings['ISBN_i'] = ratings['ISBN'].map(isbn_mapping) # map ISBN to index
        ratings.dropna(subset=['ISBN_i'], inplace=True) # drop rows with NaN ISBN_i
        ratings['ISBN_i'] = ratings['ISBN_i'].astype(np.int32)

        # Create a sparse user-item matrix
        user_item_matrix = csr_matrix((ratings['Rating'], (ratings['User-ID'], ratings['ISBN_i'])), dtype=np.float64)

        # Normalize the user-item matrix
        normalized_matrix = user_item_matrix.copy()
        self.means = np.array([normalized_matrix[i].data.mean() for i in range(normalized_matrix.shape[0])])
        normalized_matrix.data -= np.repeat(self.means, np.diff(normalized_matrix.indptr))

        # Fit a nearest neighbors model
        self.model = NearestNeighbors(n_neighbors=30, metric='cosine', n_jobs=-1)
        self.model.fit(normalized_matrix)

        self.books = books
        self.normalized_matrix = normalized_matrix
        
    def predict(self, users, items):
        user_predictions = {}
        for user_id in tqdm(users):
            # Get the row corresponding to the user
            user_row = self.normalized_matrix.getrow(user_id)

            # Find the nearest neighbors
            distances, indices = self.model.kneighbors(user_row, self.n_neighbors)
            #print(f"Indices of nearest neighbors ({user_id}): {indices}")

            # Get the ratings of the neighbors
            neighbor_ratings = self.normalized_matrix[indices[0]].toarray()
            weighted_rows = neighbor_ratings * distances[0][:, np.newaxis]

            result_vector = weighted_rows.sum(axis=0)
            result_vector = result_vector / distances.sum()

            result_vector += self.means[user_id]
            result_vector.ravel()[user_row.indices] = 0

            sorted_indices = np.argsort(result_vector)
            top_indices = sorted_indices[-self.n_recomm:][::-1]

            user_predictions[user_id] = self.books.iloc[top_indices]

        return user_predictions

    def eval(self, gt_ratings, pred_ratings):
        pass