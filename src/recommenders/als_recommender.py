import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

class ALSRecommender:
    def __init__(self, num_factors=10, regularization=0.1, num_iterations=10):
        self.num_factors = num_factors
        self.regularization = regularization
        self.num_iterations = num_iterations

    def fit(self, R):
        self.num_users, self.num_items = R.shape
        self.user_factors = np.random.rand(self.num_users, self.num_factors)
        self.item_factors = np.random.rand(self.num_items, self.num_factors)

        R_csr = sp.csr_matrix(R)

        for iteration in range(self.num_iterations):
            print(f"Iteration {iteration + 1}/{self.num_iterations}")
            self.user_factors = self._als_step(R_csr, self.user_factors, self.item_factors, self.num_users, self.num_factors)
            self.item_factors = self._als_step(R_csr.T, self.item_factors, self.user_factors, self.num_items, self.num_factors)

    def _als_step(self, R, X, Y, num_rows, num_factors):
        YTY = Y.T @ Y
        lambda_I = np.eye(num_factors) * self.regularization

        for u in range(num_rows):
            start_idx = R.indptr[u]
            end_idx = R.indptr[u + 1]
            indices = R.indices[start_idx:end_idx]
            ratings = R.data[start_idx:end_idx]

            Y_u = Y[indices]
            A = YTY + Y_u.T @ Y_u + lambda_I
            b = Y_u.T @ ratings

            X[u, :] = spsolve(A, b)

        return X

    def predict(self, user, item):
        return self.user_factors[user, :].dot(self.item_factors[item, :])

    def recommend(self, user, num_recommendations=10):
        scores = self.user_factors[user, :].dot(self.item_factors.T)
        top_items = np.argsort(scores)[-num_recommendations:][::-1]
        return top_items