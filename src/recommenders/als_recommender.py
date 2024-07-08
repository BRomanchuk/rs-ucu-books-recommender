import numpy as np
from numpy.linalg import solve
from tqdm import tqdm


class ALSRecommender:
    def __init__(self, num_users, num_items, num_factors=10, regularization=0.1, num_iterations=10):
        self.num_factors = num_factors
        self.regularization = regularization
        self.num_iterations = num_iterations
        self.user_factors = None
        self.item_factors = None
        self.isbn_to_index = None
        self.num_users = num_users
        self.num_items = num_items

    def fit(self, user_item_matrix, isbn_to_index):
        self.isbn_to_index = isbn_to_index
        self.user_factors = np.random.rand(self.num_users, self.num_factors)
        self.item_factors = np.random.rand(self.num_items, self.num_factors)

        for iteration in range(self.num_iterations):
            self.user_factors = self.als_step(user_item_matrix, self.item_factors, self.user_factors)
            self.item_factors = self.als_step(user_item_matrix.T, self.user_factors, self.item_factors)
            print(f"Iteration {iteration + 1} complete.")

    def als_step(self, R, fixed_factors, update_factors):
        num_factors = update_factors.shape[1]
        fixed_T_fixed = fixed_factors.T @ fixed_factors
        lambda_eye = self.regularization * np.eye(num_factors)

        for u in tqdm(range(update_factors.shape[0])):
            user_ratings = R[u].toarray().flatten()
            rated_indices = np.where(user_ratings != 0)[0]
            fixed_rated = fixed_factors[rated_indices]

            Ai = fixed_T_fixed + fixed_rated.T @ fixed_rated + lambda_eye
            Vi = fixed_rated.T @ user_ratings[rated_indices]

            update_factors[u] = solve(Ai, Vi)
        return update_factors

    def predict(self, user_id, isbn):
        item_idx = self.isbn_to_index.get(isbn)
        if item_idx is None:
            return None
        return self.user_factors[user_id] @ self.item_factors[item_idx].T

    def recommend(self, user_id, num_recommendations=5):
        scores = self.user_factors[user_id] @ self.item_factors.T
        best_items_indices = np.argsort(scores)[-num_recommendations:][::-1]
        index_to_isbn = {v: k for k, v in self.isbn_to_index.items()}
        best_isbns = [index_to_isbn[idx] for idx in best_items_indices]
        return best_isbns