import heapq

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from src.evaluation import abs_loss, l2_loss, ndcg, precision_at_k, recall_at_k, average_precision
from src.recommenders.base import BaseRecommender


class FunkSVDRecommender(BaseRecommender):
    def __init__(self, n_factors=10, learning_rate=0.01, regularization=0.1, iterations=20):
        super().__init__()
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.iterations = iterations
        self.P = None
        self.Q = None
        self.b_u = None
        self.b_i = None
        self.mu = None
        self.user_ids = None
        self.item_ids = None
        self.R = None
        self.user_ages = None
        self.tfidf_matrix = None
        self.tfidf = None
        self.books = None

    def fit(self, ratings, books, users):
        # Create a user-item matrix
        ratings_matrix = ratings.pivot(index='user_id', columns='isbn', values='rating').fillna(0)
        self.user_ids = ratings_matrix.index.tolist()
        self.item_ids = ratings_matrix.columns.tolist()
        self.R = ratings_matrix.values

        # Initialize user and item factors (embeddings)
        self.n_users, self.n_items = self.R.shape
        self.P = np.random.normal(scale=1. / self.n_factors, size=(self.n_users, self.n_factors))
        self.Q = np.random.normal(scale=1. / self.n_factors, size=(self.n_items, self.n_factors))
        self.b_u = np.zeros(self.n_users)
        self.b_i = np.zeros(self.n_items)
        self.mu = np.mean(self.R[np.where(self.R != 0)])

        # Initialize user ages
        user_ages = users.set_index('user_id').loc[self.user_ids, 'age'].fillna(0).values
        self.user_ages = user_ages / user_ages.max()  # Normalize ages

        # Compute TF-IDF for book descriptions
        books['Author'] = books['Author'].fillna('missing')
        books['description'] = books['Title'] + ' ' + books['Author']
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(books['description'])
        self.isbn_to_idx = {isbn: i for i, isbn in enumerate(books['ISBN'])}
        self.books = books

        # Create user-item interaction matrix
        self.R_mask = self.R != 0

        # Funk SVD optimization
        for _ in range(self.iterations):
            self._sgd_step()

    def _sgd_step(self):
        for i in range(self.n_users):
            for j in range(self.n_items):
                if self.R_mask[i, j]:
                    prediction = self.predict_single(i, j)
                    error = self.R[i, j] - prediction

                    self.b_u[i] += self.learning_rate * (error - self.regularization * self.b_u[i])
                    self.b_i[j] += self.learning_rate * (error - self.regularization * self.b_i[j])

                    for f in range(self.n_factors):
                        puf = self.P[i, f]
                        qif = self.Q[j, f]
                        self.P[i, f] += self.learning_rate * (error * qif - self.regularization * puf)
                        self.Q[j, f] += self.learning_rate * (error * puf - self.regularization * qif)

    def predict_single(self, user_index, item_index):
        user_embedding = np.hstack([self.P[user_index], self.user_ages[user_index]])
        item_embedding = np.hstack([self.Q[item_index], self.tfidf_matrix[item_index].toarray().flatten()])
        prediction = self.mu + self.b_u[user_index] + self.b_i[item_index] + np.dot(user_embedding, item_embedding)
        return prediction

    def get_rating(self, item, user):
        user_index = self.user_ids.index(user)
        item_index = self.item_ids.index(item)
        return self.predict_single(user_index, item_index)

    def predict(self, user_ratings, num_recommendations=5):
        user_predictions = {}
        for user_id, group in tqdm.tqdm(user_ratings.groupby('user_id'), desc='Generating recommendations'):
            user_index = self.user_ids.index(user_id)
            user_interactions = group['isbn'].tolist()
            user_rated_indices = [self.item_ids.index(isbn) for isbn in user_interactions if isbn in self.item_ids]

            if not user_rated_indices:
                continue

            user_vector = np.hstack([self.P[user_index], self.user_ages[user_index]])
            scores = np.array([self.predict_single(user_index, i) for i in range(self.n_items)])
            scores[user_rated_indices] = float('-inf')  # Exclude already rated items

            top_indices = heapq.nlargest(num_recommendations, range(len(scores)), key=lambda x: scores[x])
            recommended_books = [self.item_ids[idx] for idx in top_indices]
            user_predictions[user_id] = recommended_books

        return user_predictions

    def eval(self, gt_ratings, k=10):
        pred_ratings = gt_ratings.copy()
        pred_ratings['pred_rating'] = pred_ratings.apply(lambda row: self.get_rating(row['isbn'], row['user_id']),
                                                         axis=1)

        abs_loss_val = abs_loss(gt_ratings, self)
        l2_loss_val = l2_loss(gt_ratings, self)

        all_relevances = []
        all_recommended = []
        for user_id in gt_ratings['user_id'].unique():
            user_relevances = gt_ratings[gt_ratings['user_id'] == user_id].sort_values(by='rating', ascending=False)[
                'rating'].tolist()
            all_relevances.append(user_relevances)
            user_recommended = self.predict(gt_ratings[gt_ratings['user_id'] == user_id], num_recommendations=k)[
                user_id]
            all_recommended.append(user_recommended)

        ndcg_val = np.mean([ndcg(relevances, k) for relevances in all_relevances])
        precision_at_k_val = np.mean([precision_at_k(recommended, relevances, k) for recommended, relevances in
                                      zip(all_recommended, all_relevances)])
        recall_at_k_val = np.mean([recall_at_k(recommended, relevances, k) for recommended, relevances in
                                   zip(all_recommended, all_relevances)])
        average_precision_val = np.mean([average_precision(recommended, relevances, k) for recommended, relevances in
                                         zip(all_recommended, all_relevances)])

        return {
            'abs_loss': abs_loss_val,
            'l2_loss': l2_loss_val,
            'ndcg': ndcg_val,
            'precision_at_k': precision_at_k_val,
            'recall_at_k': recall_at_k_val,
            'average_precision': average_precision_val
        }
