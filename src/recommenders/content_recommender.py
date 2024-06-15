import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.evaluation import ndcg, precision_at_k, recall_at_k, average_precision
from src.recommenders.base import BaseRecommender
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


class ContentRecommender(BaseRecommender):
    def __init__(self):
        super().__init__()
        self.tfidf_matrix = None
        self.item_ids = None

    def fit(self, items, users, ratings):
        self.item_ids = items['item_id'].tolist()
        custom_stop_words = set(ENGLISH_STOP_WORDS)
        tfidf = TfidfVectorizer(stop_words=custom_stop_words)
        self.tfidf_matrix = tfidf.fit_transform(items['description'])

    def predict(self, users, items):
        user_predictions = {}
        for user in users:
            sim_scores = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
            predicted_ratings = {item: np.mean(sim_scores[self.item_ids.index(item)]) for item in items}
            user_predictions[user] = predicted_ratings
        return user_predictions

    def eval(self, gt_ratings, pred_ratings):
        ndcg_scores = []
        precision_scores = []
        recall_scores = []
        ap_scores = []
        k = 10

        for user_id in pred_ratings:
            predicted_items = pred_ratings[user_id]
            relevant_items = gt_ratings.get(user_id, [])

            relevances = [1 if item in relevant_items else 0 for item in predicted_items[:k]]
            ndcg_score = ndcg(relevances, k)
            ndcg_scores.append(ndcg_score)

            precision_score = precision_at_k(predicted_items, relevant_items, k)
            precision_scores.append(precision_score)

            recall_score = recall_at_k(predicted_items, relevant_items, k)
            recall_scores.append(recall_score)

            ap_score = average_precision(predicted_items, relevant_items)
            ap_scores.append(ap_score)

        results = {
            'mean_ndcg': np.mean(ndcg_scores),
            'mean_precision': np.mean(precision_scores),
            'mean_recall': np.mean(recall_scores),
            'mean_ap': np.mean(ap_scores)
        }
        return results
