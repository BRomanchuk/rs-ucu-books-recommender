import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def abs_loss(gt_ratings, pred_ratings):
    users = gt_ratings['User-ID'].unique()
    items = gt_ratings['ISBN'].unique()
    loss = 0
    for user in users:
        for item in items:
            gt_rating = gt_ratings.get_rating(item, user)
            if gt_rating is None:
                continue
            pred_rating = pred_ratings.get_rating(item, user)
            loss += np.abs(gt_rating - pred_rating)
    return loss / (len(users) * len(items))


def l2_loss(gt_ratings, pred_ratings):
    users = gt_ratings['User-ID'].unique()
    items = gt_ratings['ISBN'].unique()
    loss = 0
    for user in users:
        for item in items:
            gt_rating = gt_ratings.get_rating(item, user)
            if gt_rating is None:
                continue
            pred_rating = pred_ratings.get_rating(item, user)
            loss += (gt_rating - pred_rating) ** 2
    return np.sqrt(loss / (len(users) * len(items)))


def dcg(relevances, rank=10):
    relevances = np.asarray(relevances)[:rank]
    n_relevances = len(relevances)
    if n_relevances == 0:
        return 0.

    discounts = np.log2(np.arange(n_relevances) + 2)
    return np.sum(relevances / discounts)


def ndcg(relevances, rank=10):
    best_dcg = dcg(sorted(relevances, reverse=True), rank)
    if best_dcg == 0:
        return 0.

    return dcg(relevances, rank) / best_dcg


def precision_at_k(recommended_items, relevant_items, k):
    recommended_at_k = recommended_items[:k]
    relevant_and_recommended = [item for item in recommended_at_k if item in relevant_items]
    return len(relevant_and_recommended) / k


def recall_at_k(recommended_items, relevant_items, k):
    recommended_at_k = recommended_items[:k]
    relevant_and_recommended = [item for item in recommended_at_k if item in relevant_items]
    return len(relevant_and_recommended) / len(relevant_items)


def average_precision(recommended_items, relevant_items):
    ap_sum = 0
    hit_count = 0

    for k in range(1, len(recommended_items) + 1):
        if recommended_items[k - 1] in relevant_items:
            hit_count += 1
            ap_sum += hit_count / k

    return ap_sum / len(relevant_items)
