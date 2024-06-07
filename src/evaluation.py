import numpy as np
import pandas as pd

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