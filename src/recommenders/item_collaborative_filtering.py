from src.recommenders.base import BaseRecommender
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

class ItemRecommender(BaseRecommender):
    def __init__(self):
        super().__init__()
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
        self.normalized_matrix = normalized_matrix

        item_user_matrix = csr_matrix((ratings['Rating'], (ratings['ISBN_i'], ratings['User-ID'])), dtype=np.float64)
        self.similarity_matrix = cosine_similarity(normalized_matrix)

        self.books = books
        
    def predict(self, users, items):
        user_predictions = {}
        for user_id in users:
            # Get the row corresponding to the user
            user_row = self.normalized_matrix.getrow(user_id)
            top_rated = np.argsort(user_row.data)[::-1][:self.n_recomm]

            recommended_books = []
            for book_i in top_rated:
                book_row = self.similarity_matrix.getrow(book_i)
                most_similar_books = np.argsort(book_row.data)[::-1][:self.n_recomm]
                recommended_books.extend(most_similar_books)

            recommended_books = list_minus(set(recommended_books), user_row.indices)
            average_ratings = {}
            for book in recommended_books:
                # Get the average rating for the book
                average_rating = self.normalized_matrix.getcol(book).data.mean()
                average_ratings[book] = average_rating

            sorted_recommended_books = sorted(recommended_books, key=lambda book: average_ratings[book], reverse=True)[:self.n_recomm]

            user_predictions[user_id] = self.books.iloc[recommended_books]
        
        return user_predictions

    def eval(self, gt_ratings, pred_ratings):
        pass