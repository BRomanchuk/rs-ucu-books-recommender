import numpy as np
import scipy as sp
import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentRecommender():
    def __init__(self):
        super().__init__()
        self.tfidf_matrix = None
        self.isbn_to_idx = None
        self.tfidf = None
        self.books = None

    def fit(self, books):
        books['Author'] = books['Author'].fillna('missing')
        books['description'] = books['Title'] + ' ' + books['Author']
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(books['description'])
        self.isbn_to_idx = {isbn: i for i, isbn in enumerate(books['ISBN'])}
        self.books = books

    def predict(self, user_ratings, num_recommendations=5):
        user_predictions = {}
        for user_id, group in tqdm(user_ratings.groupby('User-ID'), desc='Generating recommendations'):
            user_indices = [self.isbn_to_idx.get(isbn) for isbn in group['ISBN'] if isbn in self.isbn_to_idx]
            if not user_indices:
                continue

            average_vector = self.tfidf_matrix[user_indices].mean(axis=0)

            if isinstance(average_vector, sp.sparse.csr_matrix):
                average_vector = average_vector.toarray()

            average_vector_np = np.asarray(average_vector).reshape(1, -1)

            user_sim_scores = cosine_similarity(average_vector_np, self.tfidf_matrix)[0]

            top_indices = heapq.nlargest(num_recommendations, range(len(user_sim_scores)),
                                         key=lambda x: user_sim_scores[x] if x not in user_indices else float('-inf'))

            recommended_books = [self.books['ISBN'].iloc[idx] for idx in top_indices]
            user_predictions[user_id] = recommended_books

        return user_predictions

