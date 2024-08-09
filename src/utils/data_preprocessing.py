import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.sparse import csr_matrix

def preprocess(books, ratings, users):
    df_prep_step_1 = pd.merge(books, ratings.query("`Rating` > 0"), on='ISBN', how='inner')

    # join users data
    df_prep_step_2 = pd.merge(df_prep_step_1, users, on='User-ID', how='inner')

    df_prep = df_prep_step_2.drop(['Publisher'], axis=1)
    df_result = df_prep.drop_duplicates()

    df_result['Original_NaN'] = df_result['Age'].isna()

    # Convert 'Age' to numeric, turning non-numeric values into NaN
    df_result['Age'] = pd.to_numeric(df_result['Age'], errors='coerce')

    # Drop rows where 'Age' is NaN and were not originally NaN
    df_result = df_result[~(users['Age'].isna() & ~df_result['Original_NaN'])]

    # Drop the 'Original_NaN' column as it's no longer needed
    df_result.drop(columns=['Original_NaN'], inplace=True)

    df_result['User-ID'] = pd.to_numeric(df_result['User-ID'], errors='coerce')

    # Drop rows where 'Age' is NaN and were not originally NaN
    df_result = df_result[~(df_result['User-ID'].isna())]

    age_outliers = df_result.query("Age > 100 or Age < 6")

    user_outliers = age_outliers["User-ID"].to_list()

    # exclude age outliers
    df_result = df_result[~df_result["User-ID"].isin(user_outliers)]

    users = df_result['User-ID', 'Age']

    ratings = df_result['User-ID', 'ISBN', 'Rating']

    books = df_result['ISBN', 'Title', 'Author', 'Year']

    return users, ratings, books

def preprocess2(items, users, ratings):
    #   Leave only numeric values in 'Age' column and drop NaN values
    users['Age'] = [float(x) if (isinstance(x, (str)) and x.isnumeric()) else None for x in users['Age']]

    # Drop ratings with 0 as they don't provide any information 
    ratings = ratings[ratings.Rating > 0]

    # Drop duplicates
    items.drop_duplicates(subset='ISBN', inplace=True)
    items = items.reset_index()

    # Fill rows with NaN values in 'Year' and 'Age' columns with median values
    # We are not doing this with 'Age' in the users because it's an only feature column and it has half of the values as NaN
    items['Year'] = items['Year'].fillna(items['Year'].median())

    # Some User-IDs are not numberic, so we convert them
    users['User-ID'] = pd.to_numeric(users['User-ID'], errors='coerce')

    return items.dropna(), users.dropna(), ratings.dropna()


def user_item_normalized(books, ratings):
        books = books.reset_index() # add index as a column
        isbn_mapping = {category: idx for idx, category in enumerate(books['ISBN'])}
        
        ratings = ratings.copy()
        ratings['ISBN_i'] = ratings['ISBN'].map(isbn_mapping) # map ISBN to index
        ratings.dropna(subset=['ISBN_i'], inplace=True) # drop rows with NaN ISBN_i
        ratings['ISBN_i'] = ratings['ISBN_i'].astype(np.int32)
        
        # Create a sparse user-item matrix
        user_item_matrix = csr_matrix((ratings['Rating'], (ratings['User-ID'], ratings['ISBN_i'])), dtype=np.float64)

        # Normalize the user-item matrix
        normalized = user_item_matrix.copy()
        means = np.array([normalized[i].data.mean() for i in range(normalized.shape[0])])
        normalized.data -= np.repeat(means, np.diff(normalized.indptr))
        
        return means, normalized, books, ratings

def generate_random_timestamp(start_date, end_date):
     return start_date + timedelta(seconds=np.random.randint(0, int((end_date - start_date).total_seconds())))

def augment_timestamps(start_date, end_date, ratings_df): 

    ratings_df['timestamp'] = ratings_df.apply(lambda x: generate_random_timestamp(start_date, end_date), axis=1)
    return ratings_df
