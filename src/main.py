import utils.data_preprocessing
from datetime import datetime
import numpy as np
import pandas as pd

def main():
    # Augment timestamps
    start_date = datetime(2000, 1, 1)
    end_date = datetime(2020, 1, 1)

    ratings = pd.read_csv('data/Ratings.csv', delimiter=';', dtype={'User-ID': np.int32, 'ISBN': str, 'Rating': np.int8})
    print(ratings.head())
    augmented_ratings = utils.data_preprocessing.augment_timestamps(start_date, end_date, ratings)
    print(augmented_ratings.head())

    augmented_ratings.to_csv('data/Ratings_Time.csv', index=False, sep=';')

if __name__ == '__main__':
    main()