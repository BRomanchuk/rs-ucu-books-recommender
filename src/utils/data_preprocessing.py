import pandas as pd


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


