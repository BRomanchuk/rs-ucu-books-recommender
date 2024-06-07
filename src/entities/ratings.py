
class Ratings:
    def __init__(self, ratings_df, item_col='ISBN', user_col='User-ID', rating_col='Rating'):
        self.ratings_df = ratings_df
        self.item_col = item_col
        self.user_col = user_col
        self.rating_col = rating_col

    def get_user_ratings(self, user):
        user_mask = self.get_user_mask(user)
        return self.ratings_df.loc[user_mask]
    
    def get_item_ratings(self, item):
        item_mask = self.get_item_mask(item)
        return self.ratings_df.loc[item_mask]
    
    def item_avg(self, item):
        item_ratings = self.get_item_ratings(item)
        avg_rating = item_ratings[self.rating_col].mean()
        return avg_rating
    
    def user_avg(self, user):
        user_ratings = self.get_user_ratings(user)
        avg_rating = user_ratings[self.rating_col].mean()
        return avg_rating

    def get_rating(self, item, user):
        rating_mask = (
            self.get_item_mask(item) &
            self.get_user_mask(user)
        )
        if rating_mask.sum() == 0:
            return None
        rating = self.ratings_df.loc[rating_mask].iloc[0]
        return rating

    def get_item_mask(self, item):
        mask = (self.ratings_df[self.item_col] == item)
        return mask
    
    def get_user_mask(self, user):
        mask = (self.ratings_df[self.user_col] == user)
        return mask