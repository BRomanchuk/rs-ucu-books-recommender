from src.recommenders.base import BaseRecommender

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import LabelEncoder

class DNNRecommenderWithFeatures(nn.Module):
    def __init__(self, num_users, num_items, num_user_features, num_item_features, embedding_dim, hidden_dim):
        super(DNNRecommenderWithFeatures, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        self.user_feature_layer = nn.Linear(num_user_features, embedding_dim)
        self.item_feature_layer = nn.Linear(num_item_features, embedding_dim)
        
        self.fc1 = nn.Linear(embedding_dim * 2 * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, user_ids, item_ids, user_features, item_features):
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        
        user_feature_embeds = self.user_feature_layer(user_features)
        item_feature_embeds = self.item_feature_layer(item_features)
        
        user_combined = torch.cat([user_embeds, user_feature_embeds], dim=1)
        item_combined = torch.cat([item_embeds, item_feature_embeds], dim=1)
        
        x = torch.cat([user_combined, item_combined], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze()

class DNNRecommender(BaseRecommender):

    def __init__(self):
        super().__init__()
        self.embedding_dim = 50
        self.hidden_dim = 128
        self.num_epochs = 20
        self.criterion = nn.MSELoss()
    
    def fit(self, items, users, ratings):
        # Initialize model, optimizer, and loss function
        num_users = len(self.user_encoder.classes_)
        num_items = len(self.isbn_encoder.classes_)

        items_tensor = torch.tensor(items, dtype=torch.float32)
        users_tensor = torch.tensor(users, dtype=torch.float32)
        ratings_tensor = torch.tensor(ratings, dtype=torch.float32)

        user_ids = users_tensor[:, 0].long()
        item_ids = items_tensor[:, 1].long()
        user_features = users_tensor[:, 1:]
        item_features = items_tensor[:, 1:]

        num_user_features = user_features.shape[1]  # Age
        num_item_features = item_features.shape[1]  # Author_encoded, Year, and Publisher_encoded

        self.model = DNNRecommenderWithFeatures(num_users, num_items, num_user_features, num_item_features, self.embedding_dim, self.hidden_dim)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(self.num_epochs):
            self.model.train()
            optimizer.zero_grad()
            
            predictions = self.model(user_ids, item_ids, user_features, item_features)
            loss = self.criterion(predictions, ratings_tensor)
            
            loss.backward()
            optimizer.step()
            
            print(f'Epoch {epoch+1}/{self.num_epochs}, Loss: {loss.item()}')


    def predict(self, users, items):
        with torch.no_grad():
            items_tensor = torch.tensor(items, dtype=torch.float32)
            users_tensor = torch.tensor(users, dtype=torch.float32)

            user_ids = users_tensor[:, 0].long()
            item_ids = items_tensor[:, 1].long()
            user_features = users_tensor[:, 1:]
            item_features = items_tensor[:, 1:]
            
            return self.model(user_ids, item_ids, user_features, item_features)

    def eval(self, users, items, ratings):
        self.model.eval()
        test_loss = self.criterion(self.predict(users, items), torch.tensor(ratings, dtype=torch.float32))
        print(f'Test Loss: {test_loss.item()}')
    
    def preprocess(self, items, users, ratings):
        # Encode categorical variables
        self.isbn_encoder = LabelEncoder()
        self.author_encoder = LabelEncoder()
        self.publisher_encoder = LabelEncoder()
        self.user_encoder = LabelEncoder()

        # Normalize the user-item matrix
        ratings_users_merged = ratings.merge(users, left_on='User-ID', right_on='User-ID')
        final_df = ratings_users_merged.merge(items, left_on='ISBN', right_on='ISBN')

        final_df['ISBN_encoded'] = self.isbn_encoder.fit_transform(final_df['ISBN'])
        final_df['Author_encoded'] = self.author_encoder.fit_transform(final_df['Author'])
        final_df['Publisher_encoded'] = self.publisher_encoder.fit_transform(final_df['Publisher'])
        final_df['User_ID_encoded'] = self.user_encoder.fit_transform(final_df['User-ID'])

        return final_df['Rating'], final_df[['User_ID_encoded', 'Age']], final_df[['ISBN_encoded', 'Author_encoded', 'Year', 'Publisher_encoded']]


