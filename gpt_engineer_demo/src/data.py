import pandas as pd
import numpy as np
from torch.utils.data import Dataset


# Generate a fake ratings matrix
def generate_ratings(n_rows=1000, n_users=1000, n_items=1000):
    user_ids = np.random.randint(0, n_users, n_rows)
    item_ids = np.random.randint(0, n_items, n_rows)
    ratings = np.random.uniform(0.0, 5.0, n_rows)

    ratings_df = pd.DataFrame({
        'user_id': user_ids,
        'item_id': item_ids,
        'rating': ratings
    })

    return ratings_df


class RatingsDataset(Dataset):
    def __init__(self, ratings_df):
        self.users = ratings_df['user_id'].values
        self.items = ratings_df['item_id'].values
        self.ratings = ratings_df['rating'].values

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]
