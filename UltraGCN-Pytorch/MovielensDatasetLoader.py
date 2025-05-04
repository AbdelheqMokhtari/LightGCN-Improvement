
import numpy as np
import pandas as pd
import os
from collections import Counter

class MovieLensDatasetLoader:
    def __init__(self, path):
        self.path = path
        self.n_users = 0
        self.n_items = 0
        self.train_data = None
        self.test_data = None
        self.train_user_item_set = set()
        self.all_items = set()
        self.item_popularity = None # Item degrees for negative sampling

    def load(self):
        """Loads data, creates mappings, splits data, calculates popularity."""
        file_path = os.path.join(self.path, config.data_file)
        if not os.path.exists(file_path):
            print(f"Error: Data file {file_path} not found.")
            if not download_and_extract_movielens(config.zip_url, config.data_dir):
                 raise FileNotFoundError(f"Failed to download or find {file_path}")

        df = pd.read_csv(file_path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])

        self.user_map = {id: i for i, id in enumerate(df['user_id'].unique())}
        self.item_map = {id: i for i, id in enumerate(df['item_id'].unique())}
        df['user_id'] = df['user_id'].map(self.user_map)
        df['item_id'] = df['item_id'].map(self.item_map)

        self.n_users = len(self.user_map)
        self.n_items = len(self.item_map)
        self.all_items = set(range(self.n_items))

        print(f"Loaded data: {self.n_users} users, {self.n_items} items, {len(df)} interactions.")

        df = df.sort_values(by='timestamp').reset_index(drop=True)
        df['rank_latest'] = df.groupby(['user_id'])['timestamp'].rank(method='first', ascending=False)
        self.test_data = df[df['rank_latest'] == 1][['user_id', 'item_id']].to_dict('records')
        train_df = df[df['rank_latest'] > 1]
        self.train_data = train_df[['user_id', 'item_id']].to_dict('records')

        self.train_user_item_set = set((d['user_id'], d['item_id']) for d in self.train_data)

        # Calculate item popularity (degree) from training data
        item_counts = Counter(train_df['item_id'])
        self.item_popularity = np.zeros(self.n_items)
        for item_id, count in item_counts.items():
            if 0 <= item_id < self.n_items: # Check bounds
                self.item_popularity[item_id] = count
        print("Calculated item popularity.")

        print(f"Train size: {len(self.train_data)}, Test size: {len(self.test_data)}")

        return (self.n_users, self.n_items, self.train_data, self.test_data,
                self.train_user_item_set, self.all_items, self.item_popularity)
