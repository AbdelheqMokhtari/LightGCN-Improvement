import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm 
import os
from time import time

class MovieLensDatasetLoader:
    def __init__(self, path):
        self.path = path
        self.n_users = 0
        self.n_items = 0
        self.train_data = None
        self.test_data = None
        self.train_user_item_set = set()
        self.all_items = set()
        self.user_item_matrix = None # Adjacency matrix

    def load(self):
        """Loads data, creates mappings, splits data, and builds interaction matrix."""
        file_path = os.path.join(self.path, config.data_file)
        if not os.path.exists(file_path):
            print(f"Error: Data file {file_path} not found.")
            if not download_and_extract_movielens(config.zip_url, config.data_dir):
                 raise FileNotFoundError(f"Failed to download or find {file_path}")

        # Load data
        df = pd.read_csv(file_path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])

        # Remap user and item IDs to start from 0
        self.user_map = {id: i for i, id in enumerate(df['user_id'].unique())}
        self.item_map = {id: i for i, id in enumerate(df['item_id'].unique())}
        df['user_id'] = df['user_id'].map(self.user_map)
        df['item_id'] = df['item_id'].map(self.item_map)

        self.n_users = len(self.user_map)
        self.n_items = len(self.item_map)
        self.all_items = set(range(self.n_items))

        print(f"Loaded data: {self.n_users} users, {self.n_items} items, {len(df)} interactions.")

        # Sort by timestamp for splitting (optional, could use random split)
        df = df.sort_values(by='timestamp').reset_index(drop=True)

        # Simple split: Use last interaction of each user for testing (leave-one-out style)
        # More robust splits exist (e.g., temporal split, k-fold)
        df['rank_latest'] = df.groupby(['user_id'])['timestamp'].rank(method='first', ascending=False)
        self.test_data = df[df['rank_latest'] == 1][['user_id', 'item_id']].to_dict('records')
        self.train_data = df[df['rank_latest'] > 1][['user_id', 'item_id']].to_dict('records')

        # Create train user-item set for negative sampling and evaluation filtering
        self.train_user_item_set = set((d['user_id'], d['item_id']) for d in self.train_data)

        print(f"Train size: {len(self.train_data)}, Test size: {len(self.test_data)}")

        # Build the user-item interaction matrix (adjacency matrix) for the graph
        rows = [d['user_id'] for d in self.train_data]
        cols = [d['item_id'] for d in self.train_data]
        data = np.ones(len(rows))

        # User-item interaction matrix R (n_users x n_items)
        R = sp.csr_matrix((data, (rows, cols)), shape=(self.n_users, self.n_items))

        # Create the full adjacency matrix A = [[0, R], [R.T, 0]]
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil() # Efficient for item assignment
        R = R.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print("Adjacency matrix created.")

        self.user_item_matrix = adj_mat
        return self.user_item_matrix, self.n_users, self.n_items, self.train_data, self.test_data, self.train_user_item_set, self.all_items

# --- 2. Graph Construction and Normalization ---

def normalize_graph_mat(adj_mat):
    """Compute the symmetrically normalized adjacency matrix."""
    print("Normalizing adjacency matrix...")
    adj_mat = adj_mat.tocsr()
    rowsum = np.array(adj_mat.sum(axis=1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    # A_norm = D^{-1/2} * A * D^{-1/2}
    norm_adj_mat = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt)
    print("Normalization complete.")
    return norm_adj_mat.tocoo() # Convert to COO format for PyTorch

def convert_sp_mat_to_sp_tensor(X):
    """Convert a SciPy sparse matrix to a PyTorch sparse tensor."""
    coo = X.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64))
    values = torch.from_numpy(coo.data)
    shape = torch.Size(coo.shape)
    return torch.sparse_coo_tensor(indices, values, shape, device=config.device)
