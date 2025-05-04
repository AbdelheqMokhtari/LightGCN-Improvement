import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm
import random
import math
from time import time

class BPRTrainLoader(Dataset):
    """Dataset for BPR training: samples (user, positive_item, negative_item)."""
    def __init__(self, train_data, n_users, n_items, train_user_item_set, all_items):
        self.train_data = train_data
        self.n_users = n_users
        self.n_items = n_items
        self.train_user_item_set = train_user_item_set
        self.all_items = list(all_items) # Convert set to list for random sampling

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        user_pos_pair = self.train_data[index]
        user = user_pos_pair['user_id']
        pos_item = user_pos_pair['item_id']

        # Negative Sampling
        while True:
            neg_item = random.choice(self.all_items)
            if (user, neg_item) not in self.train_user_item_set:
                break # Found a valid negative item

        return torch.tensor(user, dtype=torch.long), \
               torch.tensor(pos_item, dtype=torch.long), \
               torch.tensor(neg_item, dtype=torch.long)

# --- 6. Evaluation ---

def recall_at_k(recommendations, ground_truth_items, k):
    """Calculate Recall@K."""
    # recommendations: list of top-k recommended item IDs
    # ground_truth_items: set of item IDs the user actually interacted with
    hits = len(set(recommendations[:k]) & ground_truth_items)
    return hits / len(ground_truth_items) if ground_truth_items else 0.0

def ndcg_at_k(recommendations, ground_truth_items, k):
    """Calculate NDCG@K."""
    # recommendations: list of top-k recommended item IDs
    # ground_truth_items: set of item IDs the user actually interacted with
    dcg = 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(ground_truth_items), k)))

    for i, item in enumerate(recommendations[:k]):
        if item in ground_truth_items:
            dcg += 1.0 / math.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0

def evaluate(model, test_data, train_user_item_set, all_items, k, batch_size):
    """Evaluate the model on the test set."""
    model.eval()
    all_recalls = []
    all_ndcgs = []

    # Build a dictionary mapping users to their ground truth items in the test set
    user_to_ground_truth = {}
    for entry in test_data:
        user = entry['user_id']
        item = entry['item_id']
        if user not in user_to_ground_truth:
            user_to_ground_truth[user] = set()
        user_to_ground_truth[user].add(item)

    users_to_evaluate = list(user_to_ground_truth.keys())
    num_batches = math.ceil(len(users_to_evaluate) / batch_size)

    with torch.no_grad():
        all_users_emb, all_items_emb = model.computer() # Compute all embeddings once

        for i in tqdm(range(num_batches), desc="Evaluating", leave=False, ncols=80):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(users_to_evaluate))
            batch_users = users_to_evaluate[start_idx:end_idx]

            # Get embeddings for the batch of users
            batch_users_torch = torch.tensor(batch_users, dtype=torch.long).to(config.device)
            batch_users_emb = all_users_emb[batch_users_torch]

            # Calculate scores for these users against all items
            # Shape: (batch_size, n_items)
            batch_ratings = torch.matmul(batch_users_emb, all_items_emb.t())

            # Process ratings for each user in the batch
            for idx, user in enumerate(batch_users):
                user_ratings = batch_ratings[idx].cpu().numpy()

                # Filter out items already seen in training
                items_to_exclude = {item for u, item in train_user_item_set if u == user}
                for item_idx in items_to_exclude:
                    # Check bounds before setting score low
                    if 0 <= item_idx < len(user_ratings):
                        user_ratings[item_idx] = -np.inf # Set score very low

                # Get top-K recommendations
                # Use argpartition for efficiency if K is small relative to N_items
                # top_k_indices = np.argpartition(user_ratings, -k)[-k:]
                # top_k_scores = user_ratings[top_k_indices]
                # recommendations = top_k_indices[np.argsort(top_k_scores)[::-1]]

                # Simpler argsort for clarity
                ranked_item_indices = np.argsort(user_ratings)[::-1]
                recommendations = ranked_item_indices[:k]

                ground_truth = user_to_ground_truth[user]

                # Calculate metrics
                recall = recall_at_k(recommendations, ground_truth, k)
                ndcg = ndcg_at_k(recommendations, ground_truth, k)
                all_recalls.append(recall)
                all_ndcgs.append(ndcg)

    mean_recall = np.mean(all_recalls)
    mean_ndcg = np.mean(all_ndcgs)
    return mean_recall, mean_ndcg