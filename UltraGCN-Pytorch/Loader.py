class UltraGCNTrainLoader(Dataset):
    """
    Dataset for UltraGCN training: samples (user, positive_item, negative_item)
    using popularity-based negative sampling.
    """
    def __init__(self, train_data, n_users, n_items, train_user_item_set,
                 item_popularity, num_neg_samples, power):
        self.train_data = train_data
        self.n_users = n_users
        self.n_items = n_items
        self.train_user_item_set = train_user_item_set
        self.num_neg_samples = num_neg_samples

        # Calculate sampling probabilities based on popularity^power
        # Add small epsilon to avoid zero probability for unpopular items
        probs = np.power(item_popularity + 1e-8, power)
        self.sampling_probs = probs / np.sum(probs)
        self.all_item_indices = np.arange(n_items)

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        user_pos_pair = self.train_data[index]
        user = user_pos_pair['user_id']
        pos_item = user_pos_pair['item_id']

        # Negative Sampling based on popularity
        neg_items = []
        while len(neg_items) < self.num_neg_samples:
            # Sample items based on precomputed probabilities
            sampled_items = np.random.choice(self.all_item_indices,
                                             size=self.num_neg_samples - len(neg_items), # Sample needed amount
                                             p=self.sampling_probs,
                                             replace=True) # Allow duplicates temporarily

            for neg_item in sampled_items:
                if (user, neg_item) not in self.train_user_item_set:
                    neg_items.append(neg_item)
                    if len(neg_items) == self.num_neg_samples:
                        break # Got enough valid samples

        # Ensure we return exactly num_neg_samples
        if len(neg_items) != self.num_neg_samples:
             # Fallback or raise error - this indicates potential issues if sampling often fails
             # For simplicity, let's resample until we get enough, but this could be slow
             # A better approach might be to draw more samples initially.
             while len(neg_items) < self.num_neg_samples:
                 neg_item = np.random.choice(self.all_item_indices, p=self.sampling_probs)
                 if (user, neg_item) not in self.train_user_item_set:
                    neg_items.append(neg_item)


        return torch.tensor(user, dtype=torch.long), \
               torch.tensor(pos_item, dtype=torch.long), \
               torch.tensor(neg_items, dtype=torch.long) # Return list/tensor of negative items


# --- 4. Evaluation (Reusing from LightGCN example) ---
# (Identical recall_at_k, ndcg_at_k, evaluate functions as before)
# Make sure these functions are defined here or imported

def recall_at_k(recommendations, ground_truth_items, k):
    """Calculate Recall@K."""
    hits = len(set(recommendations[:k]) & ground_truth_items)
    return hits / len(ground_truth_items) if ground_truth_items else 0.0

def ndcg_at_k(recommendations, ground_truth_items, k):
    """Calculate NDCG@K."""
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
        # Get all embeddings ONCE before evaluation loop if model doesn't change
        # For UltraGCN, embeddings are directly used
        all_user_emb, all_item_emb = model.get_embeddings()

        for i in tqdm(range(num_batches), desc="Evaluating", leave=False, ncols=80):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(users_to_evaluate))
            batch_users = users_to_evaluate[start_idx:end_idx]

            batch_users_torch = torch.tensor(batch_users, dtype=torch.long).to(config.device)

            # Calculate scores using the model's rating method
            # For UltraGCN, this uses the direct embeddings
            batch_ratings = model.get_users_ratings(batch_users_torch) # Shape: (batch_size, n_items)
            batch_ratings = batch_ratings.cpu().numpy()

            for idx, user in enumerate(batch_users):
                user_ratings = batch_ratings[idx]
                items_to_exclude = {item for u, item in train_user_item_set if u == user}
                for item_idx in items_to_exclude:
                    if 0 <= item_idx < len(user_ratings):
                        user_ratings[item_idx] = -np.inf

                ranked_item_indices = np.argsort(user_ratings)[::-1]
                recommendations = ranked_item_indices[:k]
                ground_truth = user_to_ground_truth[user]

                recall = recall_at_k(recommendations, ground_truth, k)
                ndcg = ndcg_at_k(recommendations, ground_truth, k)
                all_recalls.append(recall)
                all_ndcgs.append(ndcg)

    mean_recall = np.mean(all_recalls)
    mean_ndcg = np.mean(all_ndcgs)
    return mean_recall, mean_ndcg