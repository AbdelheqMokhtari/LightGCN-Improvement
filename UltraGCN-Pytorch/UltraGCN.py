import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
from collections import Counter

class UltraGCN(nn.Module):
    def __init__(self, n_users, n_items, latent_dim):
        super(UltraGCN, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.latent_dim = latent_dim

        # Initialize embeddings
        self.embedding_user = nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim)
        self.embedding_item = nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim)

        # Xavier initialization
        nn.init.xavier_uniform_(self.embedding_user.weight)
        nn.init.xavier_uniform_(self.embedding_item.weight)
        print("Embeddings initialized.")

    def get_embeddings(self):
        """Returns the current user and item embeddings."""
        return self.embedding_user.weight, self.embedding_item.weight

    def forward(self, users, pos_items, neg_items):
        """Retrieves embeddings for the given users/items."""
        user_emb = self.embedding_user(users)
        pos_item_emb = self.embedding_item(pos_items)
        neg_item_emb = self.embedding_item(neg_items) # Shape: (batch_size * neg_samples, dim)

        return user_emb, pos_item_emb, neg_item_emb

    def get_users_ratings(self, users):
        """Calculate scores for given users against all items using direct embeddings."""
        user_emb = self.embedding_user(users) # Shape: (batch_size, dim)
        item_emb = self.embedding_item.weight # Shape: (n_items, dim)

        # Calculate dot product scores
        rating = torch.matmul(user_emb, item_emb.t()) # Shape: (batch_size, n_items)
        return rating

#3. Loss Function and Sampling 

def ultragcn_loss(user_emb, pos_item_emb, neg_item_emb, neg_weight):
    """
    Calculate the BPR loss for UltraGCN.
    """
    batch_size = user_emb.shape[0]
    num_neg_samples = neg_item_emb.shape[0] // batch_size # Should match config.neg_samples

    # Positive scores: (batch_size,)
    pos_scores = torch.sum(user_emb * pos_item_emb, dim=1)


    user_emb_expanded = user_emb.unsqueeze(1).expand(-1, num_neg_samples, -1).reshape(-1, user_emb.shape[1])

    neg_scores = torch.sum(user_emb_expanded * neg_item_emb, dim=1)

    # Calculate loss terms

    positive_loss = F.softplus(-pos_scores) # -log sigmoid(pos_scores)

    negative_loss = F.softplus(neg_scores).view(batch_size, num_neg_samples) # -log sigmoid(-neg_scores)

    negative_loss = torch.mean(negative_loss, dim=1)

    # Combine positive and weighted negative loss
    loss = torch.mean(positive_loss + neg_weight * negative_loss)

    return loss