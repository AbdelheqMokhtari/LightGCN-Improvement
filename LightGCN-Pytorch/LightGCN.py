import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from time import time

class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, latent_dim, n_layers, graph):
        super(LightGCN, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.graph = graph # Normalized sparse adjacency matrix

        # Initialize embeddings
        self.embedding_user = nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim)
        self.embedding_item = nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim)

        # Xavier initialization
        nn.init.xavier_uniform_(self.embedding_user.weight)
        nn.init.xavier_uniform_(self.embedding_item.weight)
        print("Embeddings initialized.")

        # Used for dropout (optional, often not used in LightGCN propagation)
        # self.dropout = nn.Dropout(1 - config.keep_prob)

    def computer(self):
        """Propagate embeddings through the graph."""
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb] # List to store embeddings at each layer

        # Graph propagation
        g_droped = self.graph # In LightGCN, dropout is usually applied on the graph itself if used

        for layer in range(self.n_layers):
            # Matrix multiplication: G * E_l
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)

        # Combine layer embeddings (mean pooling)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)

        users, items = torch.split(light_out, [self.n_users, self.n_items])
        return users, items

    def get_users_ratings(self, users):
        """Calculate scores for given users against all items."""
        all_users, all_items = self.computer()
        users_emb = all_users[users] # Get embeddings for specific users
        items_emb = all_items        # All item embeddings

        # Calculate dot product scores
        rating = torch.matmul(users_emb, items_emb.t())
        return rating

    def forward(self, users, pos_items, neg_items):
        """Compute embeddings and return embeddings for BPR loss."""
        all_users, all_items = self.computer()

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        # Get initial embeddings (0th layer) for regularization
        users_emb_initial = self.embedding_user(users)
        pos_emb_initial = self.embedding_item(pos_items)
        neg_emb_initial = self.embedding_item(neg_items)

        return users_emb, pos_emb, neg_emb, users_emb_initial, pos_emb_initial, neg_emb_initial

# --- 4. BPR Loss and Sampling ---

def bpr_loss(users_emb, pos_emb, neg_emb, users_emb_initial, pos_emb_initial, neg_emb_initial, weight_decay):
    """Calculate BPR loss with L2 regularization."""
    # BPR loss term
    pos_scores = torch.sum(users_emb * pos_emb, dim=1)
    neg_scores = torch.sum(users_emb * neg_emb, dim=1)

    bpr_loss_term = torch.mean(F.softplus(neg_scores - pos_scores))

    # L2 Regularization term (applied to initial embeddings)
    reg_loss_term = (users_emb_initial.norm(2).pow(2) +
                     pos_emb_initial.norm(2).pow(2) +
                     neg_emb_initial.norm(2).pow(2)) / 2.0 / users_emb.shape[0] 

    total_loss = bpr_loss_term + weight_decay * reg_loss_term
    return total_loss