import torch.optim as optim
import Config
import LightGCN
import MovieLensDatasetLoader
import BPRTrainLoader
import evaluate
import normalize_graph_mat
import convert_sp_mat_to_sp_tensor
import bpr_loss
from torch.utils.data import DataLoader

from tqdm import tqdm
from time import time


if __name__ == '__main__':
    # 1. Load and prepare data
    print("--- Loading Data ---")
    data_loader = MovieLensDatasetLoader(config.data_dir)
    adj_mat, n_users, n_items, train_data, test_data, train_user_item_set, all_items = data_loader.load()

    # 2. Build and normalize graph
    print("\n--- Building Graph ---")
    norm_adj_mat = normalize_graph_mat(adj_mat)
    sparse_norm_adj = convert_sp_mat_to_sp_tensor(norm_adj_mat)

    # 3. Create model and optimizer
    print("\n--- Initializing Model ---")
    model = LightGCN(n_users, n_items, config.latent_dim, config.n_layers, sparse_norm_adj).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # 4. Create DataLoader for training
    train_dataset = BPRTrainLoader(train_data, n_users, n_items, train_user_item_set, all_items)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4) # Adjust num_workers based on your system

    # 5. Training Loop
    print("\n--- Starting Training ---")
    best_recall = 0.0
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        start_time = time()

        # Wrap train_loader with tqdm for progress bar
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}", leave=False, ncols=100)

        for users, pos_items, neg_items in train_iterator:
            users = users.to(config.device)
            pos_items = pos_items.to(config.device)
            neg_items = neg_items.to(config.device)

            optimizer.zero_grad()

            # Forward pass
            users_emb, pos_emb, neg_emb, users_emb_init, pos_emb_init, neg_emb_init = model(users, pos_items, neg_items)

            # Calculate loss
            loss = bpr_loss(users_emb, pos_emb, neg_emb,
                            users_emb_init, pos_emb_init, neg_emb_init,
                            config.weight_decay) # Pass weight decay here

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Update tqdm description with current avg loss
            train_iterator.set_postfix(loss=epoch_loss/len(train_iterator))


        epoch_time = time() - start_time
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch}/{config.epochs} [{epoch_time:.1f}s] - Loss: {avg_loss:.4f}")

        # 6. Evaluation (evaluate every few epochs or at the end)
        if epoch % 5 == 0 or epoch == config.epochs: # Evaluate every 5 epochs and the last one
            print(f"--- Evaluating Epoch {epoch} ---")
            recall, ndcg = evaluate(model, test_data, train_user_item_set, all_items, config.top_k, config.test_batch_size)
            print(f"Epoch {epoch} - Recall@{config.top_k}: {recall:.4f}, NDCG@{config.top_k}: {ndcg:.4f}")

            # Simple best model saving based on Recall
            if recall > best_recall:
                best_recall = recall
                # torch.save(model.state_dict(), 'best_lightgcn_model.pth')
                print(f"✨ New best Recall@{config.top_k}: {best_recall:.4f}. Model state could be saved.")
            print("-" * 30)

    print("\n--- Training Complete ---")