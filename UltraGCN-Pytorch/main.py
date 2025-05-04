if __name__ == '__main__':
    # 1. Load and prepare data
    print("--- Loading Data ---")
    data_loader = MovieLensDatasetLoader(config.data_dir)
    (n_users, n_items, train_data, test_data,
     train_user_item_set, all_items, item_popularity) = data_loader.load()

    # 2. Create model and optimizer
    print("\n--- Initializing Model ---")
    model = UltraGCN(n_users, n_items, config.latent_dim).to(config.device)
    # Use AdamW or add weight decay directly to Adam
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # 3. Create DataLoader for training
    train_dataset = UltraGCNTrainLoader(train_data, n_users, n_items, train_user_item_set,
                                        item_popularity, config.neg_samples, config.sampling_power)
    # Use collate_fn to handle the variable number of negative samples if needed,
    # but here we ensure neg_samples is fixed.
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)

    # 4. Training Loop
    print("\n--- Starting Training ---")
    best_recall = 0.0
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        start_time = time()

        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}", leave=False, ncols=100)

        for users, pos_items, neg_items_batch in train_iterator:
            users = users.to(config.device)
            pos_items = pos_items.to(config.device)
            # neg_items_batch shape: (batch_size, neg_samples)
            # Reshape negative items for embedding lookup: (batch_size * neg_samples,)
            neg_items = neg_items_batch.view(-1).to(config.device)

            optimizer.zero_grad()

            # Forward pass: Get embeddings
            user_emb, pos_item_emb, neg_item_emb = model(users, pos_items, neg_items)

            # Calculate loss (L2 reg handled by optimizer)
            loss = ultragcn_loss(user_emb, pos_item_emb, neg_item_emb, config.neg_weight)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            train_iterator.set_postfix(loss=epoch_loss/len(train_iterator))

        epoch_time = time() - start_time
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch}/{config.epochs} [{epoch_time:.1f}s] - Loss: {avg_loss:.4f}")

        # 5. Evaluation
        if epoch % 5 == 0 or epoch == config.epochs:
            print(f"--- Evaluating Epoch {epoch} ---")
            recall, ndcg = evaluate(model, test_data, train_user_item_set, all_items, config.top_k, config.test_batch_size)
            print(f"Epoch {epoch} - Recall@{config.top_k}: {recall:.4f}, NDCG@{config.top_k}: {ndcg:.4f}")

            if recall > best_recall:
                best_recall = recall
                # torch.save(model.state_dict(), 'best_ultragcn_model.pth')
                print(f"✨ New best Recall@{config.top_k}: {best_recall:.4f}. Model state could be saved.")
            print("-" * 30)

    print("\n--- Training Complete ---")
    # Optional final evaluation with best saved model
    # model.load_state_dict(torch.load('best_ultragcn_model.pth'))
    # final_recall, final_ndcg = evaluate(model, test_data, train_user_item_set, all_items, config.top_k, config.test_batch_size)
    # print(f"\nFinal Evaluation (Best Model) - Recall@{config.top_k}: {final_recall:.4f}, NDCG@{config.top_k}: {final_ndcg:.4f}")