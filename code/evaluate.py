import torch
import os
import world
from dataloader import Loader, LastFM
from model import LightGCN
from Procedure import Test

# ===== EVALUATION CONFIG =====
DATASET = "lastfm"
MODEL_PATH = r"C:\Users\mouha\OneDrive - Université de Paris\Documents\LightGCN-PyTorch\code\checkpoints\lgn-lastfm-3-64.pth.tar"
TOP_KS = [20,30]  

# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    # Verify checkpoint
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Checkpoint not found at {MODEL_PATH}")
    print(f"Loading checkpoint from {MODEL_PATH}...")

    # Configure environment
    world.dataset = DATASET
    world.tensorboard = False  # Disable tensorboard explicitly <--- KEY FIX
    
    # Load dataset
    if DATASET == 'lastfm':
        dataset = LastFM()
    else:
        dataset = Loader(path=os.path.join("../data", DATASET))
    
    # Initialize model
    Recmodel = LightGCN(world.config, dataset)
    Recmodel.load_state_dict(torch.load(MODEL_PATH, map_location=world.device))
    Recmodel.to(world.device).eval()

    # Run evaluation
    results = Test(dataset, Recmodel, epoch=0, w=None, multicore=0)  # w=None is correct
    
    # Print results
    print("\n=== FINAL RESULTS ===")
    for k in TOP_KS:
        idx = TOP_KS.index(k)
        print(f"Recall@{k}: {results['recall'][idx]:.4f}")
        print(f"NDCG@{k}: {results['ndcg'][idx]:.4f}")
        print(f"Precision@{k}:{results['precision'][idx]:.4f}")