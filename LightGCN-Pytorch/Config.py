import torch
import numpy as np
import os
import requests
import zipfile
import io
import random
from time import time

class Config:
    data_dir = './ml-100k/'
    zip_url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
    data_file = 'u.data'
    seed = 2024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model Hyperparameters
    latent_dim = 64      # Embedding size
    n_layers = 3         # Number of LightGCN layers
    keep_prob = 1.0      

    # Training Hyperparameters
    lr = 1e-3            # Learning rate
    batch_size = 1024    # Batch size for training
    epochs = 1000         # Number of epochs
    weight_decay = 1e-4  # L2 regularization coefficient (for BPR loss)
    test_batch_size = 1024 # Batch size for evaluation

    # Evaluation Hyperparameters
    top_k = 20


config = Config()

random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

# 1. Data Loading and Preprocessing

def download_and_extract_movielens(url, save_dir):
    """Downloads and extracts the MovieLens 100k dataset."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Downloading MovieLens 100K dataset from {url}...")
        try:
            r = requests.get(url)
            r.raise_for_status() # Raise an exception for bad status codes
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(path=os.path.dirname(save_dir)) # Extract to parent dir
            print(f"Dataset downloaded and extracted to {save_dir}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading dataset: {e}")
            return False
        except zipfile.BadZipFile:
            print("Error: Downloaded file is not a valid zip file.")
            return False
    else:
        print(f"Dataset already exists in {save_dir}. Skipping download.")
    # Check if the expected file exists after potential extraction
    data_file_path = os.path.join(save_dir, config.data_file)
    if not os.path.exists(data_file_path):
         print(f"Error: Expected data file {config.data_file} not found in {save_dir} after download/extraction.")
         # Attempt re-extraction if directory exists but file doesn't
         if os.path.exists(save_dir) and not os.path.exists(data_file_path):
             try:
                 print("Attempting re-extraction...")
                 r = requests.get(url)
                 r.raise_for_status()
                 z = zipfile.ZipFile(io.BytesIO(r.content))
                 # Ensure extraction happens inside the target directory
                 # Extract files from 'ml-100k/' directory within the zip
                 for member in z.namelist():
                    if member.startswith('ml-100k/'):
                       z.extract(member, path=os.path.dirname(save_dir))
                 print("Re-extraction successful.")
                 if not os.path.exists(data_file_path):
                      print("Error: Data file still not found after re-extraction.")
                      return False
             except Exception as e:
                 print(f"Re-extraction failed: {e}")
                 return False

    return True
