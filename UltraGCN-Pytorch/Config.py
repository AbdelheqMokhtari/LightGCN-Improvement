import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import os
import requests
import zipfile
import io
import random

# --- Configuration ---
class Config:
    data_dir = './ml-100k/'
    zip_url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
    data_file = 'u.data'
    seed = 2025 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model Hyperparameters
    latent_dim = 64     

    # Training Hyperparameters
    lr = 1e-3            
    batch_size = 1024    
    epochs = 1000        
    weight_decay = 1e-4  
    neg_weight = 1.0    
    neg_samples = 1      
    sampling_power = 0.8 
    test_batch_size = 1024
    # Evaluation Hyperparameters
    top_k = 20          


config = Config()

# Set Seed for Reproducibility
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

#  1. Data Loading and Preprocessing (Reusing from LightGCN example)

def download_and_extract_movielens(url, save_dir):
    """Downloads and extracts the MovieLens 100k dataset."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Downloading MovieLens 100K dataset from {url}...")
        try:
            r = requests.get(url)
            r.raise_for_status() 
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(path=os.path.dirname(save_dir)) 
            print(f"Dataset downloaded and extracted to {save_dir}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading dataset: {e}")
            return False
        except zipfile.BadZipFile:
            print("Error: Downloaded file is not a valid zip file.")
            return False
    else:
        print(f"Dataset already exists in {save_dir}. Skipping download.")

    data_file_path = os.path.join(save_dir, config.data_file)
    if not os.path.exists(data_file_path):
         print(f"Error: Expected data file {config.data_file} not found in {save_dir}.")
         if os.path.exists(save_dir):
             try:
                 print("Attempting re-extraction...")
                 r = requests.get(url)
                 r.raise_for_status()
                 z = zipfile.ZipFile(io.BytesIO(r.content))
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





