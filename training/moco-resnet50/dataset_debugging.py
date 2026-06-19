# src/debug_dataset.py
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import TwoViewNumpyDataset
from transforms import get_view_transforms
from utils import load_config

def debug_dataset():
    # Real data
    real_path = "c:/MTI Research/Preprocessing_CPU/train_10014_a_1.npy"
    if not os.path.exists(real_path):
        print(f"Error: Real file {real_path} not found.")
        return

    items = [{'path': real_path, 'volume_name': 'train_10014_a_1', 'patient_id': 'p1', 'label': 0}]
    
    # Load transforms
    # User changed roi_size to (128, 256, 256)
    viewA, viewB = get_view_transforms(roi_size=(128, 256, 256))
    
    dataset = TwoViewNumpyDataset(items * 10, transform_q=viewA, transform_k=viewB)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    
    print("Testing dataset loading with transforms (Batch Size 4)...")
    try:
        for i, batch in enumerate(dataloader):
            img_q = batch['img_q']
            img_k = batch['img_k']
            print(f"Sample {i}:")
            print(f"  img_q shape: {img_q.shape}")
            print(f"  img_k shape: {img_k.shape}")
            
            if img_q.shape != img_k.shape:
                print("  MISMATCH DETECTED!")
            else:
                print("  Shapes match.")
                
    except Exception as e:
        print(f"Error during loading: {e}")
    finally:
        pass

if __name__ == "__main__":
    debug_dataset()
