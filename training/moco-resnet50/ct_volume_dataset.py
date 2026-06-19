# src/dataset.py
import ast
import os
import numpy as np
import copy
import random
from torch.utils.data import Dataset
from monai.transforms import Compose, EnsureTyped, ResizeWithPadOrCropd


class TwoViewNumpyDataset(Dataset):
    """
    Expects items with keys: 'volume_name', 'path', 'patient_id', 'label'
    Returns a dict with: img_q, img_k, patient_id, label
    """
    def __init__(self, items, transform_q, transform_k):
        self.items = items
        self.transform_q = transform_q
        self.transform_k = transform_k

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        
        row = self.items[idx]
        
        path = row['paths']
        
        # p = ast.literal_eval(path) if isinstance(path, str) else list(path)
        
        # p = random.choice(p_list) if isinstance(p_list, list) else p_list
        
        arr = np.load(path).astype(np.float32)  # expected shape (1,128,256,256)
        
        sample = {
            'img': arr,
            'volumeName': row.get('volumeName'),
            # 'patient_id': row.get('patient_id'),
            # 'label': row.get('label', None),
        }
        
        preprocess = Compose([
            EnsureTyped(keys=["img"]),
            ResizeWithPadOrCropd(keys=["img"], spatial_size=(32, 256, 256)),
        ])
        
        if preprocess is not None:
            sample = preprocess(sample)
        # IMPORTANT: use deep copies to keep views independent
        q = self.transform_q(copy.deepcopy(sample))['img']
        k = self.transform_k(copy.deepcopy(sample))['img']

        return {
            'img_q': q,
            'img_k': k,
            # 'patient_id': row.get('patient_id', "unknown"),
            # 'label': row.get('label', -1) if row.get('label') is not None else -1
        }
