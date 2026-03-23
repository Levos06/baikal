import h5py
import numpy as np
import torch
from torch_geometric.data import Data
import os
from tqdm import tqdm

H5_FILE = "/home3/ivkhar/Baikal/data/normed/baikal_mc2020_multi_split_0924mid_eq_norm.h5"
OUTPUT_DIR = "data_processed"
CHUNK_SIZE = 10000
K_NEIGHBORS = 2

def get_edges_fast(n, k=2):
    if n <= 1: return torch.zeros((2, 0), dtype=torch.long)
    rows = []
    cols = []
    for i in range(1, k + 1):
        # Вперед по времени
        rows.append(np.arange(0, n - i))
        cols.append(np.arange(i, n))
        # Назад по времени
        rows.append(np.arange(i, n))
        cols.append(np.arange(0, n - i))
    return torch.from_numpy(np.array([np.concatenate(rows), np.concatenate(cols)])).long()

def process_split(split='train', num_events=None):
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)
    
    with h5py.File(H5_FILE, 'r') as f:
        starts = f[f'{split}/ev_starts/data']
        total = len(starts) - 1
        if num_events: total = min(total, num_events)
        
        data_ptr = 0
        chunk_data = []
        chunk_idx = 0
        
        print(f"Processing {split} split ({total} events)...")
        for i in tqdm(range(total)):
            s, e = starts[i], starts[i+1]
            x = torch.from_numpy(f[f'{split}/data/data'][s:e]).float()
            y = torch.from_numpy((f[f'{split}/labels/data'][s:e] != 0).astype(np.int64))
            
            edge_index = get_edges_fast(e-s, K_NEIGHBORS)
            chunk_data.append(Data(x=x, edge_index=edge_index, y=y))
            
            if len(chunk_data) >= CHUNK_SIZE:
                torch.save(chunk_data, os.path.join(OUTPUT_DIR, split, f"chunk_{chunk_idx}.pt"))
                chunk_data = []
                chunk_idx += 1
        
        # Save last chunk
        if chunk_data:
            torch.save(chunk_data, os.path.join(OUTPUT_DIR, split, f"chunk_{chunk_idx}.pt"))

if __name__ == "__main__":
    # Начинаем с валидации (она меньше), чтобы проверить скорость
    process_split('val', num_events=100000)
    # Затем весь тренировочный сет
    process_split('train', num_events=None)
