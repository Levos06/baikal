import h5py
import numpy as np
import os

FILE_PATH = "/home3/ivkhar/Baikal/data/normed/baikal_mc2020_multi_split_0924mid_eq_norm.h5"

def calculate_snr_for_split(f, split):
    dataset_path = f"{split}/labels/data"
    if dataset_path not in f:
        print(f"Dataset {dataset_path} not found.")
        return None

    ds = f[dataset_path]
    total_size = ds.shape[0]
    chunk_size = 10**7
    
    counts = {}
    
    print(f"Processing {split} ({total_size} hits)...")
    for i in range(0, total_size, chunk_size):
        end = min(i + chunk_size, total_size)
        data_chunk = ds[i:end]
        unique, chunk_counts = np.unique(data_chunk, return_counts=True)
        for u, c in zip(unique, chunk_counts):
            counts[u] = counts.get(u, 0) + c
        
    noise_count = counts.get(0, 0)
    signal_count = sum(c for val, c in counts.items() if val != 0)
    
    snr = signal_count / noise_count if noise_count > 0 else float('inf')
    
    return {
        "counts": counts,
        "noise": noise_count,
        "signal": signal_count,
        "snr": snr,
        "total": total_size
    }

def main():
    if not os.path.exists(FILE_PATH):
        print(f"Error: File {FILE_PATH} not found.")
        return

    with h5py.File(FILE_PATH, 'r') as f:
        splits = ['train', 'test', 'val']
        results = {}
        
        for split in splits:
            res = calculate_snr_for_split(f, split)
            if res:
                results[split] = res
                print(f"Results for {split}:")
                print(f"  All label counts: {res['counts']}")
                print(f"  Signal hits (val != 0): {res['signal']:,}")
                print(f"  Noise hits (val == 0):  {res['noise']:,}")
                print(f"  SNR:         {res['snr']:.6f}")
                print(f"  Signal %:    {(res['signal']/res['total'])*100:.4f}%")
                print("-" * 30)

if __name__ == "__main__":
    main()
