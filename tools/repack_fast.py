import torch
import os
import glob
from multiprocessing import Pool

SRC_DIR = "data_processed/train"
DST_DIR = "data_processed_100k/train"
os.makedirs(DST_DIR, exist_ok=True)

def process_batch(args):
    batch_idx, batch_files = args
    large_data_list = []
    for f in batch_files:
        try:
            large_data_list.extend(torch.load(f, weights_only=False))
        except: continue
    
    if large_data_list:
        new_filename = f"large_chunk_{batch_idx:03d}.pt"
        # Using pickle_protocol=4 and no compression for speed
        torch.save(large_data_list, os.path.join(DST_DIR, new_filename))
        return f"Saved {new_filename}"
    return None

if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(SRC_DIR, "chunk_*.pt")))
    chunk_size = 10
    batches = [(i//chunk_size, files[i:i+chunk_size]) for i in range(0, len(files), chunk_size)]
    
    print(f"Fast Repacking with 16 workers...")
    with Pool(16) as p:
        results = p.map(process_batch, batches)
    
    print("Repacking complete!")
