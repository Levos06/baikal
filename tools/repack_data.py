import torch
import os
import glob

def repack(split, chunk_size):
    SRC_DIR = f"data_processed/{split}"
    DST_DIR = f"data_processed_100k/{split}"
    os.makedirs(DST_DIR, exist_ok=True)
    
    files = sorted(glob.glob(os.path.join(SRC_DIR, "chunk_*.pt")))
    print(f"Repacking {len(files)} files from {split}...")
    
    for i in range(0, len(files), chunk_size):
        batch_files = files[i : i + chunk_size]
        large_data_list = []
        for f in batch_files:
            try:
                large_data_list.extend(torch.load(f, weights_only=False))
            except: continue
        
        if large_data_list:
            new_filename = f"large_chunk_{i//chunk_size:03d}.pt"
            torch.save(large_data_list, os.path.join(DST_DIR, new_filename))
            print(f"Saved {DST_DIR}/{new_filename}")

if __name__ == "__main__":
    repack("train", 10) # 100k chunks
    repack("val", 10)   # 100k single chunk
    print("Repacking complete!")
