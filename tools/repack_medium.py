import torch, os, glob
from tqdm import tqdm

def repack(split, size=5):
    SRC = f"data_processed/{split}"
    DST = f"data_processed_50k/{split}"
    os.makedirs(DST, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SRC, "chunk_*.pt")))
    print(f"Repacking {split} into {DST}...")
    for i in tqdm(range(0, len(files), size)):
        batch_files = files[i:i+size]
        data = []
        for f in batch_files:
            try: data.extend(torch.load(f, weights_only=False))
            except: continue
        if data:
            torch.save(data, os.path.join(DST, f"medium_{i//size:03d}.pt"))

if __name__ == "__main__":
    repack("train", 5)
    repack("val", 5)
    print("Repacking complete!")
