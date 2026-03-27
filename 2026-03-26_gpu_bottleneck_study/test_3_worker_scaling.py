import torch
from torch_geometric.loader import DataLoader
import os
import time
import glob
import math

DATA_DIR = "data_processed"
BATCH_SIZE = 128

class ChunkedDataset(torch.utils.data.IterableDataset):
    def __init__(self, split='train'):
        super().__init__()
        self.files = sorted(glob.glob(os.path.join(DATA_DIR, split, "chunk_*.pt")))[:100]
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None: iter_files = self.files
        else:
            per_worker = int(math.ceil(len(self.files) / float(worker_info.num_workers)))
            wid = worker_info.id
            iter_files = self.files[wid * per_worker : (wid + 1) * per_worker]
        for f in iter_files:
            try:
                data_list = torch.load(f, weights_only=False)
                for data in data_list: yield data
            except: continue

def run_test(num_workers):
    dataset = ChunkedDataset()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=num_workers, pin_memory=True)
    start = time.time()
    count = 0
    for batch in loader:
        count += batch.num_graphs
        if count >= 30000: break
    duration = time.time() - start
    return count / duration

if __name__ == "__main__":
    results = {}
    for nw in [0, 2, 4, 8]:
        print(f"Testing num_workers={nw}...")
        results[nw] = run_test(nw)
        print(f"  Throughput: {results[nw]:.2f} events/s")
    
    print("\nScaling Results:")
    for nw, tp in results.items():
        print(f"Workers {nw:2d}: {tp:8.2f} events/s")
