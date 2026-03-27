import torch
import os
import glob
from tqdm import tqdm

def fuse_split(split, group_size=100):
    SRC_DIR = f"data_processed/{split}"
    DST_DIR = f"data_processed_100k/{split}"
    os.makedirs(DST_DIR, exist_ok=True)
    
    files = sorted(glob.glob(os.path.join(SRC_DIR, "chunk_*.pt")))
    print(f"\nProcessing {split} split ({len(files)} files)...")
    
    pbar = tqdm(total=len(files), desc=f"Fusing {split}")
    
    for i in range(0, len(files), group_size):
        batch_files = files[i : i + group_size]
        
        all_x = []
        all_edge_index = []
        all_y = []
        x_ptrs = [0]
        edge_ptrs = [0]
        current_x_offset = 0
        current_edge_offset = 0
        
        for f in batch_files:
            try:
                data_list = torch.load(f, weights_only=False)
                for data in data_list:
                    num_nodes = data.x.size(0)
                    num_edges = data.edge_index.size(1)
                    all_x.append(data.x)
                    all_edge_index.append(data.edge_index)
                    all_y.append(data.y.view(-1)) # Ensure 1D
                    current_x_offset += num_nodes
                    current_edge_offset += num_edges
                    x_ptrs.append(current_x_offset)
                    edge_ptrs.append(current_edge_offset)
            except: continue
            pbar.update(1)
            
        if all_x:
            super_chunk = {
                'x': torch.cat(all_x, dim=0),
                'edge_index': torch.cat(all_edge_index, dim=1),
                'y': torch.cat(all_y, dim=0),
                'x_ptr': torch.tensor(x_ptrs),
                'edge_ptr': torch.tensor(edge_ptrs)
            }
            new_filename = f"super_fused_{i//group_size:03d}.pt"
            torch.save(super_chunk, os.path.join(DST_DIR, new_filename))
    
    pbar.close()

if __name__ == "__main__":
    fuse_split("train", group_size=100) 
    fuse_split("val", group_size=100)
    print("\nData Fusion Complete!")
