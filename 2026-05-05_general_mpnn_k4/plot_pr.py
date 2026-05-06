import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
import numpy as np
import os
import sys

# Import model and dataset from the training script
sys.path.append('/home/levos/experiments/2026-05-05_general_mpnn_k4')
from train_general_mpnn import MPNN_FCLast, MediumDataset, add_features_21_norm, DEVICE, BATCH_SIZE, NUM_WORKERS

def plot_pr_curve():
    print(f"Loading model on {DEVICE}...")
    model = MPNN_FCLast(21, 2).to(DEVICE)
    
    checkpoint_path = "/home/levos/experiments/2026-05-05_general_mpnn_k4/checkpoints/model_mpnn_k4_1000.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()

    print("Loading validation data...")
    # Use more batches for a smoother curve
    val_loader = DataLoader(MediumDataset('val', k=4), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    
    all_labels = []
    all_probs = []
    
    num_batches_to_eval = 200 # Evaluate on 200 batches for a good representative sample
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= num_batches_to_eval:
                break
            batch = add_features_21_norm(batch.to(DEVICE))
            out = model(batch.x, batch.edge_index, batch.batch)
            probs = F.softmax(out, dim=1)[:, 1]
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{num_batches_to_eval} batches...")

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    print("Calculating PR curve...")
    precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall, precision)
    
    # Calculate P@R0.9
    p_at_r09 = np.interp(0.9, recall[::-1], precision[::-1])
    
    print(f"PR AUC: {pr_auc:.4f}")
    print(f"P@R0.9: {p_at_r09:.4f}")

    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='b', lw=2, label=f'General MPNN k=4 (AUC = {pr_auc:.4f})')
    
    # Highlight P@R0.9 point
    plt.plot(0.9, p_at_r09, 'ro', markersize=8, label=f'P@R=0.9 ({p_at_r09:.4f})')
    plt.axvline(x=0.9, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=p_at_r09, color='r', linestyle='--', alpha=0.5)
    
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curve: General MPNN (k=4)', fontsize=16)
    plt.legend(loc='lower left', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Set axis limits
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    
    save_path = "/home/levos/experiments/2026-05-05_general_mpnn_k4/pr_curve.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"PR curve saved to {save_path}")

if __name__ == "__main__":
    plot_pr_curve()
