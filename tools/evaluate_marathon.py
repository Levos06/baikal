import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn.metrics import precision_score, recall_score, precision_recall_curve
import glob
import os
from tqdm import tqdm

DATA_DIR = "data_processed/val"
CHECKPOINT = "2026-03-19_final_marathon/checkpoints/model_res_marathon_2000.pt"
THRESHOLD = 0.5

class ResGCN_5Layer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 512)
        self.conv2 = GCNConv(512, 1024)
        self.conv3 = GCNConv(1024, 1024)
        self.conv4 = GCNConv(1024, 512)
        self.conv5 = GCNConv(512, out_channels)
        self.proj1 = torch.nn.Linear(in_channels, 512)
        self.proj2 = torch.nn.Linear(512, 1024)
        self.proj3 = torch.nn.Linear(1024, 1024)
        self.proj4 = torch.nn.Linear(1024, 512)

    def forward(self, x, edge_index):
        x = F.gelu(self.conv1(x, edge_index) + self.proj1(x))
        x = F.gelu(self.conv2(x, edge_index) + self.proj2(x))
        x = F.gelu(self.conv3(x, edge_index) + self.proj3(x))
        x = F.gelu(self.conv4(x, edge_index) + self.proj4(x))
        return self.conv5(x, edge_index)

def get_precision_at_recall(labels, probs, target_recall=0.9):
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    if len(recall) == 0 or np.max(recall) < target_recall: return 0.0
    return np.interp(target_recall, recall[::-1], precision[::-1])

def run_eval():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResGCN_5Layer(5, 2).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, weights_only=True))
    model.eval()

    files = sorted(glob.glob(os.path.join(DATA_DIR, "chunk_*.pt")))
    all_labels, all_probs = [], []

    print(f"Evaluating {len(files)} chunks from {DATA_DIR}...")
    with torch.no_grad():
        for file in tqdm(files):
            data_list = torch.load(file, weights_only=False)
            loader = DataLoader(data_list, batch_size=256)
            for batch in loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index)
                probs = F.softmax(out, dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())

    all_labels, all_probs = np.array(all_labels), np.array(all_probs)
    preds = (all_probs > THRESHOLD).astype(int)
    
    prec = precision_score(all_labels, preds, zero_division=0)
    rec = recall_score(all_labels, preds, zero_division=0)
    p_at_r9 = get_precision_at_recall(all_labels, all_probs, 0.9)

    print("\n--- FINAL MARATHON EVALUATION ---")
    print(f"Model: {CHECKPOINT}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"P@R0.9:    {p_at_r9:.4f}")

if __name__ == "__main__":
    run_eval()
