import matplotlib.pyplot as plt
import os
import numpy as np

def load_metrics(log_path):
    epochs, metrics = [], []
    if not os.path.exists(log_path):
        return None, None
    with open(log_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if '|' not in line or 'Epoch' in line:
                continue
            parts = line.split('|')
            try:
                epoch = int(parts[0].strip())
                val_p9 = float(parts[-1].strip())
                epochs.append(epoch)
                metrics.append(val_p9)
            except:
                continue
    return epochs, metrics

# Paths
v1_log = "2026-04-02_gat_mechanisms_study/train_v1.log"
v2_log = "2026-04-02_gat_mechanisms_study/train_v2.log"
plot_dir = "2026-04-02_gat_mechanisms_study/plots"
os.makedirs(plot_dir, exist_ok=True)

# Load
e1, m1 = load_metrics(v1_log)
e2, m2 = load_metrics(v2_log)

# Plot
plt.figure(figsize=(12, 7))

# Smoothing for better visualization
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

if m1:
    plt.plot(e1, m1, color='red', alpha=0.2, label='GATv1 (Raw)')
    plt.plot(e1, smooth(m1, 15), color='darkred', linewidth=2, label='GATv1 (Smooth)')
if m2:
    plt.plot(e2, m2, color='blue', alpha=0.2, label='GATv2 (Raw)')
    plt.plot(e2, smooth(m2, 15), color='darkblue', linewidth=2, label='GATv2 (Smooth)')

plt.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='Target 0.90')
plt.title('Attention Mechanism Comparison: GATv1 vs GATv2 (No JK, 4 layers)')
plt.xlabel('Epoch')
plt.ylabel('Val P@R0.9')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.ylim(0.4, 0.95)

out_path = os.path.join(plot_dir, "attention_comparison.png")
plt.savefig(out_path)
print(f"Comparison plot saved to {out_path}")

if m1 and m2:
    print(f"Final V1: {m1[-1]:.4f} (Max: {max(m1):.4f})")
    print(f"Final V2: {m2[-1]:.4f} (Max: {max(m2):.4f})")
