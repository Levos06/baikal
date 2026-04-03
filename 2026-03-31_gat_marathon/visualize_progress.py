import matplotlib.pyplot as plt
import re
import os

epochs, metrics = [], []
log_path = '2026-03-31_gat_marathon/plot_data.txt'
if not os.path.exists(log_path):
    print(f"Error: {log_path} not found")
    exit(1)

with open(log_path, 'r') as f:
    for line in f:
        match = re.search(r'Epoch (\d+) .* Val P@R0.9: ([\d.]+)', line)
        if match:
            epochs.append(int(match.group(1)))
            metrics.append(float(match.group(2)))

if not epochs:
    print("No data found to plot")
    exit(1)

plt.figure(figsize=(10, 6))
plt.plot(epochs, metrics, label='Val P@R0.9 (Batch 512)', color='blue', alpha=0.7)

# Smoothing
if len(metrics) > 10:
    import numpy as np
    smooth_metrics = np.convolve(metrics, np.ones(10)/10, mode='valid')
    plt.plot(epochs[9:], smooth_metrics, label='Trend (SMA 10)', color='darkblue', linewidth=2)

plt.axhline(y=0.9, color='red', linestyle='--', label='Target 0.90')
plt.axhline(y=max(metrics), color='green', linestyle=':', label=f'Max: {max(metrics):.4f}')

plt.title('GATv2 Marathon: Precision @ Recall 0.90')
plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

os.makedirs('2026-03-31_gat_marathon/plots', exist_ok=True)
plt.savefig('2026-03-31_gat_marathon/plots/training_progress.png')
print(f"Plot saved to 2026-03-31_gat_marathon/plots/training_progress.png. Max metric: {max(metrics):.4f}")
