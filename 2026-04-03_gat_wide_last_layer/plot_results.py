import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training(log_path, output_path):
    epochs = []
    train_loss, val_loss = [], []
    train_p9, val_p9 = [], []
    
    with open(log_path, 'r') as f:
        for line in f:
            parts = line.split('|')
            if len(parts) < 12: continue
            try:
                epochs.append(int(parts[0]))
                train_loss.append(float(parts[4]))
                train_p9.append(float(parts[7]))
                val_loss.append(float(parts[8]))
                val_p9.append(float(parts[11]))
            except:
                continue

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Train Loss', alpha=0.6)
    plt.plot(epochs, val_loss, label='Val Loss', alpha=0.6)
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_p9, label='Train P@R0.9', alpha=0.6)
    plt.plot(epochs, val_p9, label='Val P@R0.9', alpha=0.6)
    plt.title('P@R0.9')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot_training('2026-04-03_gat_wide_last_layer/train_1536.log', '2026-04-03_gat_wide_last_layer/training_plot.png')
