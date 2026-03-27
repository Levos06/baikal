import matplotlib.pyplot as plt
import re
import os
import numpy as np

def parse_log(log_path):
    data = {
        'epoch': [],
        't_loss': [], 't_prec': [], 't_rec': [], 't_p9': [],
        'v_loss': [], 'v_prec': [], 'v_rec': [], 'v_p9': []
    }
    
    current_epoch = None
    with open(log_path, 'r') as f:
        for line in f:
            # Epoch match: Epoch 0123
            e_match = re.search(r"Epoch (\d+)", line)
            if e_match:
                current_epoch = int(e_match.group(1))
            
            if current_epoch is None: continue

            # Robust parsing using labels
            if "Train:" in line:
                l = re.search(r"Loss\s+([0-9.]+)", line)
                p = re.search(r"Prec\s+([0-9.]+)", line)
                r = re.search(r"Rec\s+([0-9.]+)", line)
                p9 = re.search(r"P@R0.9\s+([0-9.]+)", line)
                if all([l, p, r, p9]):
                    data['epoch'].append(current_epoch)
                    data['t_loss'].append(float(l.group(1)))
                    data['t_prec'].append(float(p.group(1)))
                    data['t_rec'].append(float(r.group(1)))
                    data['t_p9'].append(float(p9.group(1)))
            
            elif "Val  :" in line:
                l = re.search(r"Loss\s+([0-9.]+)", line)
                p = re.search(r"Prec\s+([0-9.]+)", line)
                r = re.search(r"Rec\s+([0-9.]+)", line)
                p9 = re.search(r"P@R0.9\s+([0-9.]+)", line)
                if all([l, p, r, p9]):
                    data['v_loss'].append(float(l.group(1)))
                    data['v_prec'].append(float(p.group(1)))
                    data['v_rec'].append(float(r.group(1)))
                    data['v_p9'].append(float(p9.group(1)))
    
    # Sync lengths (Train and Val must match for the same epoch)
    min_len = min(len(data['epoch']), len(data['t_loss']), len(data['v_loss']))
    for k in data:
        data[k] = data[k][:min_len]
    return data

def plot_experiment(log_path, output_dir, title_prefix):
    os.makedirs(output_dir, exist_ok=True)
    data = parse_log(log_path)
    if not data['epoch']:
        print(f"No data found in {log_path}")
        return

    metrics = [
        ('Loss', 'loss', 'Value'),
        ('Precision', 'prec', 'Score'),
        ('Recall', 'rec', 'Score'),
        ('P@R0.9', 'p9', 'Score')
    ]

    for label, key, ylabel in metrics:
        plt.figure(figsize=(10, 6))
        t_vals = data[f't_{key}']
        v_vals = data[f'v_{key}']
        epochs = data['epoch']
        
        plt.plot(epochs, t_vals, label=f'Train {label}', color='blue', alpha=0.4)
        plt.plot(epochs, v_vals, label=f'Val {label}', color='red', linewidth=2)
        
        # Calculate smoothing for Train
        if len(t_vals) > 20:
            t_smooth = np.convolve(t_vals, np.ones(10)/10, mode='valid')
            plt.plot(epochs[9:], t_smooth, color='darkblue', linewidth=1.5, label=f'Train {label} (Smooth)')

        if key == 'loss':
            best_val = min(v_vals)
            idx = v_vals.index(best_val)
            plt.title(f"{title_prefix}: {label}\nBest Val: {best_val:.4f} at Epoch {epochs[idx]}")
        else:
            best_val = max(v_vals)
            idx = v_vals.index(best_val)
            plt.title(f"{title_prefix}: {label}\nBest Val: {best_val:.4f} at Epoch {epochs[idx]}")

        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        filename = f"metric_{label.lower().replace('@', '_at_')}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        print(f"Corrected {filename} in {output_dir}")

plot_experiment('2026-03-25_full_mlp_training_1000ep/train_full_mlp.log', '2026-03-25_full_mlp_training_1000ep/plots', 'Full MLP Training')
plot_experiment('2026-03-25_learnable_c_water/train_learnable_c.log', '2026-03-25_learnable_c_water/plots', 'Learnable C_Water')
