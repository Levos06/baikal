import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

FILE_PATH = "/home3/ivkhar/Baikal/data/normed/baikal_mc2020_multi_split_0924mid_eq_norm.h5"
PLOT_PATH = "2026-01-28_snr_calculation/plots/snr_event_distribution.png"
NUM_EVENTS_SAMPLE = 10000

def get_event_snrs(f, split):
    labels_ds = f[f"{split}/labels/data"]
    starts_ds = f[f"{split}/ev_starts/data"]
    
    num_events = min(NUM_EVENTS_SAMPLE, starts_ds.shape[0] - 1)
    event_snrs = []
    
    # Read starts for sampled events
    starts = starts_ds[:num_events + 1]
    
    # Read all labels for these events in one go to be faster
    total_hits = starts[-1]
    labels = labels_ds[:total_hits]
    
    for i in range(num_events):
        ev_labels = labels[starts[i]:starts[i+1]]
        signal = np.sum(ev_labels != 0)
        noise = np.sum(ev_labels == 0)
        
        if noise > 0:
            event_snrs.append(signal / noise)
        else:
            event_snrs.append(float('nan')) # Or some very high value
            
    return np.array(event_snrs)

def main():
    if not os.path.exists(FILE_PATH):
        print(f"Error: File {FILE_PATH} not found.")
        return

    snr_data = {}
    with h5py.File(FILE_PATH, 'r') as f:
        for split in ['train', 'test', 'val']:
            print(f"Calculating event SNRs for {split}...")
            snr_data[split] = get_event_snrs(f, split)

    # Create boxplot
    plt.figure(figsize=(10, 6))
    
    data_to_plot = [snr_data['train'], snr_data['test'], snr_data['val']]
    # Filter out NaNs for plotting
    data_to_plot = [d[~np.isnan(d)] for d in data_to_plot]
    
    plt.boxplot(data_to_plot, labels=['train', 'test', 'val'], patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='red'))
    
    plt.title(f"Distribution of Event-level SNR (Sample of {NUM_EVENTS_SAMPLE} events per split)")
    plt.ylabel("SNR (Signal Hits / Noise Hits)")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Use log scale if values vary wildly
    # plt.yscale('log') 
    
    plt.savefig(PLOT_PATH)
    print(f"Plot saved to {PLOT_PATH}")

if __name__ == "__main__":
    main()
