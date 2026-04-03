import json
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import curve_fit

def log_func(x, a, b):
    # Log-scaling law fit
    return a * np.log10(x) + b

def build_plot():
    registry_path = 'model_registry.json'
    if not os.path.exists(registry_path): return

    with open(registry_path, 'r') as f:
        data = json.load(f)

    # Categories for plotting
    cat_data = {
        'Clean GCN/GAT': {'x': [], 'y': [], 'color': 'blue'},
        'Cheating (t_res)': {'x': [], 'y': [], 'color': 'red'},
        'CNN/Baselines': {'x': [], 'y': [], 'color': 'green'}
    }

    all_x, all_y = [], []

    for model in data:
        metrics = model.get('metrics', {})
        val_p9 = (metrics.get('best', {}) or {}).get('val_p_at_r09') or (metrics.get('final', {}) or {}).get('val_p_at_r09')
        
        if val_p9 is not None:
            p = model['params']
            all_x.append(p)
            all_y.append(val_p9)
            
            # Determine category
            if model.get('cheating_tres', False):
                cat = 'Cheating (t_res)'
            elif 'CNN' in model['id']:
                cat = 'CNN/Baselines'
            else:
                cat = 'Clean GCN/GAT'
            
            cat_data[cat]['x'].append(p)
            cat_data[cat]['y'].append(val_p9)

    all_x = np.array(all_x)
    all_y = np.array(all_y)

    # Fit logarithmic curve to ALL data points
    # We use a robust fit to ensure it captures the trend from the lowest points
    popt, _ = curve_fit(log_func, all_x, all_y, p0=[0.3, -1.5])
    
    # Generate smooth curve (starting from the smallest model's params)
    x_range = np.linspace(all_x.min(), all_x.max(), 1000)
    y_pred = log_func(x_range, *popt)

    plt.figure(figsize=(12, 8))
    
    # Plot categories
    for name, d in cat_data.items():
        if d['x']:
            plt.scatter(d['x'], d['y'], color=d['color'], s=120, edgecolors='black', alpha=0.6, label=name, zorder=3)

    # Plot the global Scaling Law curve
    plt.plot(x_range, y_pred, color='black', linestyle='-', linewidth=3, 
             label=f'Global Scaling Law: {popt[0]:.3f}*log10(p) + {popt[1]:.3f}', zorder=2)

    plt.grid(True, which="both", ls="-", alpha=0.15)
    plt.xlabel('Number of Parameters (N)')
    plt.ylabel('P@R0.9 Score')
    plt.title('The Grand Marathon: Universal Scaling Law (All Models)')
    plt.ylim(0, 1.0) # Quality is between 0 and 1
    plt.legend(loc='lower right')

    # Table of points for reference (minimal)
    print(f"Total models plotted: {len(all_x)}")
    
    output_path = 'plots/grand_marathon_scaling.png'
    os.makedirs('plots', exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    build_plot()
