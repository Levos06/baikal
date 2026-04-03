import json
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import curve_fit

def log_func(x, a, b):
    return a * np.log10(x) + b

def build_plot():
    registry_path = 'model_registry.json'
    if not os.path.exists(registry_path): return

    with open(registry_path, 'r') as f:
        data = json.load(f)

    params, p_at_r9 = [], []
    for model in data:
        if model.get('cheating_tres', False) or 'CNN' in model['id']: continue
        metrics = model.get('metrics', {})
        val_p9 = (metrics.get('best', {}) or {}).get('val_p_at_r09') or (metrics.get('final', {}) or {}).get('val_p_at_r09')
        if val_p9 is not None:
            params.append(model['params'])
            p_at_r9.append(val_p9)

    params = np.array(params)
    p_at_r9 = np.array(p_at_r9)

    # Filter out outliers ( Pareto frontier logic - only best for size)
    # This helps the curve to follow the POTENTIAL of the architecture
    unique_params = np.unique(params)
    frontier_x, frontier_y = [], []
    for p in unique_params:
        frontier_x.append(p)
        frontier_y.append(np.max(p_at_r9[params == p]))
    
    frontier_x = np.array(frontier_x)
    frontier_y = np.array(frontier_y)

    # Fit logarithmic curve (Scaling Law)
    popt, _ = curve_fit(log_func, frontier_x, frontier_y)
    
    # Generate smooth curve for plot
    x_range = np.linspace(frontier_x.min(), frontier_x.max(), 500)
    y_pred = log_func(x_range, *popt)

    plt.figure(figsize=(10, 7))
    plt.scatter(params, p_at_r9, color='blue', alpha=0.5, s=60, label='All Experiments')
    plt.scatter(frontier_x, frontier_y, color='darkblue', s=100, label='Pareto Frontier (Best)')
    plt.plot(x_range, y_pred, color='red', linewidth=3, label=f'Scaling Law (Logarithmic)')

    plt.grid(True, which="both", ls="-", alpha=0.1)
    plt.xlabel('Number of Parameters (Linear)')
    plt.ylabel('P@R0.9 Score')
    plt.title('Correct Scaling Law: Quality vs Parameter Count')
    plt.legend()

    output_path = 'plots/clean_regression_recalculated.png'
    os.makedirs('plots', exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")
    print(f"Scaling Formula: P@R0.9 = {popt[0]:.4f} * log10(params) + {popt[1]:.4f}")

if __name__ == "__main__":
    build_plot()
