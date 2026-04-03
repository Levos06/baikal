import json
import matplotlib.pyplot as plt
import numpy as np
import os

def build_plot():
    registry_path = 'model_registry.json'
    if not os.path.exists(registry_path):
        print("Registry not found!")
        return

    with open(registry_path, 'r') as f:
        data = json.load(f)

    params = []
    p_at_r9 = []
    labels = []
    colors = []

    for model in data:
        # Extract best P@R0.9 if available, otherwise final, otherwise skip
        metrics = model.get('metrics', {})
        best = metrics.get('best', {})
        final = metrics.get('final', {})
        
        val_p9 = best.get('val_p_at_r09') or final.get('val_p_at_r09')
        
        if val_p9 is not None:
            params.append(model['params'])
            p_at_r9.append(val_p9)
            labels.append(model['id'])
            # Color coding: cheating models in red, others in blue
            colors.append('red' if model.get('cheating_tres', False) else 'blue')

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(params, p_at_r9, c=colors, alpha=0.6, s=100, edgecolors='black')
    
    # Annotate points
    for i, txt in enumerate(labels):
        plt.annotate(txt, (params[i], p_at_r9[i]), xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)

    plt.xscale('log') # Params vary from 17k to 8.5M, so log scale is better
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xlabel('Number of Parameters (Log Scale)')
    plt.ylabel('P@R0.9 Score')
    plt.title('Model Quality (P@R0.9) vs Parameter Count')
    
    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Clean Model', markerfacecolor='blue', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='Cheating (t_res)', markerfacecolor='red', markersize=10)]
    plt.legend(handles=legend_elements)

    output_path = 'plots/params_vs_quality.png'
    os.makedirs('plots', exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    build_plot()
