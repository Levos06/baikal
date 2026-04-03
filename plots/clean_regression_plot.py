import json
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.linear_model import LinearRegression

def build_plot():
    registry_path = 'model_registry.json'
    if not os.path.exists(registry_path):
        print("Registry not found!")
        return

    with open(registry_path, 'r') as f:
        data = json.load(f)

    params = []
    p_at_r9 = []

    for model in data:
        # Filter out cheating and CNN models
        if model.get('cheating_tres', False) or 'CNN' in model['id']:
            continue
            
        metrics = model.get('metrics', {})
        best = metrics.get('best', {})
        final = metrics.get('final', {})
        
        val_p9 = best.get('val_p_at_r09') or final.get('val_p_at_r09')
        
        if val_p9 is not None:
            params.append(model['params'])
            p_at_r9.append(val_p9)

    if not params:
        print("No suitable data found for regression!")
        return

    # Prepare data for regression (Linear scale for params)
    X = np.array(params).reshape(-1, 1)
    y = np.array(p_at_r9)

    model_lr = LinearRegression()
    model_lr.fit(X, y)
    
    # Generate line points
    x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred = model_lr.predict(x_range)

    plt.figure(figsize=(10, 7))
    plt.scatter(params, p_at_r9, color='blue', alpha=0.7, s=80, label='Clean Models (GCN/GAT)')
    plt.plot(x_range, y_pred, color='red', linestyle='--', linewidth=2, label=f'Linear Regression (R²={model_lr.score(X, y):.3f})')

    plt.grid(True, which="both", ls="-", alpha=0.15)
    plt.xlabel('Number of Parameters')
    plt.ylabel('P@R0.9 Score')
    plt.title('Linear Scaling: Quality vs Model Complexity')
    plt.legend()

    output_path = 'plots/clean_regression.png'
    os.makedirs('plots', exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")
    # Print with safety
    coef = model_lr.coef_[0] if hasattr(model_lr.coef_, "__len__") else model_lr.coef_
    intercept = model_lr.intercept_[0] if hasattr(model_lr.intercept_, "__len__") else model_lr.intercept_
    print(f"Regression formula: P@R0.9 = {coef:.8f} * params + {intercept:.4f}")

if __name__ == "__main__":
    build_plot()
