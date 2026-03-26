import numpy as np
import re
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load data
epochs, c_values = [], []
with open('c_water_data.txt', 'r') as f:
    for line in f:
        matches = re.findall(r"([0-9.]+)", line)
        if len(matches) >= 2:
            epochs.append(float(matches[0]))
            c_values.append(float(matches[-1]))

x = np.array(epochs)
y = np.array(c_values)

# Fitted parameters from previous run
a = 0.05248567
b = 0.00184625
S = 0.00061969
O = 0.22500581

def fit_func(x, a, b, S, O):
    arg = -a * x
    arg = np.clip(arg, -50, 50)
    sigmoid = 1.0 / (1.0 + np.exp(arg))
    return S * (sigmoid + b * x - 0.5) + O

y_pred = fit_func(x, a, b, S, O)

# Metrics
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_pred)
max_err = np.max(np.abs(y - y_pred))

# Relative error (%)
mean_c = np.mean(y)
rel_rmse_pct = (rmse / mean_c) * 100

print(f"--- Fit Quality Metrics ---")
print(f"R^2 Score:          {r2:.8f} (Excellent if > 0.99)")
print(f"RMSE:               {rmse:.8e} m/ns")
print(f"MAE:                {mae:.8e} m/ns")
print(f"Max Absolute Error: {max_err:.8e} m/ns")
print(f"Relative RMSE:      {rel_rmse_pct:.6f} %")
print(f"---------------------------")
