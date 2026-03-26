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

y_actual = np.array(c_values)
y_fixed = np.full_like(y_actual, 0.225) # Fixed at initial value

# Metrics for "Fixed" assumption
mse_fixed = mean_squared_error(y_actual, y_fixed)
rmse_fixed = np.sqrt(mse_fixed)
mae_fixed = mean_absolute_error(y_actual, y_fixed)
max_err_fixed = np.max(np.abs(y_actual - y_fixed))

# Previous metrics for your Formula (from previous run)
rmse_formula = 2.0339e-06
mae_formula = 1.7004e-06

print(f"--- Fixed C=0.225 vs. Learned Data ---")
print(f"RMSE (Fixed 0.225): {rmse_fixed:.8e} m/ns")
print(f"MAE (Fixed 0.225):  {mae_fixed:.8e} m/ns")
print(f"Max Deviation:     {max_err_fixed:.8e} m/ns")
print(f"\n--- Comparison with your Formula ---")
print(f"RMSE Degradation:   {rmse_fixed / rmse_formula:.1f}x times worse")
print(f"MAE Degradation:    {mae_fixed / mae_formula:.1f}x times worse")
print(f"---------------------------------------")
