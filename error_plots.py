import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


def process_data(data):
    # Split the input data by lines
    lines = data.strip().split("\n")

    # Initialize the arrays
    mae = []
    rmse = []
    mape = []

    # Process each line
    for line in lines:
        # Split the line by commas and extract the relevant parts
        parts = line.split(", ")
        mae_part = parts[1]
        rmse_part = parts[2]
        mape_part = parts[3]

        # Extract the numeric values and convert to float
        mae_value = float(mae_part.split(": ")[1])
        rmse_value = float(rmse_part.split(": ")[1])
        mape_value = float(mape_part.split(": ")[1])

        # Append the values to the respective arrays
        mae.append(mae_value)
        rmse.append(rmse_value)
        mape.append(mape_value)

    return mae, rmse, mape


# Example usage
data = """
Evaluate best model on test data for horizon 1, Test MAE: 3.3206, Test RMSE: 6.0431, Test MAPE: 0.0898
Evaluate best model on test data for horizon 2, Test MAE: 3.7107, Test RMSE: 6.8217, Test MAPE: 0.1024
Evaluate best model on test data for horizon 3, Test MAE: 4.0279, Test RMSE: 7.4150, Test MAPE: 0.1138
Evaluate best model on test data for horizon 4, Test MAE: 4.4690, Test RMSE: 8.3471, Test MAPE: 0.1283
Evaluate best model on test data for horizon 5, Test MAE: 4.6344, Test RMSE: 8.6962, Test MAPE: 0.1338
Evaluate best model on test data for horizon 6, Test MAE: 4.7738, Test RMSE: 8.9907, Test MAPE: 0.1390
Evaluate best model on test data for horizon 7, Test MAE: 4.9423, Test RMSE: 9.4476, Test MAPE: 0.1448
Evaluate best model on test data for horizon 8, Test MAE: 5.0122, Test RMSE: 9.6084, Test MAPE: 0.1471
Evaluate best model on test data for horizon 9, Test MAE: 5.0772, Test RMSE: 9.7491, Test MAPE: 0.1494
Evaluate best model on test data for horizon 10, Test MAE: 5.1313, Test RMSE: 9.8893, Test MAPE: 0.1517
Evaluate best model on test data for horizon 11, Test MAE: 5.1780, Test RMSE: 9.9854, Test MAPE: 0.1531
Evaluate best model on test data for horizon 12, Test MAE: 5.2283, Test RMSE: 10.0700, Test MAPE: 0.1547
Evaluate best model on test data for horizon 13, Test MAE: 5.2461, Test RMSE: 10.1218, Test MAPE: 0.1555
Evaluate best model on test data for horizon 14, Test MAE: 5.2799, Test RMSE: 10.1839, Test MAPE: 0.1566
Evaluate best model on test data for horizon 15, Test MAE: 5.3170, Test RMSE: 10.2398, Test MAPE: 0.1579
Evaluate best model on test data for horizon 16, Test MAE: 5.3266, Test RMSE: 10.2943, Test MAPE: 0.1584
Evaluate best model on test data for horizon 17, Test MAE: 5.3443, Test RMSE: 10.3276, Test MAPE: 0.1591
Evaluate best model on test data for horizon 18, Test MAE: 5.3673, Test RMSE: 10.3641, Test MAPE: 0.1601
Evaluate best model on test data for horizon 19, Test MAE: 5.3867, Test RMSE: 10.4392, Test MAPE: 0.1609
Evaluate best model on test data for horizon 20, Test MAE: 5.3993, Test RMSE: 10.4575, Test MAPE: 0.1615
Evaluate best model on test data for horizon 21, Test MAE: 5.4144, Test RMSE: 10.4844, Test MAPE: 0.1623
Evaluate best model on test data for horizon 22, Test MAE: 5.4192, Test RMSE: 10.5360, Test MAPE: 0.1628
Evaluate best model on test data for horizon 23, Test MAE: 5.4311, Test RMSE: 10.5496, Test MAPE: 0.1635
Evaluate best model on test data for horizon 24, Test MAE: 5.4444, Test RMSE: 10.5778, Test MAPE: 0.1644
Evaluate best model on test data for horizon 25, Test MAE: 5.4493, Test RMSE: 10.6377, Test MAPE: 0.1651
Evaluate best model on test data for horizon 26, Test MAE: 5.4580, Test RMSE: 10.6445, Test MAPE: 0.1656
Evaluate best model on test data for horizon 27, Test MAE: 5.4704, Test RMSE: 10.6629, Test MAPE: 0.1663
Evaluate best model on test data for horizon 28, Test MAE: 5.4952, Test RMSE: 10.7378, Test MAPE: 0.1672
Evaluate best model on test data for horizon 29, Test MAE: 5.4970, Test RMSE: 10.7395, Test MAPE: 0.1674
Evaluate best model on test data for horizon 30, Test MAE: 5.5047, Test RMSE: 10.7486, Test MAPE: 0.1679
Evaluate best model on test data for horizon 31, Test MAE: 5.5215, Test RMSE: 10.8013, Test MAPE: 0.1684
Evaluate best model on test data for horizon 32, Test MAE: 5.5216, Test RMSE: 10.7946, Test MAPE: 0.1685
Evaluate best model on test data for horizon 33, Test MAE: 5.5230, Test RMSE: 10.7940, Test MAPE: 0.1688
Evaluate best model on test data for horizon 34, Test MAE: 5.5329, Test RMSE: 10.8276, Test MAPE: 0.1692
Evaluate best model on test data for horizon 35, Test MAE: 5.5336, Test RMSE: 10.8214, Test MAPE: 0.1692
Evaluate best model on test data for horizon 36, Test MAE: 5.5352, Test RMSE: 10.8152, Test MAPE: 0.1695
Evaluate best model on test data for horizon 37, Test MAE: 5.5381, Test RMSE: 10.8364, Test MAPE: 0.1697
Evaluate best model on test data for horizon 38, Test MAE: 5.5341, Test RMSE: 10.8208, Test MAPE: 0.1695
Evaluate best model on test data for horizon 39, Test MAE: 5.5334, Test RMSE: 10.8105, Test MAPE: 0.1695
Evaluate best model on test data for horizon 40, Test MAE: 5.5442, Test RMSE: 10.8353, Test MAPE: 0.1701
Evaluate best model on test data for horizon 41, Test MAE: 5.5411, Test RMSE: 10.8183, Test MAPE: 0.1699
Evaluate best model on test data for horizon 42, Test MAE: 5.5433, Test RMSE: 10.8066, Test MAPE: 0.1700
Evaluate best model on test data for horizon 43, Test MAE: 5.5556, Test RMSE: 10.8460, Test MAPE: 0.1710
Evaluate best model on test data for horizon 44, Test MAE: 5.5584, Test RMSE: 10.8396, Test MAPE: 0.1710
Evaluate best model on test data for horizon 45, Test MAE: 5.5638, Test RMSE: 10.8420, Test MAPE: 0.1713
Evaluate best model on test data for horizon 46, Test MAE: 5.5788, Test RMSE: 10.8845, Test MAPE: 0.1724
Evaluate best model on test data for horizon 47, Test MAE: 5.5842, Test RMSE: 10.8824, Test MAPE: 0.1724
Evaluate best model on test data for horizon 48, Test MAE: 5.5912, Test RMSE: 10.8857, Test MAPE: 0.1728
Evaluate best model on test data for horizon 49, Test MAE: 5.5965, Test RMSE: 10.9133, Test MAPE: 0.1736
Evaluate best model on test data for horizon 50, Test MAE: 5.5985, Test RMSE: 10.9129, Test MAPE: 0.1735
Evaluate best model on test data for horizon 51, Test MAE: 5.6070, Test RMSE: 10.9185, Test MAPE: 0.1738
Evaluate best model on test data for horizon 52, Test MAE: 5.6276, Test RMSE: 10.9773, Test MAPE: 0.1749
Evaluate best model on test data for horizon 53, Test MAE: 5.6386, Test RMSE: 10.9941, Test MAPE: 0.1751
Evaluate best model on test data for horizon 54, Test MAE: 5.6558, Test RMSE: 11.0195, Test MAPE: 0.1759
Evaluate best model on test data for horizon 55, Test MAE: 5.6892, Test RMSE: 11.0853, Test MAPE: 0.1770
Evaluate best model on test data for horizon 56, Test MAE: 5.6967, Test RMSE: 11.0963, Test MAPE: 0.1770
Evaluate best model on test data for horizon 57, Test MAE: 5.7094, Test RMSE: 11.1068, Test MAPE: 0.1772
Evaluate best model on test data for horizon 58, Test MAE: 5.7456, Test RMSE: 11.1741, Test MAPE: 0.1782
Evaluate best model on test data for horizon 59, Test MAE: 5.7587, Test RMSE: 11.1896, Test MAPE: 0.1783
Evaluate best model on test data for horizon 60, Test MAE: 5.7724, Test RMSE: 11.2011, Test MAPE: 0.1784
Evaluate best model on test data for horizon 61, Test MAE: 5.7928, Test RMSE: 11.2408, Test MAPE: 0.1790
Evaluate best model on test data for horizon 62, Test MAE: 5.8040, Test RMSE: 11.2502, Test MAPE: 0.1790
Evaluate best model on test data for horizon 63, Test MAE: 5.8176, Test RMSE: 11.2621, Test MAPE: 0.1794
Evaluate best model on test data for horizon 64, Test MAE: 5.8311, Test RMSE: 11.2789, Test MAPE: 0.1797
Evaluate best model on test data for horizon 65, Test MAE: 5.8450, Test RMSE: 11.2996, Test MAPE: 0.1799
Evaluate best model on test data for horizon 66, Test MAE: 5.8590, Test RMSE: 11.3138, Test MAPE: 0.1802
Evaluate best model on test data for horizon 67, Test MAE: 5.8723, Test RMSE: 11.3321, Test MAPE: 0.1806
Evaluate best model on test data for horizon 68, Test MAE: 5.8803, Test RMSE: 11.3371, Test MAPE: 0.1807
Evaluate best model on test data for horizon 69, Test MAE: 5.8925, Test RMSE: 11.3410, Test MAPE: 0.1811
Evaluate best model on test data for horizon 70, Test MAE: 5.9065, Test RMSE: 11.3619, Test MAPE: 0.1816
Evaluate best model on test data for horizon 71, Test MAE: 5.9125, Test RMSE: 11.3677, Test MAPE: 0.1818
Evaluate best model on test data for horizon 72, Test MAE: 5.9204, Test RMSE: 11.3693, Test MAPE: 0.1820
Evaluate best model on test data for horizon 73, Test MAE: 5.9422, Test RMSE: 11.3939, Test MAPE: 0.1827
Evaluate best model on test data for horizon 74, Test MAE: 5.9452, Test RMSE: 11.3957, Test MAPE: 0.1827
Evaluate best model on test data for horizon 75, Test MAE: 5.9534, Test RMSE: 11.3962, Test MAPE: 0.1831
Evaluate best model on test data for horizon 76, Test MAE: 5.9721, Test RMSE: 11.4119, Test MAPE: 0.1842
Evaluate best model on test data for horizon 77, Test MAE: 5.9924, Test RMSE: 11.4310, Test MAPE: 0.1849
Evaluate best model on test data for horizon 78, Test MAE: 6.0209, Test RMSE: 11.4578, Test MAPE: 0.1859
Evaluate best model on test data for horizon 79, Test MAE: 6.0523, Test RMSE: 11.4895, Test MAPE: 0.1877
Evaluate best model on test data for horizon 80, Test MAE: 6.0696, Test RMSE: 11.5087, Test MAPE: 0.1884
Evaluate best model on test data for horizon 81, Test MAE: 6.0999, Test RMSE: 11.5407, Test MAPE: 0.1896
Evaluate best model on test data for horizon 82, Test MAE: 6.1323, Test RMSE: 11.5843, Test MAPE: 0.1919
Evaluate best model on test data for horizon 83, Test MAE: 6.1521, Test RMSE: 11.6056, Test MAPE: 0.1928
Evaluate best model on test data for horizon 84, Test MAE: 6.1788, Test RMSE: 11.6280, Test MAPE: 0.1940
Evaluate best model on test data for horizon 85, Test MAE: 6.2344, Test RMSE: 11.7128, Test MAPE: 0.1971
Evaluate best model on test data for horizon 86, Test MAE: 6.2566, Test RMSE: 11.7344, Test MAPE: 0.1982
Evaluate best model on test data for horizon 87, Test MAE: 6.2950, Test RMSE: 11.7750, Test MAPE: 0.2000
Evaluate best model on test data for horizon 88, Test MAE: 6.3982, Test RMSE: 11.9250, Test MAPE: 0.2040
Evaluate best model on test data for horizon 89, Test MAE: 6.4263, Test RMSE: 11.9578, Test MAPE: 0.2052
Evaluate best model on test data for horizon 90, Test MAE: 6.4636, Test RMSE: 12.0019, Test MAPE: 0.2071
Evaluate best model on test data for horizon 91, Test MAE: 6.6083, Test RMSE: 12.2061, Test MAPE: 0.2126
Evaluate best model on test data for horizon 92, Test MAE: 6.6266, Test RMSE: 12.2320, Test MAPE: 0.2136
Evaluate best model on test data for horizon 93, Test MAE: 6.6442, Test RMSE: 12.2501, Test MAPE: 0.2146
Evaluate best model on test data for horizon 94, Test MAE: 6.7951, Test RMSE: 12.4309, Test MAPE: 0.2196
Evaluate best model on test data for horizon 95, Test MAE: 6.8065, Test RMSE: 12.4469, Test MAPE: 0.2206
Evaluate best model on test data for horizon 96, Test MAE: 6.8265, Test RMSE: 12.4646, Test MAPE: 0.2219
Evaluate best model on test data for horizon 97, Test MAE: 6.9813, Test RMSE: 12.6424, Test MAPE: 0.2263
Evaluate best model on test data for horizon 98, Test MAE: 6.9980, Test RMSE: 12.6740, Test MAPE: 0.2272
Evaluate best model on test data for horizon 99, Test MAE: 7.0146, Test RMSE: 12.6829, Test MAPE: 0.2286
Evaluate best model on test data for horizon 100, Test MAE: 7.1611, Test RMSE: 12.8450, Test MAPE: 0.2326
Evaluate best model on test data for horizon 101, Test MAE: 7.1785, Test RMSE: 12.8781, Test MAPE: 0.2337
Evaluate best model on test data for horizon 102, Test MAE: 7.1960, Test RMSE: 12.8954, Test MAPE: 0.2350
Evaluate best model on test data for horizon 103, Test MAE: 7.3759, Test RMSE: 13.1095, Test MAPE: 0.2407
Evaluate best model on test data for horizon 104, Test MAE: 7.3935, Test RMSE: 13.1467, Test MAPE: 0.2418
Evaluate best model on test data for horizon 105, Test MAE: 7.4110, Test RMSE: 13.1670, Test MAPE: 0.2430
Evaluate best model on test data for horizon 106, Test MAE: 7.6107, Test RMSE: 13.4160, Test MAPE: 0.2498
Evaluate best model on test data for horizon 107, Test MAE: 7.6470, Test RMSE: 13.4880, Test MAPE: 0.2515
Evaluate best model on test data for horizon 108, Test MAE: 7.6780, Test RMSE: 13.5481, Test MAPE: 0.2536
Evaluate best model on test data for horizon 109, Test MAE: 7.8663, Test RMSE: 13.7662, Test MAPE: 0.2610
Evaluate best model on test data for horizon 110, Test MAE: 7.9163, Test RMSE: 13.8752, Test MAPE: 0.2631
Evaluate best model on test data for horizon 111, Test MAE: 7.9564, Test RMSE: 13.9789, Test MAPE: 0.2656
Evaluate best model on test data for horizon 112, Test MAE: 8.1521, Test RMSE: 14.2085, Test MAPE: 0.2734
Evaluate best model on test data for horizon 113, Test MAE: 8.2205, Test RMSE: 14.3709, Test MAPE: 0.2759
Evaluate best model on test data for horizon 114, Test MAE: 8.2760, Test RMSE: 14.5213, Test MAPE: 0.2792
Evaluate best model on test data for horizon 115, Test MAE: 8.4906, Test RMSE: 14.7670, Test MAPE: 0.2856
Evaluate best model on test data for horizon 116, Test MAE: 8.5629, Test RMSE: 14.9422, Test MAPE: 0.2889
Evaluate best model on test data for horizon 117, Test MAE: 8.6169, Test RMSE: 15.0667, Test MAPE: 0.2928
Evaluate best model on test data for horizon 118, Test MAE: 8.8364, Test RMSE: 15.3336, Test MAPE: 0.2998
Evaluate best model on test data for horizon 119, Test MAE: 8.9309, Test RMSE: 15.5308, Test MAPE: 0.3039
Evaluate best model on test data for horizon 120, Test MAE: 8.9893, Test RMSE: 15.6442, Test MAPE: 0.3077
"""

mae, rmse, mape = process_data(data)

mae = mae[:120]
rmse = rmse[:120]
mape = mape[:120]

horizons = np.arange(1, 121)

# Normalize the data
max_mae = np.max(mae)
max_rmse = np.max(rmse)
max_mape = np.max(mape)

min_mae = np.min(mae)
min_rmse = np.min(rmse)
min_mape = np.min(mape)

normalized_mae = (mae - min_mae) / (max_mae - min_mae)
normalized_rmse = (rmse - min_rmse) / (max_rmse - min_rmse)
normalized_mape = (mape - min_mape) / (max_mape - min_mape)

def log_func(x, a, b):
    return a * np.log10(x) + b

# Fit the logarithmic model to each metric
params_mae, _ = curve_fit(log_func, horizons, normalized_mae)
params_rmse, _ = curve_fit(log_func, horizons, normalized_rmse)
params_mape, _ = curve_fit(log_func, horizons, normalized_mape)

# Predicted values
y_pred_mae = log_func(horizons, *params_mae)
y_pred_rmse = log_func(horizons, *params_rmse)
y_pred_mape = log_func(horizons, *params_mape)

# Calculate R-squared values
r2_mae = r2_score(normalized_mae, y_pred_mae)
r2_rmse = r2_score(normalized_rmse, y_pred_rmse)
r2_mape = r2_score(normalized_mape, y_pred_mape)

# Plot the data and the logarithmic fits
plt.figure(figsize=(12, 6))
plt.plot(horizons, normalized_mae, label='Normalized MAE', marker='o')
plt.plot(horizons, normalized_rmse, label='Normalized RMSE', marker='x')
plt.plot(horizons, normalized_mape, label='Normalized MAPE', marker='s')

plt.plot(horizons, y_pred_mae, label=f'Logarithmic Fit MAE ($R^2$={r2_mae:.2f})', linestyle='--')
plt.plot(horizons, y_pred_rmse, label=f'Logarithmic Fit RMSE ($R^2$={r2_rmse:.2f})', linestyle='--')
plt.plot(horizons, y_pred_mape, label=f'Logarithmic Fit MAPE ($R^2$={r2_mape:.2f})', linestyle='--')

plt.xlabel('Horizon', fontsize=20)
plt.ylabel('Normalized Error Value', fontsize=20)
plt.title('Normalized Test Metrics over Horizons', fontsize=20)
plt.legend(fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.show()

# Print the parameters and R-squared values
print(f"MAE: a = {params_mae[0]:.2f}, b = {params_mae[1]:.2f}, R-squared = {r2_mae:.2f}")
print(f"RMSE: a = {params_rmse[0]:.2f}, b = {params_rmse[1]:.2f}, R-squared = {r2_rmse:.2f}")
print(f"MAPE: a = {params_mape[0]:.2f}, b = {params_mape[1]:.2f}, R-squared = {r2_mape:.2f}")