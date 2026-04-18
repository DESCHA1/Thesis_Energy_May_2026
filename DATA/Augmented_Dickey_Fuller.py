import pandas as pd
from statsmodels.tsa.stattools import adfuller
import os

# 1. Load the merged data
df = pd.read_csv('DATA/final_merged_data_for_modeling.csv', index_col='utc_timestamp')

# 2. Run the ADF Test
print("Calculating Stationarity (ADF Test)...")
adf_result = adfuller(df['residual_load'].dropna())

# 3. Save to the EDA_Results folder
output_dir = 'EDA_Results'
os.makedirs(output_dir, exist_ok=True)

with open(f'{output_dir}/03_adf_test_results.txt', 'w') as f:
    f.write("--- Augmented Dickey-Fuller Test Results ---\n")
    f.write(f"ADF Statistic: {adf_result[0]:.4f}\n")
    f.write(f"p-value: {adf_result[1]:.4e}\n")
    f.write("Critical Values:\n")
    for key, value in adf_result[4].items():
        f.write(f"   {key}: {value:.4f}\n")
    
    if adf_result[1] <= 0.05:
        f.write("\nConclusion: The series is STATIONARY (Reject Null Hypothesis)")
    else:
        f.write("\nConclusion: The series is NON-STATIONARY (Fail to Reject Null Hypothesis)")

print(f"ADF results successfully saved to {output_dir}/03_adf_test_results.txt")