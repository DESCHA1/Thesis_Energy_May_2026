import pandas as pd
import numpy as np

# 1. Load the file
df = pd.read_csv('DATA/final_data_with_features.csv')

# 2. Convert timestamp to datetime objects and sort
# This is the "Safety First" step for time-series
df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
df = df.sort_values('utc_timestamp').reset_index(drop=True)

# 3. Create the 'time_idx' (Standard for TFT)
# It must be an integer sequence 0, 1, 2...
df['time_idx'] = np.arange(len(df))

# 4. Create the 'region' group (Standard for TFT)
df['region'] = "DE"

# 5. Quick Sanity Check
print(f"✅ Data loaded. Shape: {df.shape}")
print(f"Columns available: {df.columns.tolist()}")

TFT_FEATURES = {
    "target": "residual_load",
    "group_ids": ["region"],
    "time_idx": "time_idx",
    "static_categoricals": ["region"],
    "time_varying_known_reals": [
        "is_holiday", "is_weekend", 
        "hour_sin", "hour_cos", 
        "day_sin", "day_cos"
    ],
    "time_varying_unknown_reals": [
        "residual_load", # Historical values of the target
        "wind_u", "wind_v", "ghi"
    ],
}