import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('DATA/time_series_60min_singleindex_filtered.csv')

# Ensures timestamp is the index (crucial for time-series splits)
df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
df = df.sort_values('utc_timestamp').set_index('utc_timestamp')

# Run the sanity check 
time_diffs = df.index.to_series().diff().dt.total_seconds() / 3600
gaps = time_diffs[1:][time_diffs[1:] != 1.0]

if gaps.empty:
    print("✅ 'df' is defined and the timeline is continuous.")
else:
    print(f"⚠️ 'df' defined, but found {len(gaps)} timeline gaps!")