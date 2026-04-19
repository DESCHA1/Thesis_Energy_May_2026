import pandas as pd
import numpy as np
import holidays

# 1. Load data
df = pd.read_csv('DATA/final_merged_data_for_modeling.csv', parse_dates=['utc_timestamp'])
df.set_index('utc_timestamp', inplace=True)

# 2. Add Calendar Flags
de_holidays = holidays.Germany()
df['is_holiday'] = df.index.map(lambda x: 1 if x in de_holidays else 0)
df['is_weekend'] = df.index.dayofweek.map(lambda x: 1 if x >= 5 else 0)

# 3. Add Cyclical Encoding for Hour and Day of Year
def encode_cyclical(series, max_val):
    sin = np.sin(2 * np.pi * series / max_val)
    cos = np.cos(2 * np.pi * series / max_val)
    return sin, cos

df['hour_sin'], df['hour_cos'] = encode_cyclical(df.index.hour, 24)
df['day_sin'], df['day_cos'] = encode_cyclical(df.index.dayofyear, 365.25)

# 4. Save the Final Feature Engineered Dataset
df.to_csv('DATA/final_data_with_features.csv')
print("Features added: Holidays, Weekends, and Cyclical Time.")
print(df[['is_holiday', 'hour_sin', 'hour_cos']].head()) # Jan 1 should return a holiday
print(df[['is_holiday', 'hour_sin', 'hour_cos']].iloc[24:48]) # Jan 2 should not return a holiday