import pandas as pd

# 1. Load the OPSD data
file_path = r'C:\Users\desch\.vscode\THESIS_ENERGY_2026\DATA\time_series_60min_singleindex_filtered.csv'
df_energy = pd.read_csv(file_path)

# 2. Define the exact columns we need for the target (Germany wide)
# Based off a prior calculation comparing wind_generation_total with the combined sum 
# of wind_offshore and wind_onshore (74.83% match rate) we are only using wind_generation_total
cols_to_keep = [
    'utc_timestamp',
    'DE_load_actual_entsoe_transparency',
    'DE_solar_generation_actual',
    'DE_wind_generation_actual'
]

# Filter the dataframe
df_clean = df_energy[cols_to_keep].copy()

# 3. Format the Timestamp for a safe join
# Ensure it is a datetime object and explicitly set to UTC
df_clean['utc_timestamp'] = pd.to_datetime(df_clean['utc_timestamp'], utc=True)

# 4. Handle Missing Values BEFORE calculating residual load
# Time-Aware linear interpolation is standard for small 1-2 hour gaps in ENTSO-E data
df_clean.set_index('utc_timestamp', inplace=True)
df_clean = df_clean.interpolate(method='time', limit=3)

#PyTorch will crash if we leave in any NaNs


df_clean = df_clean.dropna()
# 5. Calculate our Target Variable: Residual Load
df_clean['residual_load'] = df_clean['DE_load_actual_entsoe_transparency'] - \
                            (df_clean['DE_solar_generation_actual'] + df_clean['DE_wind_generation_actual'])

# We will keep the raw load/generation columns in order to perform an ablation study that compares TFT's performance on 
# A) residual load + weather data versus 
# B) residual load + load + solar + wind + weather data 
# as TFT might benefit from seeing these as separate features.)
# df_clean = df_clean[['residual_load']] 

# Reset index to prepare for the merge with weather data
df_clean.reset_index(inplace=True)

print(df_clean.head())
save_path = r'C:\Users\desch\.vscode\THESIS_ENERGY_2026\DATA\cleaned_residual_load_2015_2019.csv'
df_clean.to_csv(save_path, index=False)