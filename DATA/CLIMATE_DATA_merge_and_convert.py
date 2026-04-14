import xarray as xr
import pandas as pd
import glob

# 1. Get the list of all .nc files
file_list = sorted(glob.glob('era5_ger_*.nc'))
print(f"Opening {len(file_list)} weather files...")

# 2. Merge all files using the netcdf4 engine
ds = xr.open_mfdataset(file_list, combine='by_coords', engine='netcdf4')

# 3. Spatial Averaging (National Mean)
ds_avg = ds.mean(dim=['latitude', 'longitude'])

# 4. Human-Readable Renaming
# var165 -> wind_u, var166 -> wind_v
# var169 -> ghi (Global), var235 -> dhi (Diffuse)
mapping = {
    'var165': 'wind_u', 
    'var166': 'wind_v', 
    'var169': 'ghi_raw', 
    'var235': 'dhi_raw'
}
existing_vars = list(ds_avg.data_vars)
rename_dict = {k: v for k, v in mapping.items() if k in existing_vars}
ds_avg = ds_avg.rename(rename_dict)

# 5. Convert to Pandas
df = ds_avg.to_dataframe()

# 6. Unit Conversion & De-accumulation (Joules -> Watts)
# Step A: Hourly difference
df['ghi_diff'] = df['ghi_raw'].diff().fillna(0)
df['dhi_diff'] = df['dhi_raw'].diff().fillna(0)

# Step B: Reset negatives at midnight to zero
df.loc[df['ghi_diff'] < 0, 'ghi_diff'] = 0
df.loc[df['dhi_diff'] < 0, 'dhi_diff'] = 0

# Step C: Final conversion to Watts/m^2
df['ghi'] = df['ghi_diff'] / 3600
df['dhi'] = df['dhi_diff'] / 3600

# 7. Interpolate the October 2018 hole
df = df.interpolate(method='time')

# 8. Save only the clean, human-readable columns
df_final = df[['wind_u', 'wind_v', 'ghi', 'dhi']]
df_final.to_csv('germany_weather_final_2015_2019.csv')

print("--- FINISHED ---")
print("Saved columns: wind_u, wind_v, ghi, dhi")