import xarray as xr
import pandas as pd
import glob
import sys

# 1. Locate all the extracted .nc files
search_pattern = 'DATA/ERA5_Weather/*.nc'
file_list = sorted(glob.glob(search_pattern))
print(f"Opening {len(file_list)} weather files...")

if len(file_list) == 0:
    print(f"ERROR: Could not find any files matching: {search_pattern}")
    sys.exit()

# 2. Merge all files
print("Merging NetCDF files (this might take a minute or two)...")
ds = xr.open_mfdataset(file_list, combine='by_coords', engine='netcdf4')

# 3. Spatial Averaging (National Mean for Germany)
print("Averaging spatially across latitudes and longitudes...")
ds_avg = ds.mean(dim=['latitude', 'longitude'])

# 4. Human-Readable Renaming
mapping = {
    'var165': 'wind_u', 'u10': 'wind_u',
    'var166': 'wind_v', 'v10': 'wind_v',
    'var169': 'ghi_raw', 'ssrd': 'ghi_raw'   
}
existing_vars = list(ds_avg.data_vars)
rename_dict = {k: v for k, v in mapping.items() if k in existing_vars}
ds_avg = ds_avg.rename(rename_dict)

# 5. Convert to Pandas
print("Converting to Pandas DataFrame...")
df = ds_avg.to_dataframe()

# 6. Unit Conversion & De-accumulation
print("Converting accumulated Joules to hourly Watts/m²...")

if 'ghi_raw' in df.columns:
    df['ghi_diff'] = df['ghi_raw'].diff().fillna(0)
    df.loc[df['ghi_diff'] < 0, 'ghi_diff'] = 0
    df['ghi'] = df['ghi_diff'] / 3600


# 7. Interpolate any missing holes
print("Interpolating missing data points...")
# this silences Panda warnings
df = df.infer_objects(copy=False) 
df = df.interpolate(method='time')

# 8. Clean up the time column
print("Formatting timestamps to UTC...")
df = df.reset_index()

# Copernicus often uses 'valid_time' or 'time'
# this was a major headache! GOOD to REMEMBER!
if 'valid_time' in df.columns:
    df.rename(columns={'valid_time': 'utc_timestamp'}, inplace=True)
elif 'time' in df.columns:
    df.rename(columns={'time': 'utc_timestamp'}, inplace=True)

df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'], utc=True)

# 9. Save only the clean, human-readable columns
print("Saving final dataset...")

final_columns = ['utc_timestamp']
for col in ['wind_u', 'wind_v', 'ghi']:
    if col in df.columns:
        final_columns.append(col)

df_final = df[final_columns]

save_path = 'DATA/cleaned_era5_weather.csv'
df_final.to_csv(save_path, index=False)
