import xarray as xr
import glob

# 1. Find an 'accum' file (where radiation lives)
# We search for 'accum' in the filename to be specific
files = glob.glob('DATA/old ERA5 data/*accum*.nc')

if not files:
    print("No 'accum' files found. Checking all .nc files...")
    files = glob.glob('DATA/ERA5_Weather/*.nc')

test_file = sorted(files)[0]
print(f"--- Analyzing file: {test_file} ---\n")

ds = xr.open_dataset(test_file, engine='netcdf4')

# 2. Print detailed info for every variable
print(f"{'Short Name':<15} | {'Long Name':<40} | {'Units'}")
print("-" * 70)

for var in ds.data_vars:
    long_name = ds[var].attrs.get('long_name', 'No long name')
    units = ds[var].attrs.get('units', 'No units')
    print(f"{var:<15} | {long_name:<40} | {units}")

print("\n--- End of Variables ---")