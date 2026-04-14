import cdsapi

c = cdsapi.Client()

# Numeric IDs for Wind U, Wind V, GHI, and DHI
variables = ['165.128', '166.128', '169.128', '235.128']
target_file = 'era5_ger_2018_10_NATIVE.nc'

print("--- Rescue Attempt: Requesting Native 0.25 Grid for Oct 2018 ---")

try:
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'data_format': 'netcdf',
            'variable': variables,
            'year': '2018',
            'month': '10',
            'day': [str(d).zfill(2) for d in range(1, 32)],
            'time': [f"{str(h).zfill(2)}:00" for h in range(24)],
            'area': [55.1, 5.8, 47.2, 15.2], 
            # CHANGED TO NATIVE RESOLUTION
            'grid': [0.25, 0.25], 
        },
        target_file)
    print(f"Check file size. At 0.25 grid, it should be significantly LARGER (~6-8MB).")
except Exception as e:
    print(f"Native grid attempt failed: {e}")