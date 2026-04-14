import cdsapi
import os
import calendar
import ctypes # Built-in Windows library to prevent sleep

# This line tells Windows the script is busy so it won't sleep
# 0x80000002 = ES_CONTINUOUS | ES_SYSTEM_REQUIRED
ctypes.windll.kernel32.SetThreadExecutionState(0x80000002)

c = cdsapi.Client()

years = ['2015', '2016', '2017', '2018', '2019']
variables = ['165.128', '166.128', '169.128', '235.128']

for year in years:
    for month in range(1, 13):
        month_str = str(month).zfill(2)
        target_file = f'era5_ger_{year}_{month_str}.nc'
        
        if os.path.exists(target_file):
            continue
            
        # Dynamically find the last day of the month (28, 30, or 31)
        last_day = calendar.monthrange(int(year), month)[1]
        days = [str(d).zfill(2) for d in range(1, last_day + 1)]
        
        print(f"--- Requesting {year}-{month_str} ---")
        
        try:
            c.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'data_format': 'netcdf',
                    'variable': variables,
                    'year': year,
                    'month': month_str,
                    'day': days,
                    'time': [f"{str(h).zfill(2)}:00" for h in range(24)],
                    'area': [55.1, 5.8, 47.2, 15.2], 
                    'grid': [0.5, 0.5], 
                },
                target_file)
        except Exception as e:
            print(f"Error on {year}-{month_str}: {e}")

# Return Windows to normal sleep behavior when finished
ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)