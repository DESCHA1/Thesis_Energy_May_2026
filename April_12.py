import cdsapi
import os

c = cdsapi.Client()

# Numeric IDs for ERA5 Single Levels
# 165.128 = 10m u-wind
# 166.128 = 10m v-wind
# 169.128 = Surface solar radiation downwards (GHI)
# 235.128 = Surface diffuse solar radiation downwards (DHI)
variables = ['165.128', '166.128', '169.128', '235.128']

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'data_format': 'netcdf',
        'variable': variables, 
        'year': '2019',
        'month': '01',
        'day': [str(d).zfill(2) for d in range(1, 32)],
        'time': [f"{str(h).zfill(2)}:00" for h in range(24)],
        'area': [55.1, 5.8, 47.2, 15.2], 
        'grid': [0.5, 0.5], 
    },
    'era5_data_fixed.nc')
