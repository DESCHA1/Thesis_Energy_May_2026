import pandas as pd

# 1. Load the RAW data
df_raw = pd.read_csv('DATA/time_series_60min_singleindex_filtered.csv')

# 2. Select the exact columns corresponding to my Appendix table
cols_for_appendix = [
    'DE_load_actual_entsoe_transparency',
    'DE_solar_generation_actual',
    'DE_wind_generation_actual',
    'DE_wind_onshore_generation_actual',
    'DE_wind_offshore_generation_actual'
]

# 3. Generate describe() with the custom 1%, 50%, and 99% percentiles most useful in power systems analysis
summary_stats = df_raw[cols_for_appendix].describe(percentiles=[0.01, 0.50, 0.99])

# 4. Reorder the rows 
row_order = ['count', 'mean', 'std', 'min', '1%', '50%', '99%', 'max']
summary_stats = summary_stats.loc[row_order]

# 5. Round everything to 2 decimal places
summary_stats = summary_stats.round(2)

# 6. Display the results in the console
print(summary_stats)

# 7. Save to csv
summary_stats.to_csv('appendix_summary_stats.csv')
print("\n Summary statistics successfully saved to 'appendix_summary_stats.csv'")