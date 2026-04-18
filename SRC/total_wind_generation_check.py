import pandas as pd

file_path = r'C:\Users\desch\.vscode\THESIS_ENERGY_2026\DATA\time_series_60min_singleindex_filtered.csv'

# Load your file
df = pd.read_csv(file_path)

# 1. Fill NaNs with 0 to safely add them up
onshore = df['DE_wind_onshore_generation_actual'].fillna(0)
offshore = df['DE_wind_offshore_generation_actual'].fillna(0)
total = df['DE_wind_generation_actual'].fillna(0)

# 2. Calculate the manual sum
calculated_sum = onshore + offshore

# 3. Check where the manual sum exactly matches the reported total
matches = (calculated_sum.round(2) == total.round(2))

# 4. Filter for rows where there is actually wind data
has_data = total > 0
match_percentage = (matches[has_data].sum() / has_data.sum()) * 100

print(f"Match rate: {match_percentage:.2f}%")