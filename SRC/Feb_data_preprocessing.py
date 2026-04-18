import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf


# Load the dataset
df = pd.read_csv('time_series_60min_singleindex_filtered.csv')


# Inspect the data
print("Data Info:")
print(df.info())
print("\nFirst few rows:")
print(df.head())


# Check for missing values in the key columns
columns_to_check = [
  'utc_timestamp',
  'DE_load_actual_entsoe_transparency',
  'DE_solar_generation_actual',
  'DE_wind_generation_actual'
]
print("\nMissing values in key columns:")
print(df[columns_to_check].isnull().sum())


# 1. Reload Data and Preprocess
df = pd.read_csv('time_series_60min_singleindex_filtered.csv')
df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
df = df.set_index('utc_timestamp')


# 2. Preprocessing & Feature Engineering
# Target Calculation
df['Residual_Load'] = (
  df['DE_load_actual_entsoe_transparency'] -
  (df['DE_solar_generation_actual'].fillna(0) + df['DE_wind_generation_actual'].fillna(0))
)


# Focus on primary columns
cols = ['DE_load_actual_entsoe_transparency', 'DE_solar_generation_actual', 'DE_wind_generation_actual']
df_subset = df[cols].copy()


# Simple cleaning for EDA: Interpolate missing values
# For solar, NaN usually means zero or missing data. We will look at the first few rows.
# Filling NaNs
df_subset['DE_solar_generation_actual'] = df_subset['DE_solar_generation_actual'].fillna(0)
df_subset['DE_wind_generation_actual'] = df_subset['DE_wind_generation_actual'].interpolate(method='time')
df_subset['DE_load_actual_entsoe_transparency'] = df_subset['DE_load_actual_entsoe_transparency'].interpolate(method='time')


# Calculate Residual Load
df_subset['Residual_Load'] = df_subset['DE_load_actual_entsoe_transparency'] - (
  df_subset['DE_solar_generation_actual'] + df_subset['DE_wind_generation_actual']
)


# 1. Visualize Time Series for a sample week
sample_week = df_subset.loc['2015-06-01':'2015-06-07']
plt.figure(figsize=(12, 6))
plt.plot(sample_week.index, sample_week['DE_load_actual_entsoe_transparency'], label='Total Load', alpha=0.7)
plt.plot(sample_week.index, sample_week['Residual_Load'], label='Residual Load', linewidth=2, color='black')
plt.fill_between(sample_week.index, sample_week['DE_solar_generation_actual'], label='Solar', alpha=0.3)
plt.fill_between(sample_week.index, sample_week['DE_wind_generation_actual'], label='Wind', alpha=0.3)
plt.title('Components of Residual Load (Sample Week June 2015)')
plt.ylabel('MW')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('load_components_week.png')


# 2. Seasonality - Hourly Profiles
df_subset['hour'] = df_subset.index.hour
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_subset, x='hour', y='Residual_Load')
plt.title('Residual Load Distribution by Hour of Day')
plt.ylabel('Residual Load (MW)')
plt.savefig('residual_load_hourly.png')


# 3. Correlation Matrix
# Rename
df_subset = df[['DE_load_actual_entsoe_transparency', 'DE_solar_generation_actual', 'DE_wind_generation_actual', 'Residual_Load']].rename(
  columns={'DE_load_actual_entsoe_transparency': 'Electrical Demand',
       'DE_solar_generation_actual': 'Solar Generation',
       'DE_wind_generation_actual': 'Wind Generation',
       'Residual Load' : 'Residual Load'}
)
corr_data = df_subset.corr()


plt.figure(figsize=(12, 10))



# Create the heatmap
ax = sns.heatmap(corr_data,
         annot=True,
         cmap='coolwarm',
         fmt=".2f",
         cbar_kws={'label': 'Correlation Coefficient'})


# Rotate and Align X-axis labels
# 'ha=right' ensures the end of the word aligns with the grid tick
plt.xticks(rotation=45, ha='right', fontsize=10)


# Ensure Y-axis labels are readable
plt.yticks(rotation=0, fontsize=10)


plt.title('Correlation Matrix of German Energy Variables (2015-2019)', pad=20, fontsize=15)
plt.tight_layout()


plt.savefig('correlation_matrix.png')
plt.show


#Summary Statistics
print("Summary Statistics for Residual Load:")
print(df_subset['Residual_Load'].describe())



from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# 4. Weekly/Monthly Seasonality
df_subset['day_of_week'] = df_subset.index.dayofweek
df_subset['month'] = df_subset.index.month


fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.boxplot(data=df_subset, x='day_of_week', y='Residual_Load', ax=axes[0])
axes[0].set_title('Residual Load by Day of Week (0=Mon, 6=Sun)')
sns.boxplot(data=df_subset, x='month', y='Residual_Load', ax=axes[1])
axes[1].set_title('Residual Load by Month')
plt.savefig('seasonality_day_month.png')


# 5. Autocorrelation (ACF) - very important for PatchTST/TFT window size
# Using a subset or sampling for speed, but here full series is fine for ACF
plt.figure(figsize=(12, 5))
plot_acf(df_subset['Residual_Load'].dropna(), lags=48, ax=plt.gca())
plt.title('Autocorrelation of Residual Load (48 Hours)')
plt.savefig('acf_plot.png')


# 6. Stationary Check (Informal)
plt.figure(figsize=(15, 5))
df_subset['Residual_Load'].resample('D').mean().plot()
plt.title('Daily Average Residual Load (Long-term Trend)')
plt.ylabel('MW')
plt.savefig('daily_trend.png')


# Save the processed target for the user
df_subset[['Residual_Load']].to_csv('residual_load_target.csv')
print("Processed data saved to residual_load_target.csv")


# Time Features (Essential for TFT)
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month


# 3. Visualization: Sample Week
# This helps us see the daily solar "dips" and wind variability
sample = df.loc['2015-06-01':'2015-06-07']
plt.figure(figsize=(12, 5))
plt.plot(sample.index, sample['DE_load_actual_entsoe_transparency'], label='Total Load', alpha=0.5)
plt.plot(sample.index, sample['Residual_Load'], label='Residual Load', color='black', linewidth=2)
plt.fill_between(sample.index, sample['DE_solar_generation_actual'], label='Solar', alpha=0.2)
plt.title('Residual Load Components (Sample Week)')
plt.ylabel('MW')
plt.legend()
plt.savefig('residual_load_week.png')


# 4. Seasonality Analysis
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.boxplot(data=df, x='hour', y='Residual_Load', ax=axes[0])
axes[0].set_title('Hourly Seasonality')
sns.boxplot(data=df, x='day_of_week', y='Residual_Load', ax=axes[1])
axes[1].set_title('Weekly Seasonality (0=Mon, 6=Sun)')
plt.savefig('seasonality.png')


# 5. Autocorrelation (Critical for PatchTST 'patch' size)
plt.figure(figsize=(10, 4))
plot_acf(df['Residual_Load'].dropna(), lags=48, ax=plt.gca())
plt.title('Autocorrelation (48 Hours)')
plt.savefig('acf.png')
