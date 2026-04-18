import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import scipy.stats as stats
import os
import matplotlib


# --- Step 0: Setup ---
output_dir = 'EDA_Results'
os.makedirs(output_dir, exist_ok=True)
plt.rcParams.update({'font.size': 12, 'figure.facecolor': 'white'})

# 1. LOAD AND MERGE
print("--- Step 1: Merging Datasets ---")
df_energy = pd.read_csv('DATA/cleaned_residual_load_2015_2019.csv')
df_weather = pd.read_csv('DATA/cleaned_era5_weather.csv')

df_energy['utc_timestamp'] = pd.to_datetime(df_energy['utc_timestamp'], utc=True)
df_weather['utc_timestamp'] = pd.to_datetime(df_weather['utc_timestamp'], utc=True)

df = pd.merge(df_energy, df_weather, on='utc_timestamp', how='inner')
df.set_index('utc_timestamp', inplace=True)
df = df.sort_index()

features = ['residual_load', 'wind_u', 'wind_v', 'ghi']
print(f"Merge Complete. Final Dataset Rows: {len(df)}")

# 2. MULTI-LEVEL SEASONALITY (14-Day Snapshot)
print("--- Step 2: Seasonality Snapshot ---")
plt.figure(figsize=(15, 5))
df['residual_load'].iloc[:24*14].plot(color='#2b7bba', lw=2)
plt.title('Residual Load: 14-Day Snapshot (Daily/Weekly Seasonality)', pad=15)
plt.ylabel('MW')
plt.grid(True, alpha=0.3)
plt.savefig(f'{output_dir}/01_seasonality_snapshot.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. TIME SERIES DECOMPOSITION
print("--- Step 3: Running Decomposition ---")
decomp = seasonal_decompose(df['residual_load'], model='additive', period=168)
fig = decomp.plot()
fig.set_size_inches(12, 10)
plt.tight_layout()
plt.savefig(f'{output_dir}/02_decomposition.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. AUTOCORRELATION (ACF/PACF)
print("--- Step 4: ACF and PACF ---")
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
plot_acf(df['residual_load'], lags=72, ax=axes[0], color='#d62728')
plot_pacf(df['residual_load'], lags=72, ax=axes[1], method='ywm', color='#2ca02c')
axes[0].set_title('Autocorrelation (ACF)')
axes[1].set_title('Partial Autocorrelation (PACF)')
plt.savefig(f'{output_dir}/03_acf_pacf.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. BOXPLOTS (Range and Skewness)
print("--- Step 5: Boxplots ---")
fig, axes = plt.subplots(1, len(features), figsize=(18, 6))
for i, col in enumerate(features):
    sns.boxplot(y=df[col], ax=axes[i], color='skyblue', width=0.5)
    axes[i].set_title(f'{col}\nSkew: {df[col].skew():.2f}')
    axes[i].set_ylabel('')
plt.tight_layout()
plt.savefig(f'{output_dir}/04_boxplots.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. DISTRIBUTION & Q-Q PLOTS (Normalization Justification)
print("--- Step 6: Distribution Analysis ---")
fig, axes = plt.subplots(2, len(features), figsize=(20, 10))
for i, col in enumerate(features):
    sns.histplot(df[col], kde=True, ax=axes[0, i], color='salmon')
    stats.probplot(df[col].dropna(), dist="norm", plot=axes[1, i])
    axes[0, i].set_title(f'Histogram: {col}')
    axes[1, i].set_title(f'Q-Q Plot: {col}')
plt.tight_layout()
plt.savefig(f'{output_dir}/05_distributions_qq.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. CORRELATION HEATMAP
print("--- Step 7: Correlation Matrix ---")
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='RdBu_r', center=0, fmt='.2f', square=True, cbar_kws={"shrink": .8})
plt.title('Correlation Matrix: Weather vs. Residual Load', fontsize=16, pad=20)
plt.savefig(f'{output_dir}/06_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. SCATTER PLOTS WITH REGRESSION LINES
print("--- Step 8: Scatter/Regression Analysis ---")
weather_vars = ['wind_u', 'wind_v', 'ghi']
fig, axes = plt.subplots(1, len(weather_vars), figsize=(20, 6))
for i, col in enumerate(weather_vars):
    sns.regplot(x=df[col], y=df['residual_load'], ax=axes[i], 
                scatter_kws={'color': '#2b7bba', 'alpha': 0.2, 's': 10}, 
                line_kws={'color': '#d62728', 'lw': 2})
    axes[i].set_title(f'{col} vs. Residual Load')
plt.tight_layout()
plt.savefig(f'{output_dir}/07_scatter_regression.png', dpi=300, bbox_inches='tight')
plt.show()

# 9. DAILY LOAD PROFILES BY SEASON
print("--- Step 9: Daily Profiles ---")
df_eda = df.copy()
df_eda['hour'] = df_eda.index.hour
df_eda['month'] = df_eda.index.month
df_eda['season'] = df_eda['month'].map(lambda m: 'Winter' if m in [12,1,2] else 'Spring' if m in [3,4,5] else 'Summer' if m in [6,7,8] else 'Autumn')

plt.figure(figsize=(12, 6))
sns.lineplot(data=df_eda, x='hour', y='residual_load', hue='season', palette='RdBu_r', hue_order=['Winter', 'Autumn', 'Spring', 'Summer'])
plt.title('Average Daily Residual Load Profile by Season')
plt.grid(True, alpha=0.3)
plt.savefig(f'{output_dir}/08_daily_profiles.png', dpi=300, bbox_inches='tight')
plt.show()

# 10. LAG PLOT (1-Hour Persistence)
print("--- Step 10: Lag Plot ---")
plt.figure(figsize=(8, 8))
pd.plotting.lag_plot(df['residual_load'], lag=1, c='#2b7bba', alpha=0.1)
plt.title('Persistence Check: Load(t) vs Load(t-1)')
plt.savefig(f'{output_dir}/09_lag_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# 11. WEEKLY SEASONALITY BOXPLOT
print("--- Step 11: Weekly Boxplot ---")
df_eda['day'] = df_eda.index.day_name()
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_eda, x='day', y='residual_load', order=day_order, palette='Blues')
plt.title('Load Distribution by Day of Week')
plt.savefig(f'{output_dir}/10_weekly_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()

# SAVE DATA
df.to_csv('DATA/final_merged_data_for_modeling.csv')
print(f"\n ALL STEPS COMPLETE. 10 figures saved in '{output_dir}/'.")