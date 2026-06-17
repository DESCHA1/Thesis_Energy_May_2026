import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the data your 20-hour run already calculated
df_importance = pd.read_csv("DATA/TFT_run_02/feature_importance.csv")

# 2. Explicitly define your true variable names in their alphabetical order
decoder_names = ['day_cos', 'day_sin', 'hour_cos', 'hour_sin', 'is_holiday', 'is_weekend']
encoder_names = ['day_cos', 'day_sin', 'ghi', 'hour_cos', 'hour_sin', 'is_holiday', 'is_weekend', 'residual_load', 'wind_u', 'wind_v']

# 3. Filter for the specific plot you want (e.g., 24h horizon, Fold 1, Decoder Variables)
filtered_df = df_importance[
    (df_importance["label"] == "cv_24h_fold1") & 
    (df_importance["importance_type"] == "decoder_variables")
].sort_values("feature_index")

# 4. Map the true column names directly to the indices
filtered_df["feature_name"] = decoder_names

# 5. Plot instantly
plt.figure(figsize=(10, 4), dpi=150)
plt.barh(filtered_df["feature_name"], filtered_df["importance"], color="#1f77b4")
plt.xlabel("Importance")
plt.title("Decoder Variables — 24h Fold 1 (With Real Names)")
plt.gca().invert_yaxis()  # Keeps feature_0 at the top if desired
plt.tight_layout()
plt.savefig("DATA/TFT_run_02/named_decoder_importance_24h_fold1.png")
plt.show()