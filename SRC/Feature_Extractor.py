import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 1. IRON-CLAD PATH RESOLUTION
# ==============================================================================
# Automatically find the root directory (THESIS_ENERGY_MAY_2026)
# This finds the directory of this script (SRC) and goes one level up to the root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Build the absolute path to your feature_importance.csv
csv_path = os.path.join(ROOT_DIR, "SRC", "MAY_5_optimized.py", "feature_importance.csv")

print(f"[INFO] Resolving path to data: {csv_path}")

# Load the data your 20-hour run already calculated
try:
    df_importance = pd.read_csv(csv_path)
    print("[SUCCESS] Feature importance data loaded successfully.\n")
except FileNotFoundError:
    print(f"[ERROR] Could not find the file at {csv_path}. Please check your folder names.")
    exit()

# ==============================================================================
# 2. DEFINE TRUE FEATURE NAMES (Alphabetical Sorting Order)
# ==============================================================================
# Sorted alphabetically according to PyTorch Forecasting internal schema layout
decoder_names = ['day_cos', 'day_sin', 'hour_cos', 'hour_sin', 'is_holiday', 'is_weekend']
encoder_names = ['day_cos', 'day_sin', 'ghi', 'hour_cos', 'hour_sin', 'is_holiday', 'is_weekend', 'residual_load', 'wind_u', 'wind_v']

# ==============================================================================
# 3. FILTER & MAP DECODER VARIABLES (24h Horizon, Fold 1 Example)
# ==============================================================================
print("[PROCESSING] Filtering and mapping Decoder variables...")

filtered_decoder = df_importance[
    (df_importance["label"] == "cv_24h_fold1") & 
    (df_importance["importance_type"] == "decoder_variables")
].sort_values("feature_index").copy()

# Inject the real column names mapping directly to the feature indices
if len(filtered_decoder) == len(decoder_names):
    filtered_decoder["feature_name"] = decoder_names
else:
    print(f"[WARN] Decoder weight count mismatch. Found {len(filtered_decoder)}, expected {len(decoder_names)}.")
    filtered_decoder["feature_name"] = [f"feature_{i}" for i in range(len(filtered_decoder))]

# ==============================================================================
# 4. FILTER & MAP ENCODER VARIABLES (24h Horizon, Fold 1 Example)
# ==============================================================================
print("[PROCESSING] Filtering and mapping Encoder variables...")

filtered_encoder = df_importance[
    (df_importance["label"] == "cv_24h_fold1") & 
    (df_importance["importance_type"] == "encoder_variables")
].sort_values("feature_index").copy()

# Inject the real column names mapping directly to the feature indices
if len(filtered_encoder) == len(encoder_names):
    filtered_encoder["feature_name"] = encoder_names
else:
    print(f"[WARN] Encoder weight count mismatch. Found {len(filtered_encoder)}, expected {len(encoder_names)}.")
    filtered_encoder["feature_name"] = [f"feature_{i}" for i in range(len(filtered_encoder))]

# ==============================================================================
# 5. GENERATE AND SAVE PLOTS
# ==============================================================================
# Make sure an output directory for plots exists inside your root repo folder
output_plot_dir = os.path.join(ROOT_DIR, "PLOTS")
os.makedirs(output_plot_dir, exist_ok=True)

# --- Plot 1: Decoder Variable Importance ---
plt.figure(figsize=(10, 4), dpi=150)
plt.barh(filtered_decoder["feature_name"], filtered_decoder["importance"], color="#1f77b4", edgecolor='none', height=0.6)
plt.xlabel("Importance")
plt.title("Decoder Variable Importance — 24h Horizon (Fold 1)")
plt.gca().invert_yaxis()  # Keeps feature_0 mapping ('day_cos') at the top
plt.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()

decoder_plot_save_path = os.path.join(output_plot_dir, "named_decoder_importance_24h_fold1.png")
plt.savefig(decoder_plot_save_path, bbox_inches="tight")
print(f"[SAVED] Named Decoder chart exported to: {decoder_plot_save_path}")
plt.close()

# --- Plot 2: Encoder Variable Importance ---
plt.figure(figsize=(10, 5), dpi=150)
plt.barh(filtered_encoder["feature_name"], filtered_encoder["importance"], color="#2ca02c", edgecolor='none', height=0.6)
plt.xlabel("Importance")
plt.title("Encoder Variable Importance — 24h Horizon (Fold 1)")
plt.gca().invert_yaxis()  # Keeps feature