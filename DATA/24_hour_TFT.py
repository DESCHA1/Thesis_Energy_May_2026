import os
import pandas as pd
import numpy as np
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, GroupNormalizer, MAE, RMSE, SMAPE
from sklearn.model_selection import TimeSeriesSplit

# 1. SETUP & CONFIG (Test Lap Version)
CONFIG = {
    "file_name": "DATA/final_data_with_features.csv",
    "horizons": [24],                 # TEST LAP: Only 24h
    "n_splits": 3,                    
    "robustness_runs": 3,
    "batch_size": 16,                 
    "max_epochs": 30,                 # Reduced to 30 for laptop safety
    "patience": 8,                    # Slightly tighter patience for test
    "encoder_multiplier": 2,
    "tft_hidden_size": 16,            
    "tft_attention_heads": 4
}

# 2. DATA PREP
df = pd.read_csv(CONFIG["file_name"])
df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
df = df.sort_values('utc_timestamp').reset_index(drop=True)
df['time_idx'] = np.arange(len(df))
df['region'] = "DE"

# 3. FEATURE MAPPING
TFT_FEATURES = {
    "target": "residual_load",
    "time_varying_known_reals": ["is_holiday", "is_weekend", "hour_sin", "hour_cos", "day_sin", "day_cos"],
    "time_varying_unknown_reals": ["residual_load", "wind_u", "wind_v", "ghi"],
    "static_categoricals": ["region"]
}

# 4. CROSS-VALIDATION LOOP
tscv = TimeSeriesSplit(n_splits=CONFIG["n_splits"])
results_file = "test_lap_results.csv"

for horizon in CONFIG["horizons"]:
    print(f"\n>>> STARTING TEST LAP: {horizon}h")
    encoder_len = horizon * CONFIG["encoder_multiplier"]

    for fold, (train_idx, val_idx) in enumerate(tscv.split(df)):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        training = TimeSeriesDataSet(
            train_df,
            time_idx="time_idx",
            target=TFT_FEATURES["target"],
            group_ids=["region"],
            max_encoder_length=encoder_len,
            max_prediction_length=horizon,
            static_categoricals=TFT_FEATURES["static_categoricals"],
            time_varying_known_reals=TFT_FEATURES["time_varying_known_reals"],
            time_varying_unknown_reals=TFT_FEATURES["time_varying_unknown_reals"],
            target_normalizer=GroupNormalizer(groups=["region"])
        )
        
        validation = TimeSeriesDataSet.from_dataset(training, val_df)
        train_loader = training.to_dataloader(train=True, batch_size=CONFIG["batch_size"], num_workers=7)
        val_loader = validation.to_dataloader(train=False, batch_size=CONFIG["batch_size"], num_workers=7)

        for run in range(CONFIG["robustness_runs"]):
            seed = 42 + run
            pl.seed_everything(seed)
            print(f"  Fold {fold+1} | Run {run+1} (Seed {seed})")

            tft = TemporalFusionTransformer.from_dataset(
                training,
                learning_rate=0.03,
                hidden_size=CONFIG["tft_hidden_size"], 
                attention_head_size=CONFIG["tft_attention_heads"],
                dropout=0.1,
                loss=MAE(),
                logging_metrics=torch.nn.ModuleList([MAE(), RMSE(), SMAPE()])
            )

            early_stop = EarlyStopping(monitor="val_loss", patience=CONFIG["patience"], mode="min")
            trainer = pl.Trainer(
                max_epochs=CONFIG["max_epochs"],
                accelerator="auto",
                callbacks=[early_stop],
                enable_model_summary=False,
                logger=False 
            )

            trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

            metrics = trainer.callback_metrics
            new_row = {
                "horizon": horizon, "fold": fold+1, "run": run+1,
                "mae": metrics["val_mae"].item(),
                "rmse": metrics["val_rmse"].item(),
                "smape": metrics["val_smape"].item()
            }
            
            pd.DataFrame([new_row]).to_csv(results_file, mode='a', index=False, header=not os.path.exists(results_file))

            # Cleanup to keep laptop cool
            del tft
            del trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

print(f"\n✅ Test lap complete. Check '{results_file}' for your metrics!")