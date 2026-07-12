import os
import pandas as pd
import numpy as np
import torch
import optuna
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Import CarbonTracker
from carbontracker.tracker import CarbonTracker

# 1. SETUP & CONFIG

CONFIG = {
    "dataset_path": "DATA/final_data_with_features.csv",
    "results_path": "DATA/optimized_final_results.csv",
    "test_results_path": "DATA/optimized_test_results.csv",
    "tuning_trials": 20,
    "n_splits": 3,
    "robustness_runs": 3,
    "encoder_multiplier": 2,
    "quantiles": [0.1, 0.5, 0.9]
}

SCALE_FEATURES = ["wind_u", "wind_v", "ghi"]

def scale_fold(train_df, val_df, features):
    scaler = StandardScaler()
    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    train_scaled[features] = scaler.fit_transform(train_df[features])
    val_scaled[features] = scaler.transform(val_df[features])
    return train_scaled, val_scaled, scaler

def compute_metrics(predictions, actuals):
    """Compute MAE, RMSE, SMAPE on the 0.5 quantile (median) predictions."""
    median_preds = predictions[:, :, 1]  # index 1 = 0.5 quantile
    mae = torch.mean(torch.abs(median_preds - actuals)).item()
    rmse = torch.sqrt(torch.mean((median_preds - actuals) ** 2)).item()
    smape = torch.mean(2 * torch.abs(median_preds - actuals) /
                      (torch.abs(median_preds) + torch.abs(actuals))).item()
    return mae, rmse, smape

# 2. DATA PREP
df = pd.read_csv(CONFIG["dataset_path"])
df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
df = df.sort_values('utc_timestamp').reset_index(drop=True)
df['time_idx'] = np.arange(len(df))
df['region'] = "DE"

# Hold out final 10% as test set
test_cutoff = int(len(df) * 0.90)
df_trainval = df.iloc[:test_cutoff].reset_index(drop=True)
df_test = df.iloc[test_cutoff:].reset_index(drop=True)
print(f"Train/val size: {len(df_trainval)} | Test size: {len(df_test)}")

TFT_FEATURES = {
    "target": "residual_load",
    "time_varying_known_reals": ["is_holiday", "is_weekend", "hour_sin", "hour_cos", "day_sin", "day_cos"],
    "time_varying_unknown_reals": ["residual_load", "wind_u", "wind_v", "ghi"],
    "static_categoricals": ["region"]
}

# 3. OPTUNA OBJECTIVE (Now accepts horizon)
def objective(trial, horizon):
    encoder_len = horizon * CONFIG["encoder_multiplier"]

    hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64, 128])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.3)

    tscv = TimeSeriesSplit(n_splits=CONFIG["n_splits"])
    train_idx, val_idx = next(tscv.split(df_trainval))

    train_df, val_df, _ = scale_fold(df_trainval.iloc[train_idx], df_trainval.iloc[val_idx], SCALE_FEATURES)

    training = TimeSeriesDataSet(
        train_df, time_idx="time_idx", target=TFT_FEATURES["target"], group_ids=["region"],
        max_encoder_length=encoder_len, max_prediction_length=horizon,
        static_categoricals=TFT_FEATURES["static_categoricals"],
        time_varying_known_reals=TFT_FEATURES["time_varying_known_reals"],
        time_varying_unknown_reals=TFT_FEATURES["time_varying_unknown_reals"],
        target_normalizer=GroupNormalizer(groups=["region"])
    )

    train_loader = training.to_dataloader(train=True, batch_size=64, num_workers=8)
    val_loader = TimeSeriesDataSet.from_dataset(training, val_df).to_dataloader(train=False, batch_size=64)

    tft = TemporalFusionTransformer.from_dataset(
        training, learning_rate=learning_rate, hidden_size=hidden_size,
        attention_head_size=4, dropout=dropout,
        loss=QuantileLoss(quantiles=CONFIG["quantiles"])
    )

    trainer = pl.Trainer(max_epochs=10, accelerator="gpu", devices=1, enable_checkpointing=False, logger=False)
    
    # EMISSION TRACKING FOR OPTUNA
    tracker = CarbonTracker(epochs=1, components="gpu", log_dir="DATA")
    tracker.epoch_start()
    
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    tracker.epoch_end()

    return trainer.callback_metrics["val_loss"].item()

# 4. OPTUNA EXECUTION
print("--- Starting Optuna Hyperparameter Optimization ---")
best_params_dict = {}

# Optimize for 24h and 168h
for tune_horizon in [24, 168]:
    print(f"\n>>> Tuning for horizon: {tune_horizon}h")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, tune_horizon), n_trials=CONFIG["tuning_trials"])
    best_params_dict[tune_horizon] = study.best_params
    print(f"   Best Params for {tune_horizon}h: {study.best_params}")

# Bypass optimization for 720h and apply 168h parameters
print("\n>>> Applying 168h optimized parameters to 720h horizon")
best_params_dict[720] = best_params_dict[168].copy()


# 5. CROSS-VALIDATION LOOP
horizons = [24, 168, 720]
tscv = TimeSeriesSplit(n_splits=CONFIG["n_splits"])

for horizon in horizons:
    print(f"\n{'='*40}\n>>> CROSS-VALIDATION: {horizon}h\n{'='*40}")
    encoder_len = horizon * CONFIG["encoder_multiplier"]
    params = best_params_dict[horizon]  # Pull the specific params for this horizon

    for fold, (train_idx, val_idx) in enumerate(tscv.split(df_trainval)):

        train_df, val_df, _ = scale_fold(
            df_trainval.iloc[train_idx],
            df_trainval.iloc[val_idx],
            SCALE_FEATURES
        )

        training = TimeSeriesDataSet(
            train_df, time_idx="time_idx", target=TFT_FEATURES["target"], group_ids=["region"],
            max_encoder_length=encoder_len, max_prediction_length=horizon,
            static_categoricals=TFT_FEATURES["static_categoricals"],
            time_varying_known_reals=TFT_FEATURES["time_varying_known_reals"],
            time_varying_unknown_reals=TFT_FEATURES["time_varying_unknown_reals"],
            target_normalizer=GroupNormalizer(groups=["region"])
        )
        train_loader = training.to_dataloader(train=True, batch_size=32, num_workers=8)
        val_loader = TimeSeriesDataSet.from_dataset(training, val_df).to_dataloader(train=False, batch_size=32)

        for run in range(CONFIG["robustness_runs"]):
            seed = 42 + run
            pl.seed_everything(seed)

            tft = TemporalFusionTransformer.from_dataset(
                training,
                learning_rate=params["learning_rate"],
                hidden_size=params["hidden_size"],
                dropout=params["dropout"],
                attention_head_size=4,
                loss=QuantileLoss(quantiles=CONFIG["quantiles"]),
            )

            trainer = pl.Trainer(
                max_epochs=30, accelerator="gpu", devices=1,
                callbacks=[EarlyStopping(monitor="val_loss", patience=8)],
                logger=False
            )

            # EMISSION TRACKING FOR CROSS-VALIDATION
            tracker = CarbonTracker(epochs=1, components="gpu", log_dir="DATA")
            tracker.epoch_start()
            
            trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
            
            tracker.epoch_end()

            predictions = tft.predict(val_loader, mode="quantiles")
            actuals = torch.cat([y[0] for x, y in val_loader])
            mae, rmse, smape = compute_metrics(predictions, actuals)

            res = {
                "horizon": horizon, "fold": fold+1, "run": run+1,
                "mae": mae, "rmse": rmse, "smape": smape,
                "val_loss": trainer.callback_metrics.get("val_loss").item()
            }

            pd.DataFrame([res]).to_csv(CONFIG["results_path"], mode='a', index=False,
                                       header=not os.path.exists(CONFIG["results_path"]))

            if run == 0:
                raw_predictions = tft.predict(val_loader, mode="raw", return_x=True)
                interpretation = tft.interpret_output(raw_predictions.output)
                figs = tft.plot_interpretation(interpretation)
                figs['static_variables'].get_figure().savefig(f"DATA/optimized_importance_{horizon}h_fold{fold+1}.png")
                plt.close('all')

            del tft, trainer
            torch.cuda.empty_cache()

# 6. FINAL TEST EVALUATION
print(f"\n{'='*40}\n>>> FINAL TEST EVALUATION\n{'='*40}")

for horizon in horizons:
    encoder_len = horizon * CONFIG["encoder_multiplier"]
    params = best_params_dict[horizon] # Pull the specific params for this horizon

    scaler = StandardScaler()
    full_trainval = df_trainval.copy()
    full_trainval[SCALE_FEATURES] = scaler.fit_transform(df_trainval[SCALE_FEATURES])
    test_scaled = df_test.copy()
    test_scaled[SCALE_FEATURES] = scaler.transform(df_test[SCALE_FEATURES])

    training = TimeSeriesDataSet(
        full_trainval, time_idx="time_idx", target=TFT_FEATURES["target"], group_ids=["region"],
        max_encoder_length=encoder_len, max_prediction_length=horizon,
        static_categoricals=TFT_FEATURES["static_categoricals"],
        time_varying_known_reals=TFT_FEATURES["time_varying_known_reals"],
        time_varying_unknown_reals=TFT_FEATURES["time_varying_unknown_reals"],
        target_normalizer=GroupNormalizer(groups=["region"])
    )
    train_loader = training.to_dataloader(train=True, batch_size=32, num_workers=8)
    test_loader = TimeSeriesDataSet.from_dataset(training, test_scaled).to_dataloader(train=False, batch_size=32)

    for run in range(CONFIG["robustness_runs"]):
        seed = 42 + run
        pl.seed_everything(seed)

        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=params["learning_rate"],
            hidden_size=params["hidden_size"],
            dropout=params["dropout"],
            attention_head_size=4,
            loss=QuantileLoss(quantiles=CONFIG["quantiles"]),
        )

        trainer = pl.Trainer(
            max_epochs=30, accelerator="gpu", devices=1,
            callbacks=[EarlyStopping(monitor="val_loss", patience=8)],
            logger=False
        )

        # EMISSION TRACKING FOR FINAL TEST
        tracker = CarbonTracker(epochs=1, components="gpu", log_dir="DATA")
        tracker.epoch_start()
        
        trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=test_loader)
        
        tracker.epoch_end()

        predictions = tft.predict(test_loader, mode="quantiles")
        actuals = torch.cat([y[0] for x, y in test_loader])
        mae, rmse, smape = compute_metrics(predictions, actuals)

        res = {
            "horizon": horizon, "run": run+1,
            "test_mae": mae, "test_rmse": rmse, "test_smape": smape,
            "test_loss": trainer.callback_metrics.get("val_loss").item()
        }

        pd.DataFrame([res]).to_csv(CONFIG["test_results_path"], mode='a', index=False,
                                   header=not os.path.exists(CONFIG["test_results_path"]))

        del tft, trainer
        torch.cuda.empty_cache()

print("   All Tasks Complete!")
