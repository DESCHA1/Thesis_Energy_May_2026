import os
import pandas as pd
import numpy as np
import torch
import optuna
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, GroupNormalizer, MAE, RMSE, SMAPE
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

# 1. SETUP & CONFIG
CONFIG = {
    "dataset_path": "final_data_with_features.csv", 
    "results_path": "optimized_final_results.csv",
    "tuning_trials": 20,           
    "n_splits": 3,
    "robustness_runs": 3,          # Number of seeds for final evaluation
    "encoder_multiplier": 2
}

# 2. DATA PREP
df = pd.read_csv(CONFIG["dataset_path"])
df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
df = df.sort_values('utc_timestamp').reset_index(drop=True)
df['time_idx'] = np.arange(len(df))
df['region'] = "DE"

TFT_FEATURES = {
    "target": "residual_load",
    "time_varying_known_reals": ["is_holiday", "is_weekend", "hour_sin", "hour_cos", "day_sin", "day_cos"],
    "time_varying_unknown_reals": ["residual_load", "wind_u", "wind_v", "ghi"],
    "static_categoricals": ["region"]
}

# 3. OPTUNA OBJECTIVE (Tuning on 24h Horizon, Fold 1)
def objective(trial):
    horizon = 24
    encoder_len = horizon * CONFIG["encoder_multiplier"]
    
    # Hyperparameters to tune
    hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64, 128])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.3)

    tscv = TimeSeriesSplit(n_splits=CONFIG["n_splits"])
    train_idx, val_idx = next(tscv.split(df)) # Tuning on Fold 1 for efficiency
    
    training = TimeSeriesDataSet(
        df.iloc[train_idx], time_idx="time_idx", target=TFT_FEATURES["target"], group_ids=["region"],
        max_encoder_length=encoder_len, max_prediction_length=horizon,
        static_categoricals=TFT_FEATURES["static_categoricals"],
        time_varying_known_reals=TFT_FEATURES["time_varying_known_reals"],
        time_varying_unknown_reals=TFT_FEATURES["time_varying_unknown_reals"],
        target_normalizer=GroupNormalizer(groups=["region"])
    )
    
    val_loader = TimeSeriesDataSet.from_dataset(training, df.iloc[val_idx]).to_dataloader(train=False, batch_size=64)
    train_loader = training.to_dataloader(train=True, batch_size=64, num_workers=8)

    tft = TemporalFusionTransformer.from_dataset(
        training, learning_rate=learning_rate, hidden_size=hidden_size, 
        attention_head_size=4, dropout=dropout, loss=MAE()
    )

    trainer = pl.Trainer(max_epochs=10, accelerator="gpu", devices=1, enable_checkpointing=False, logger=False)
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    return trainer.callback_metrics["val_loss"].item()

# 4. EXECUTION
print("--- Starting Optuna Hyperparameter Optimization ---")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=CONFIG["tuning_trials"])
best_params = study.best_params
print(f"🏆 Optimization Complete. Best Params: {best_params}")

# 5. FINAL CROSS-VALIDATION LOOP (All Horizons, All Folds)
horizons = [24, 168, 720]
tscv = TimeSeriesSplit(n_splits=CONFIG["n_splits"])

for horizon in horizons:
    print(f"\n{'='*40}\n>>> FINAL EVALUATION: {horizon}h\n{'='*40}")
    encoder_len = horizon * CONFIG["encoder_multiplier"]

    for fold, (train_idx, val_idx) in enumerate(tscv.split(df)):
        training = TimeSeriesDataSet(
            df.iloc[train_idx], time_idx="time_idx", target=TFT_FEATURES["target"], group_ids=["region"],
            max_encoder_length=encoder_len, max_prediction_length=horizon,
            static_categoricals=TFT_FEATURES["static_categoricals"],
            time_varying_known_reals=TFT_FEATURES["time_varying_known_reals"],
            time_varying_unknown_reals=TFT_FEATURES["time_varying_unknown_reals"],
            target_normalizer=GroupNormalizer(groups=["region"])
        )
        val_loader = TimeSeriesDataSet.from_dataset(training, df.iloc[val_idx]).to_dataloader(train=False, batch_size=32)
        train_loader = training.to_dataloader(train=True, batch_size=32, num_workers=8)

        for run in range(CONFIG["robustness_runs"]):
            seed = 42 + run
            pl.seed_everything(seed)
            
            tft = TemporalFusionTransformer.from_dataset(
                training, 
                learning_rate=best_params["learning_rate"], 
                hidden_size=best_params["hidden_size"], 
                dropout=best_params["dropout"],
                attention_head_size=4,
                loss=MAE(),
                logging_metrics=torch.nn.ModuleList([MAE(), RMSE(), SMAPE()])
            )

            trainer = pl.Trainer(
                max_epochs=30, accelerator="gpu", devices=1,
                callbacks=[EarlyStopping(monitor="val_loss", patience=8)],
                logger=False
            )
            
            trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

            # Metrics
            m = trainer.callback_metrics
            res = {"horizon": horizon, "fold": fold+1, "run": run+1, 
                   "mae": m.get("val_mae").item(), "rmse": m.get("val_rmse").item(), "smape": m.get("val_smape").item()}
            
            pd.DataFrame([res]).to_csv(CONFIG["results_path"], mode='a', index=False, header=not os.path.exists(CONFIG["results_path"]))

            # Save Feature Importance Plot for the first run of each horizon
            if run == 0:
                raw_predictions = tft.predict(val_loader, mode="raw", return_x=True)
                interpretation = tft.interpret_output(raw_predictions.output)
                figs = tft.plot_interpretation(interpretation)
                # Static variables plot
                figs['static_variables'].get_figure().savefig(f"importance_{horizon}h_fold{fold+1}.png")
                plt.close('all')

            del tft, trainer
            torch.cuda.empty_cache()

print("🏁 All Tasks Complete!")