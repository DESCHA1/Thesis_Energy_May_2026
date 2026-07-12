import os
import sys
import json
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import CarbonTracker
from carbontracker.tracker import CarbonTracker
from carbontracker import parser as ct_parser


# 0. Logging Setup

RUN_DIR = "DATA/run_03"
os.makedirs(RUN_DIR, exist_ok=True)
LOG_PATH = f"{RUN_DIR}/run_log.txt"

class Tee:
    """Mirrors writes to both a file and the original stream."""
    def __init__(self, stream, filepath):
        self.stream = stream
        self.file = open(filepath, "a", buffering=1)   # line-buffered

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def isatty(self):
        return False

sys.stdout = Tee(sys.stdout, LOG_PATH)
sys.stderr = Tee(sys.stderr, LOG_PATH)


# 1. Setup and Configuration

CONFIG = {
    "dataset_path": "DATA/final_data_with_features.csv",
    "results_path": f"{RUN_DIR}/optimized_final_results.csv",
    "test_results_path": f"{RUN_DIR}/optimized_test_results.csv",
    # --- NEW output paths ---
    "best_params_path": f"{RUN_DIR}/best_hyperparams.json",
    "carbon_summary_path": f"{RUN_DIR}/carbon_summary.csv",
    "feature_importance_path": f"{RUN_DIR}/feature_importance.csv",
    # ------------------------
    "tuning_trials": 20,
    "n_splits": 3,
    "robustness_runs": 3,
    "encoder_multiplier": 2,
    "quantiles": [0.1, 0.5, 0.9]
}

SCALE_FEATURES = ["wind_u", "wind_v", "ghi"]

# Carbon Tracker

carbon_records = []   # accumulated across the whole run

def make_tracker(label: str) -> CarbonTracker:
    """Return a fresh CarbonTracker that logs to DATA/run_02/carbontracker/."""
    os.makedirs(f"{RUN_DIR}/carbontracker", exist_ok=True)
    return CarbonTracker(
        epochs=1,
        components="gpu",
        log_dir=f"{RUN_DIR}/carbontracker",
        monitor_epochs=1,
        verbose=2,
    )

def record_carbon(tracker: CarbonTracker, label: str):
    """
    After tracker.epoch_end() call this to pull the latest log entry and
    append it to carbon_records so we can save a tidy CSV at the end.
    """
    try:
        # carbontracker writes a JSON log file inside log_dir
        log_dir = f"{RUN_DIR}/carbontracker"
        logs = sorted(
            [f for f in os.listdir(log_dir) if f.endswith(".json")],
            key=lambda f: os.path.getmtime(os.path.join(log_dir, f))
        )
        if logs:
            with open(os.path.join(log_dir, logs[-1])) as fh:
                data = json.load(fh)
            # carbontracker v2 structure: list of epoch dicts
            epoch_data = data[-1] if isinstance(data, list) else data
            carbon_records.append({
                "label": label,
                "duration_s": epoch_data.get("duration", None),
                "energy_kWh": epoch_data.get("actual", {}).get("energy (kWh)", None),
                "co2_g": epoch_data.get("actual", {}).get("co2eq (g)", None),
            })
    except Exception as e:
        # Never let carbon-logging failure abort the main run
        carbon_records.append({"label": label, "duration_s": None,
                                "energy_kWh": None, "co2_g": None,
                                "error": str(e)})


def save_carbon_summary():
    df = pd.DataFrame(carbon_records)
    df.to_csv(CONFIG["carbon_summary_path"], index=False)
    print(f"[SAVED] Carbon summary → {CONFIG['carbon_summary_path']}")



# FEATURE IMPORTANCE HELPER

importance_records = []   # accumulated across the whole run

def record_importance(interpretation: dict, label: str,
                      save_plot: bool = True, plot_path: str = None):
    """
    Extract scalar importance values from a TFT interpretation dict and
    append them to importance_records.  Optionally save the bar chart.
    """
    for key, tensor in interpretation.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        vals = tensor.detach().cpu().numpy()
        # vals shape: (n_features,) after mean over batch
        if vals.ndim > 1:
            vals = vals.mean(axis=0)
        for i, v in enumerate(vals):
            importance_records.append({
                "label": label,
                "importance_type": key,
                "feature_index": i,
                "importance": float(v),
            })

    if save_plot and plot_path:
        try:
            figs = None
            # interpretation is already the dict returned by interpret_output
            # We need the raw output to call plot_interpretation — caller passes figs directly
            pass
        except Exception:
            pass


def save_importance_csv():
    if importance_records:
        df = pd.DataFrame(importance_records)
        df.to_csv(CONFIG["feature_importance_path"], index=False)
        print(f"[SAVED] Feature importance → {CONFIG['feature_importance_path']}")
    else:
        print("[WARN] No feature importance records to save.")



# 2. UTILITY FUNCTIONS

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


def append_csv(df_row: dict, path: str):
    """Append a single-row dict to a CSV, writing the header only once."""
    pd.DataFrame([df_row]).to_csv(
        path, mode="a", index=False,
        header=not os.path.exists(path)
    )



# 3. DATA PREP

df = pd.read_csv(CONFIG["dataset_path"])
df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"])
df = df.sort_values("utc_timestamp").reset_index(drop=True)
df["time_idx"] = np.arange(len(df))
df["region"] = "DE"

test_cutoff = int(len(df) * 0.90)
df_trainval = df.iloc[:test_cutoff].reset_index(drop=True)
df_test = df.iloc[test_cutoff:].reset_index(drop=True)
print(f"Train/val size: {len(df_trainval)} | Test size: {len(df_test)}", flush=True)

TFT_FEATURES = {
    "target": "residual_load",
    "time_varying_known_reals": [
        "is_holiday", "is_weekend", "hour_sin", "hour_cos", "day_sin", "day_cos"
    ],
    "time_varying_unknown_reals": ["residual_load", "wind_u", "wind_v", "ghi"],
    "static_categoricals": ["region"],
}



# 4. OPTUNA OBJECTIVE

def objective(trial, horizon):
    encoder_len = horizon * CONFIG["encoder_multiplier"]

    hidden_size    = trial.suggest_categorical("hidden_size", [16, 32, 64, 128])
    learning_rate  = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    dropout        = trial.suggest_float("dropout", 0.1, 0.3)

    tscv = TimeSeriesSplit(n_splits=CONFIG["n_splits"])
    train_idx, val_idx = next(tscv.split(df_trainval))
    train_df, val_df, _ = scale_fold(
        df_trainval.iloc[train_idx], df_trainval.iloc[val_idx], SCALE_FEATURES
    )

    training = TimeSeriesDataSet(
        train_df, time_idx="time_idx", target=TFT_FEATURES["target"],
        group_ids=["region"],
        max_encoder_length=encoder_len, max_prediction_length=horizon,
        static_categoricals=TFT_FEATURES["static_categoricals"],
        time_varying_known_reals=TFT_FEATURES["time_varying_known_reals"],
        time_varying_unknown_reals=TFT_FEATURES["time_varying_unknown_reals"],
        target_normalizer=GroupNormalizer(groups=["region"]),
    )
    train_loader = training.to_dataloader(train=True, batch_size=64, num_workers=8)
    val_loader   = TimeSeriesDataSet.from_dataset(training, val_df).to_dataloader(
        train=False, batch_size=64
    )

    tft = TemporalFusionTransformer.from_dataset(
        training, learning_rate=learning_rate, hidden_size=hidden_size,
        attention_head_size=4, dropout=dropout,
        loss=QuantileLoss(quantiles=CONFIG["quantiles"]),
    )
    trainer = pl.Trainer(
        max_epochs=10, accelerator="gpu", devices=1,
        enable_checkpointing=False, logger=False,
    )

    label = f"optuna_horizon{horizon}_trial{trial.number}"
    tracker = make_tracker(label)
    tracker.epoch_start()
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
    tracker.epoch_end()
    record_carbon(tracker, label)

    return trainer.callback_metrics["val_loss"].item()



# 5. OPTUNA EXECUTION

print("\n--- Starting Optuna Hyperparameter Optimization ---", flush=True)
best_params_dict = {}

for tune_horizon in [24, 168]:
    print(f"\n>>> Tuning for horizon: {tune_horizon}h", flush=True)
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial, h=tune_horizon: objective(trial, h),
        n_trials=CONFIG["tuning_trials"],
    )
    best_params_dict[tune_horizon] = study.best_params
    print(f"   Best Params for {tune_horizon}h: {study.best_params}", flush=True)

# Bypass optimization for 720h
print("\n>>> Applying 168h optimized parameters to 720h horizon", flush=True)
best_params_dict[720] = best_params_dict[168].copy()


# SAVE BEST HYPERPARAMETERS TO JSON 

serialisable = {str(k): v for k, v in best_params_dict.items()}
with open(CONFIG["best_params_path"], "w") as fh:
    json.dump(serialisable, fh, indent=2)
print(f"[SAVED] Best hyperparameters → {CONFIG['best_params_path']}", flush=True)

# Also log them clearly in the run log
print("\n===== BEST HYPERPARAMETERS SUMMARY =====", flush=True)
for horizon, params in best_params_dict.items():
    print(f"  {horizon}h : {params}", flush=True)
print("=========================================\n", flush=True)


# 6. CROSS-VALIDATION LOOP

horizons = [24, 168, 720]
tscv = TimeSeriesSplit(n_splits=CONFIG["n_splits"])

for horizon in horizons:
    print(f"\n{'='*40}\n>>> CROSS-VALIDATION: {horizon}h\n{'='*40}", flush=True)
    encoder_len = horizon * CONFIG["encoder_multiplier"]
    params = best_params_dict[horizon]

    for fold, (train_idx, val_idx) in enumerate(tscv.split(df_trainval)):
        train_df, val_df, _ = scale_fold(
            df_trainval.iloc[train_idx],
            df_trainval.iloc[val_idx],
            SCALE_FEATURES,
        )

        training = TimeSeriesDataSet(
            train_df, time_idx="time_idx", target=TFT_FEATURES["target"],
            group_ids=["region"],
            max_encoder_length=encoder_len, max_prediction_length=horizon,
            static_categoricals=TFT_FEATURES["static_categoricals"],
            time_varying_known_reals=TFT_FEATURES["time_varying_known_reals"],
            time_varying_unknown_reals=TFT_FEATURES["time_varying_unknown_reals"],
            target_normalizer=GroupNormalizer(groups=["region"]),
        )
        train_loader = training.to_dataloader(train=True, batch_size=32, num_workers=8)
        val_loader   = TimeSeriesDataSet.from_dataset(training, val_df).to_dataloader(
            train=False, batch_size=32
        )

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
                logger=False,
            )

            label = f"cv_horizon{horizon}_fold{fold+1}_run{run+1}"
            tracker = make_tracker(label)
            tracker.epoch_start()
            trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
            tracker.epoch_end()
            record_carbon(tracker, label)

            predictions = tft.predict(val_loader, mode="quantiles")
            actuals     = torch.cat([y[0] for x, y in val_loader])
            mae, rmse, smape = compute_metrics(predictions, actuals)

            res = {
                "horizon": horizon, "fold": fold + 1, "run": run + 1,
                "mae": mae, "rmse": rmse, "smape": smape,
                "val_loss": trainer.callback_metrics.get("val_loss").item(),
            }
            append_csv(res, CONFIG["results_path"])

            
            # FEATURE IMPORTANCE  ← important note: run on CPU to avoid CUDA OOM
            
            if run == 0:
                # Clear GPU memory before interpretation
                torch.cuda.empty_cache()
                tft_cpu = tft.cpu()
                cpu_loader = TimeSeriesDataSet.from_dataset(
                    training, val_df
                ).to_dataloader(train=False, batch_size=32, num_workers=8)
                with torch.no_grad():
                    raw_predictions = tft_cpu.predict(
                        cpu_loader, mode="raw", return_x=True
                    )
                    interpretation = tft_cpu.interpret_output(
                        raw_predictions.output
                    )

                # --- record importance values to CSV ---
                importance_label = f"cv_{horizon}h_fold{fold+1}"
                for imp_key, imp_tensor in interpretation.items():
                    if not isinstance(imp_tensor, torch.Tensor):
                        continue
                    vals = imp_tensor.detach().cpu().numpy()
                    if vals.ndim > 1:
                        vals = vals.mean(axis=0)
                    for i, v in enumerate(vals):
                        importance_records.append({
                            "label":            importance_label,
                            "importance_type":  imp_key,
                            "feature_index":    i,
                            "importance":       float(v),
                        })

                # --- save plots manually (avoids pytorch-forecasting /
                #     matplotlib version incompatibility in plot_interpretation)
                for imp_key, imp_tensor in interpretation.items():
                    if not isinstance(imp_tensor, torch.Tensor):
                        continue
                    vals = imp_tensor.detach().cpu().numpy()
                    if vals.ndim > 1:
                        vals = vals.mean(axis=0)
                    vals = vals.astype(float)
                    plot_path = (
                        f"{RUN_DIR}/optimized_importance_{horizon}h"
                        f"_fold{fold+1}_{imp_key}.png"
                    )
                    try:
                        fig, ax = plt.subplots(figsize=(8, max(3, len(vals) * 0.4)))
                        indices = list(range(len(vals)))
                        ax.barh(indices, vals.tolist())
                        ax.set_yticks(indices)
                        ax.set_yticklabels([f"feature_{i}" for i in indices])
                        ax.set_xlabel("Importance")
                        ax.set_title(f"{imp_key} — {horizon}h fold {fold+1}")
                        fig.tight_layout()
                        fig.savefig(plot_path, bbox_inches="tight", dpi=150)
                        plt.close(fig)
                        print(f"[SAVED] Importance plot → {plot_path}", flush=True)
                    except Exception as e:
                        print(f"[WARN] Could not save {plot_path}: {e}", flush=True)
                        plt.close("all")

            del tft, trainer
            torch.cuda.empty_cache()

# Save importance CSV after cross-validation loop
save_importance_csv()

# 7. FINAL TEST EVALUATION

print(f"\n{'='*40}\n>>> FINAL TEST EVALUATION\n{'='*40}", flush=True)

for horizon in horizons:
    encoder_len = horizon * CONFIG["encoder_multiplier"]
    params = best_params_dict[horizon]

    scaler = StandardScaler()
    full_trainval = df_trainval.copy()
    full_trainval[SCALE_FEATURES] = scaler.fit_transform(df_trainval[SCALE_FEATURES])
    test_scaled = df_test.copy()
    test_scaled[SCALE_FEATURES] = scaler.transform(df_test[SCALE_FEATURES])

    training = TimeSeriesDataSet(
        full_trainval, time_idx="time_idx", target=TFT_FEATURES["target"],
        group_ids=["region"],
        max_encoder_length=encoder_len, max_prediction_length=horizon,
        static_categoricals=TFT_FEATURES["static_categoricals"],
        time_varying_known_reals=TFT_FEATURES["time_varying_known_reals"],
        time_varying_unknown_reals=TFT_FEATURES["time_varying_unknown_reals"],
        target_normalizer=GroupNormalizer(groups=["region"]),
    )
    train_loader = training.to_dataloader(train=True, batch_size=32, num_workers=8)
    test_loader  = TimeSeriesDataSet.from_dataset(training, test_scaled).to_dataloader(
        train=False, batch_size=32
    )

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
            logger=False,
        )

        label = f"test_horizon{horizon}_run{run+1}"
        tracker = make_tracker(label)
        tracker.epoch_start()
        trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=test_loader)
        tracker.epoch_end()
        record_carbon(tracker, label)

        predictions = tft.predict(test_loader, mode="quantiles")
        actuals     = torch.cat([y[0] for x, y in test_loader])
        mae, rmse, smape = compute_metrics(predictions, actuals)

        res = {
            "horizon": horizon, "run": run + 1,
            "test_mae": mae, "test_rmse": rmse, "test_smape": smape,
            "test_loss": trainer.callback_metrics.get("val_loss").item(),
        }
        append_csv(res, CONFIG["test_results_path"])

        del tft, trainer
        torch.cuda.empty_cache()


# 8. FINAL SAVES

save_carbon_summary()

# Re-save hyperparams (redundant but this guarantees they're on disk even after crash)
with open(CONFIG["best_params_path"], "w") as fh:
    json.dump(serialisable, fh, indent=2)

print("\n===== ALL TASKS COMPLETE =====", flush=True)
print(f"  Hyperparameters  → {CONFIG['best_params_path']}", flush=True)
print(f"  Carbon summary   → {CONFIG['carbon_summary_path']}", flush=True)
print(f"  Feature importance CSV → {CONFIG['feature_importance_path']}", flush=True)
print(f"  CV results       → {CONFIG['results_path']}", flush=True)
print(f"  Test results     → {CONFIG['test_results_path']}", flush=True)
print(f"  Full run log     → {LOG_PATH}", flush=True)
