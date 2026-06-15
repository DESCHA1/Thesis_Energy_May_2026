import os
import sys
import json
import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from carbontracker.tracker import CarbonTracker

# ==============================================================================
# LOGGING SETUP
# ==============================================================================
RUN_DIR = "DATA/qr_patchtst_run_01"
os.makedirs(RUN_DIR, exist_ok=True)
LOG_PATH = f"{RUN_DIR}/run_log.txt"

class Tee:
    def __init__(self, stream, filepath):
        self.stream = stream
        self.file = open(filepath, "a", buffering=1)

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

# ==============================================================================
# 1. CONFIG
# ==============================================================================
CONFIG = {
    "dataset_path":           "DATA/final_data_with_features.csv",
    "results_path":           f"{RUN_DIR}/cv_results.csv",
    "test_results_path":      f"{RUN_DIR}/test_results.csv",
    "best_params_path":       f"{RUN_DIR}/best_hyperparams.json",
    "carbon_summary_path":    f"{RUN_DIR}/carbon_summary.csv",
    "feature_importance_path":f"{RUN_DIR}/feature_importance.csv",
    "tuning_trials":          20,
    "n_splits":               3,
    "robustness_runs":        3,
    "quantiles":              [0.1, 0.5, 0.9],
    # PatchTST fixed architectural choices (from paper Table 1)
    "patch_size":             15,      # P
    "stride":                 8,       # S  (non-overlapping part)
    "n_encoder_layers":       2,
    "n_heads":                8,
    "ff_dim_multiplier":      4,       # d_ff = hidden_size * 4
    "dropout":                0.1,
    "encoder_multiplier":     2,       # encoder_len = horizon * multiplier
}

QUANTILES   = CONFIG["quantiles"]
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Features — kept identical to TFT experiment
TARGET      = "residual_load"
KNOWN_REALS = ["is_holiday", "is_weekend", "hour_sin", "hour_cos",
               "day_sin",    "day_cos"]
UNK_REALS   = ["wind_u", "wind_v", "ghi"]          # scaled
ALL_FEATURES = KNOWN_REALS + UNK_REALS + [TARGET]  # channel-independent inputs
N_CHANNELS   = len(ALL_FEATURES)
SCALE_FEATURES = ["wind_u", "wind_v", "ghi"]

# ==============================================================================
# 2. CARBON TRACKING
# ==============================================================================
carbon_records = []

def make_tracker(label: str) -> CarbonTracker:
    ct_dir = f"{RUN_DIR}/carbontracker"
    os.makedirs(ct_dir, exist_ok=True)
    return CarbonTracker(epochs=1, components="gpu",
                         log_dir=ct_dir, monitor_epochs=1, verbose=2)

def record_carbon(tracker: CarbonTracker, label: str):
    try:
        ct_dir = f"{RUN_DIR}/carbontracker"
        logs = sorted(
            [f for f in os.listdir(ct_dir) if f.endswith(".json")],
            key=lambda f: os.path.getmtime(os.path.join(ct_dir, f))
        )
        if logs:
            with open(os.path.join(ct_dir, logs[-1])) as fh:
                data = json.load(fh)
            epoch_data = data[-1] if isinstance(data, list) else data
            carbon_records.append({
                "label":      label,
                "duration_s": epoch_data.get("duration"),
                "energy_kWh": epoch_data.get("actual", {}).get("energy (kWh)"),
                "co2_g":      epoch_data.get("actual", {}).get("co2eq (g)"),
            })
    except Exception as e:
        carbon_records.append({"label": label, "error": str(e)})

def save_carbon():
    pd.DataFrame(carbon_records).to_csv(CONFIG["carbon_summary_path"], index=False)
    print(f"[SAVED] Carbon summary → {CONFIG['carbon_summary_path']}", flush=True)

# ==============================================================================
# 3. DATASET
# ==============================================================================
class ResidualLoadDataset(Dataset):
    """
    Sliding-window dataset.
    x : (n_channels, encoder_len)   — past observations
    y : (horizon,)                  — future residual load (target only)
    """
    def __init__(self, df: pd.DataFrame, encoder_len: int, horizon: int):
        self.encoder_len = encoder_len
        self.horizon     = horizon
        # (T, C) array — channel order matches ALL_FEATURES
        self.data   = df[ALL_FEATURES].values.astype(np.float32)
        self.target = df[TARGET].values.astype(np.float32)
        self.n      = len(self.data) - encoder_len - horizon + 1

    def __len__(self):
        return max(self.n, 0)

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.encoder_len]          # (enc, C)
        y = self.target[idx + self.encoder_len :
                        idx + self.encoder_len + self.horizon] # (H,)
        return torch.tensor(x.T), torch.tensor(y)            # (C, enc), (H,)

def make_loaders(train_df, val_df, encoder_len, horizon, batch_size=32):
    tr_ds = ResidualLoadDataset(train_df, encoder_len, horizon)
    va_ds = ResidualLoadDataset(val_df,   encoder_len, horizon)
    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                       num_workers=8, pin_memory=True)
    va_ld = DataLoader(va_ds, batch_size=batch_size, shuffle=False,
                       num_workers=8, pin_memory=True)
    return tr_ld, va_ld

# ==============================================================================
# 4. QR-PatchTST MODEL
# ==============================================================================
class PatchEmbedding(nn.Module):
    """
    Splits a univariate time series of length L into Z patches of length P
    with stride S, then projects each patch to d_model dimensions.
    """
    def __init__(self, seq_len: int, patch_size: int, stride: int, d_model: int,
                 dropout: float = 0.1):
        super().__init__()
        self.patch_size = patch_size
        self.stride     = stride
        # number of patches  Z = floor((L - P) / S) + 2
        self.n_patches  = math.floor((seq_len - patch_size) / stride) + 2
        self.proj       = nn.Linear(patch_size, d_model)
        self.pos_embed  = nn.Parameter(torch.randn(1, self.n_patches, d_model))
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, L)
        returns : (B, Z, d_model)
        """
        B, L = x.shape
        # Pad end so we always get n_patches complete patches
        pad = (self.n_patches - 1) * self.stride + self.patch_size - L
        if pad > 0:
            x = torch.nn.functional.pad(x, (0, pad), mode="replicate")
        # Unfold into patches: (B, Z, P)
        patches = x.unfold(-1, self.patch_size, self.stride)
        out = self.proj(patches) + self.pos_embed[:, :patches.size(1), :]
        return self.dropout(out)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                           batch_first=True)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.ReLU(),
            nn.Linear(ff_dim, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention + residual
        a, _ = self.attn(x, x, x)
        x    = self.norm1(x + self.drop(a))
        # Feed-forward + residual
        x    = self.norm2(x + self.drop(self.ff(x)))
        return x


class QRPatchTST(nn.Module):
    """
    Channel-independent PatchTST with quantile regression head.

    For each of the C input channels:
      1. Patch + embed  → (B, Z, d_model)
      2. Transformer encoder
      3. Flatten → linear head → (B, H * Q)

    All C channels share the same patch embedding & encoder weights
    (channel-independence), then their outputs are averaged before
    the prediction head, which outputs H * Q values.
    """
    def __init__(self, n_channels: int, encoder_len: int, horizon: int,
                 d_model: int, n_heads: int, n_layers: int, ff_dim: int,
                 patch_size: int, stride: int, dropout: float,
                 quantiles: list):
        super().__init__()
        self.n_channels = n_channels
        self.horizon    = horizon
        self.quantiles  = quantiles
        self.Q          = len(quantiles)

        self.patch_embed = PatchEmbedding(encoder_len, patch_size, stride,
                                          d_model, dropout)
        n_patches = self.patch_embed.n_patches

        self.encoder = nn.Sequential(*[
            TransformerEncoderLayer(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])

        # Flatten patches → predict H * Q values
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_patches * d_model, horizon * self.Q),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x      : (B, C, L)
        returns: (B, H, Q)
        """
        B, C, L = x.shape
        # Process all channels in one batched forward pass
        # Reshape to (B*C, L) for channel independence
        xc  = x.reshape(B * C, L)
        emb = self.patch_embed(xc)           # (B*C, Z, d_model)
        enc = self.encoder(emb)              # (B*C, Z, d_model)
        # Reshape back → (B, C, Z, d_model) then mean over channels
        enc = enc.reshape(B, C, enc.size(1), enc.size(2)).mean(dim=1)  # (B, Z, d)
        out = self.head(enc)                 # (B, H*Q)
        out = out.reshape(B, self.horizon, self.Q)                      # (B, H, Q)
        return out

# ==============================================================================
# 5. PINBALL LOSS
# ==============================================================================
def pinball_loss(preds: torch.Tensor, targets: torch.Tensor,
                 quantiles: list) -> torch.Tensor:
    """
    preds   : (B, H, Q)
    targets : (B, H)
    """
    q = torch.tensor(quantiles, dtype=torch.float32, device=preds.device)
    t = targets.unsqueeze(-1).expand_as(preds)  # (B, H, Q)
    err   = t - preds
    loss  = torch.max(q * err, (q - 1) * err)  # pinball per element
    return loss.mean()

# ==============================================================================
# 6. METRICS
# ==============================================================================
def compute_metrics(preds: torch.Tensor, actuals: torch.Tensor):
    """preds (N, H, Q), actuals (N, H) — uses median (Q index 1)."""
    median = preds[:, :, 1]
    mae    = torch.mean(torch.abs(median - actuals)).item()
    rmse   = torch.sqrt(torch.mean((median - actuals) ** 2)).item()
    smape  = torch.mean(2 * torch.abs(median - actuals) /
                        (torch.abs(median) + torch.abs(actuals) + 1e-8)).item()
    return mae, rmse, smape

# ==============================================================================
# 7. TRAINING UTILITIES
# ==============================================================================
def train_one_epoch(model, loader, optimizer):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        pred = model(xb)
        loss = pinball_loss(pred, yb, QUANTILES)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * len(xb)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_acts = [], []
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pred = model(xb)
        total_loss += pinball_loss(pred, yb, QUANTILES).item() * len(xb)
        all_preds.append(pred.cpu())
        all_acts.append(yb.cpu())
    preds   = torch.cat(all_preds)
    actuals = torch.cat(all_acts)
    val_loss = total_loss / len(loader.dataset)
    return val_loss, preds, actuals


def train_model(model, tr_loader, va_loader,
                lr: float, max_epochs: int = 30, patience: int = 8,
                label: str = ""):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3,
                                                      factor=0.5)
    best_val, best_state, wait = float("inf"), None, 0

    tracker = make_tracker(label)
    tracker.epoch_start()

    for epoch in range(max_epochs):
        tr_loss = train_one_epoch(model, tr_loader, optimizer)
        va_loss, _, _ = evaluate(model, va_loader)
        scheduler.step(va_loss)

        print(f"  Epoch {epoch+1:02d} | train={tr_loss:.2f} | val={va_loss:.2f}",
              flush=True)

        if va_loss < best_val:
            best_val   = va_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch+1}", flush=True)
                break

    tracker.epoch_end()
    record_carbon(tracker, label)

    model.load_state_dict(best_state)
    return best_val


def build_model(params: dict, encoder_len: int, horizon: int) -> QRPatchTST:
    d_model = params["d_model"]
    return QRPatchTST(
        n_channels  = N_CHANNELS,
        encoder_len = encoder_len,
        horizon     = horizon,
        d_model     = d_model,
        n_heads     = CONFIG["n_heads"],
        n_layers    = CONFIG["n_encoder_layers"],
        ff_dim      = d_model * CONFIG["ff_dim_multiplier"],
        patch_size  = CONFIG["patch_size"],
        stride      = CONFIG["stride"],
        dropout     = params.get("dropout", CONFIG["dropout"]),
        quantiles   = QUANTILES,
    ).to(DEVICE)

# ==============================================================================
# 8. DATA PREP (identical splits to TFT script)
# ==============================================================================
df = pd.read_csv(CONFIG["dataset_path"])
df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"])
df = df.sort_values("utc_timestamp").reset_index(drop=True)
df["region"] = "DE"

test_cutoff  = int(len(df) * 0.90)
df_trainval  = df.iloc[:test_cutoff].reset_index(drop=True)
df_test      = df.iloc[test_cutoff:].reset_index(drop=True)
print(f"Train/val size: {len(df_trainval)} | Test size: {len(df_test)}", flush=True)

def scale_fold(train_df, val_df):
    scaler = StandardScaler()
    tr = train_df.copy()
    va = val_df.copy()
    tr[SCALE_FEATURES] = scaler.fit_transform(train_df[SCALE_FEATURES])
    va[SCALE_FEATURES] = scaler.transform(val_df[SCALE_FEATURES])
    return tr, va, scaler

def append_csv(row: dict, path: str):
    pd.DataFrame([row]).to_csv(path, mode="a", index=False,
                               header=not os.path.exists(path))

# ==============================================================================
# 9. OPTUNA OBJECTIVE
# ==============================================================================
def objective(trial, horizon: int) -> float:
    d_model = trial.suggest_categorical("d_model", [32, 64, 128, 256])
    lr      = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.05, 0.3)

    encoder_len = horizon * CONFIG["encoder_multiplier"]
    tscv        = TimeSeriesSplit(n_splits=CONFIG["n_splits"])
    train_idx, val_idx = next(tscv.split(df_trainval))

    tr_df, va_df, _ = scale_fold(df_trainval.iloc[train_idx],
                                  df_trainval.iloc[val_idx])
    tr_ld, va_ld = make_loaders(tr_df, va_df, encoder_len, horizon,
                                 batch_size=64)

    params = {"d_model": d_model, "dropout": dropout}
    model  = build_model(params, encoder_len, horizon)
    val_loss = train_model(
        model, tr_ld, va_ld, lr=lr, max_epochs=10, patience=5,
        label=f"optuna_h{horizon}_trial{trial.number}"
    )
    del model
    torch.cuda.empty_cache()
    return val_loss

# ==============================================================================
# 10. OPTUNA EXECUTION
# ==============================================================================
print("\n--- Starting Optuna Hyperparameter Optimisation ---", flush=True)
best_params_dict = {}

for tune_horizon in [24, 168]:
    print(f"\n>>> Tuning for horizon: {tune_horizon}h", flush=True)
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial, h=tune_horizon: objective(trial, h),
                   n_trials=CONFIG["tuning_trials"])
    best_params_dict[tune_horizon] = study.best_params
    print(f"   Best Params for {tune_horizon}h: {study.best_params}", flush=True)

print("\n>>> Applying 168h parameters to 720h horizon", flush=True)
best_params_dict[720] = best_params_dict[168].copy()

# Save hyperparameters
serialisable = {str(k): v for k, v in best_params_dict.items()}
with open(CONFIG["best_params_path"], "w") as fh:
    json.dump(serialisable, fh, indent=2)
print(f"[SAVED] Best hyperparameters → {CONFIG['best_params_path']}", flush=True)

print("\n===== BEST HYPERPARAMETERS SUMMARY =====", flush=True)
for h, p in best_params_dict.items():
    print(f"  {h}h : {p}", flush=True)
print("=========================================\n", flush=True)

# ==============================================================================
# 11. FEATURE IMPORTANCE via attention weights
#     We compute average attention weights across heads & encoder layers
#     as a proxy for input patch importance, then aggregate to features.
# ==============================================================================
importance_records = []

@torch.no_grad()
def extract_attention_importance(model: QRPatchTST, loader: DataLoader,
                                 label: str, horizon: int, fold: int):
    """
    Register hooks on each TransformerEncoderLayer's MultiheadAttention,
    collect attention weights, average over batch / heads / query positions,
    and save a bar chart + CSV rows.
    """
    model.eval()
    attn_maps = []   # list of (B, n_heads, Z, Z) tensors per layer

    hooks = []
    for layer in model.encoder:
        def hook_fn(module, inp, out, storage=attn_maps):
            # MultiheadAttention returns (output, attn_weights)
            # but nn.MultiheadAttention with need_weights=True by default
            # We need to re-run with need_weights=True — easier to subclass,
            # so we just call the attention module directly here.
            pass
        # Use forward hook on the attn sub-module instead
        def make_hook(store):
            def h(module, inp, out):
                # out[1] is the attention weight tensor (B, Z, Z)
                if isinstance(out, tuple) and len(out) == 2:
                    store.append(out[1].detach().cpu())
            return h
        hooks.append(layer.attn.register_forward_hook(make_hook(attn_maps)))

    # Run one batch through on CPU to avoid OOM
    model_cpu = model.cpu()
    for xb, _ in loader:
        xb = xb  # already CPU
        _ = model_cpu(xb)
        break    # one batch is enough for importance estimates

    for h in hooks:
        h.remove()

    model.to(DEVICE)

    if not attn_maps:
        print(f"[WARN] No attention maps captured for {label}", flush=True)
        return

    # Average over layers, batch dim, and query positions → (Z,) patch importance
    avg_attn = torch.stack(attn_maps).mean(dim=0)   # (B, Z, Z)
    patch_imp = avg_attn.mean(dim=[0, 1])            # (Z,) = avg over batch & queries

    vals = patch_imp.numpy().astype(float)
    for i, v in enumerate(vals):
        importance_records.append({
            "label":           label,
            "importance_type": "patch_attention",
            "feature_index":   i,
            "importance":      float(v),
        })

    # Plot
    plot_path = f"{RUN_DIR}/importance_{horizon}h_fold{fold}_patch_attention.png"
    try:
        fig, ax = plt.subplots(figsize=(10, max(3, len(vals) * 0.25)))
        ax.bar(range(len(vals)), vals)
        ax.set_xlabel("Patch index")
        ax.set_ylabel("Mean attention weight")
        ax.set_title(f"Patch attention importance — {horizon}h fold {fold}")
        fig.tight_layout()
        fig.savefig(plot_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"[SAVED] Importance plot → {plot_path}", flush=True)
    except Exception as e:
        print(f"[WARN] Could not save importance plot: {e}", flush=True)
        plt.close("all")


def save_importance_csv():
    if importance_records:
        pd.DataFrame(importance_records).to_csv(
            CONFIG["feature_importance_path"], index=False)
        print(f"[SAVED] Feature importance → {CONFIG['feature_importance_path']}",
              flush=True)

# ==============================================================================
# 12. CROSS-VALIDATION
# ==============================================================================
horizons = [24, 168, 720]
tscv     = TimeSeriesSplit(n_splits=CONFIG["n_splits"])

for horizon in horizons:
    print(f"\n{'='*40}\n>>> CROSS-VALIDATION: {horizon}h\n{'='*40}", flush=True)
    encoder_len = horizon * CONFIG["encoder_multiplier"]
    params      = best_params_dict[horizon]

    for fold, (train_idx, val_idx) in enumerate(tscv.split(df_trainval)):
        tr_df, va_df, _ = scale_fold(df_trainval.iloc[train_idx],
                                      df_trainval.iloc[val_idx])
        tr_ld, va_ld = make_loaders(tr_df, va_df, encoder_len, horizon)

        for run in range(CONFIG["robustness_runs"]):
            torch.manual_seed(42 + run)
            np.random.seed(42 + run)

            model    = build_model(params, encoder_len, horizon)
            label    = f"cv_h{horizon}_fold{fold+1}_run{run+1}"
            val_loss = train_model(
                model, tr_ld, va_ld,
                lr        = params["learning_rate"],
                max_epochs= 30,
                patience  = 8,
                label     = label,
            )

            _, preds, actuals = evaluate(model, va_ld)
            mae, rmse, smape  = compute_metrics(preds, actuals)

            append_csv({
                "horizon": horizon, "fold": fold+1, "run": run+1,
                "mae": mae, "rmse": rmse, "smape": smape,
                "val_loss": val_loss,
            }, CONFIG["results_path"])

            # Feature importance for first run of each fold
            if run == 0:
                cpu_ld = DataLoader(
                    ResidualLoadDataset(va_df, encoder_len, horizon),
                    batch_size=32, shuffle=False
                )
                extract_attention_importance(
                    model, cpu_ld,
                    label=f"cv_{horizon}h_fold{fold+1}",
                    horizon=horizon, fold=fold+1
                )

            del model
            torch.cuda.empty_cache()

save_importance_csv()

# ==============================================================================
# 13. FINAL TEST EVALUATION
# ==============================================================================
print(f"\n{'='*40}\n>>> FINAL TEST EVALUATION\n{'='*40}", flush=True)

for horizon in horizons:
    encoder_len = horizon * CONFIG["encoder_multiplier"]
    params      = best_params_dict[horizon]

    scaler = StandardScaler()
    full_tr = df_trainval.copy()
    full_tr[SCALE_FEATURES] = scaler.fit_transform(df_trainval[SCALE_FEATURES])
    test_sc = df_test.copy()
    test_sc[SCALE_FEATURES] = scaler.transform(df_test[SCALE_FEATURES])

    tr_ld, te_ld = make_loaders(full_tr, test_sc, encoder_len, horizon)

    for run in range(CONFIG["robustness_runs"]):
        torch.manual_seed(42 + run)
        np.random.seed(42 + run)

        model    = build_model(params, encoder_len, horizon)
        label    = f"test_h{horizon}_run{run+1}"
        val_loss = train_model(
            model, tr_ld, te_ld,
            lr        = params["learning_rate"],
            max_epochs= 30,
            patience  = 8,
            label     = label,
        )

        _, preds, actuals = evaluate(model, te_ld)
        mae, rmse, smape  = compute_metrics(preds, actuals)

        append_csv({
            "horizon":   horizon, "run": run+1,
            "test_mae":  mae, "test_rmse": rmse, "test_smape": smape,
            "test_loss": val_loss,
        }, CONFIG["test_results_path"])

        del model
        torch.cuda.empty_cache()

# ==============================================================================
# 14. FINAL SAVES & SUMMARY
# ==============================================================================
save_carbon()

# Re-save hyperparams as safety net
with open(CONFIG["best_params_path"], "w") as fh:
    json.dump(serialisable, fh, indent=2)

print("\n===== ALL TASKS COMPLETE =====", flush=True)
print(f"  Hyperparameters     → {CONFIG['best_params_path']}", flush=True)
print(f"  Carbon summary      → {CONFIG['carbon_summary_path']}", flush=True)
print(f"  Feature importance  → {CONFIG['feature_importance_path']}", flush=True)
print(f"  CV results          → {CONFIG['results_path']}", flush=True)
print(f"  Test results        → {CONFIG['test_results_path']}", flush=True)
print(f"  Full run log        → {LOG_PATH}", flush=True)
