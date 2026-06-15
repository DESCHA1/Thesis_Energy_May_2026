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
RUN_DIR = "DATA/qr_patchtst_improved_run_01"
os.makedirs(RUN_DIR, exist_ok=True)
LOG_PATH = f"{RUN_DIR}/run_log.txt"

class Tee:
    def __init__(self, stream, filepath):
        self.stream = stream
        self.file   = open(filepath, "a", buffering=1)

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
    "dataset_path":            "DATA/final_data_with_features.csv",
    "results_path":            f"{RUN_DIR}/cv_results.csv",
    "test_results_path":       f"{RUN_DIR}/test_results.csv",
    "best_params_path":        f"{RUN_DIR}/best_hyperparams.json",
    "carbon_summary_path":     f"{RUN_DIR}/carbon_summary.csv",
    "feature_importance_path": f"{RUN_DIR}/feature_importance.csv",
    "tuning_trials":           20,
    "n_splits":                3,
    "robustness_runs":         3,
    "quantiles":               [0.1, 0.5, 0.9],
    # IMPROVEMENT 1: encoder_multiplier increased from 2 → 4
    # gives 4x lookback window (e.g. 720h horizon → 2880h context)
    "encoder_multiplier":      4,
    # IMPROVEMENT 3: patch_size and stride now tuned by Optuna (see objective)
    # Fixed architectural choices
    "n_heads":                 8,
    # IMPROVEMENT 4: n_encoder_layers now tuned by Optuna (2–4)
    "ff_dim_multiplier":       4,    # d_ff = d_model * 4
}

QUANTILES = CONFIG["quantiles"]
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}", flush=True)

# IMPROVEMENT 2: separate known-future from unknown features
# Known reals: available for the full forecast horizon (calendar, holidays)
# Unknown reals: only observed in the past (wind, solar, target)
TARGET        = "residual_load"
KNOWN_REALS   = ["is_holiday", "is_weekend",
                  "hour_sin",   "hour_cos",
                  "day_sin",    "day_cos"]          # 6 features
UNKNOWN_REALS = ["wind_u", "wind_v", "ghi"]        # 3 features — scaled
ALL_PAST_FEATURES = KNOWN_REALS + UNKNOWN_REALS + [TARGET]  # encoder input (10 ch)
N_PAST_CH         = len(ALL_PAST_FEATURES)          # 10
N_KNOWN_CH        = len(KNOWN_REALS)                # 6  — also fed to decoder
SCALE_FEATURES    = ["wind_u", "wind_v", "ghi"]

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
        ct_dir   = f"{RUN_DIR}/carbontracker"
        # Try JSON first
        json_logs = sorted(
            [f for f in os.listdir(ct_dir) if f.endswith(".json")],
            key=lambda f: os.path.getmtime(os.path.join(ct_dir, f))
        )
        if json_logs:
            with open(os.path.join(ct_dir, json_logs[-1])) as fh:
                data = json.load(fh)
            epoch_data = data[-1] if isinstance(data, list) else data
            energy = epoch_data.get("actual", {}).get("energy (kWh)")
            co2    = epoch_data.get("actual", {}).get("co2eq (g)")
            dur    = epoch_data.get("duration")
            if energy is not None:
                carbon_records.append({
                    "label": label, "duration_s": dur,
                    "energy_kWh": energy, "co2_g": co2,
                })
                return
        # Fallback: parse plain text log
        txt_logs = sorted(
            [f for f in os.listdir(ct_dir) if f.endswith(".log")],
            key=lambda f: os.path.getmtime(os.path.join(ct_dir, f))
        )
        if txt_logs:
            import re
            with open(os.path.join(ct_dir, txt_logs[-1])) as fh:
                text = fh.read()
            energy = re.search(r"Energy:\s+([\d.]+)\s+kWh", text)
            co2    = re.search(r"CO2eq:\s+([\d.]+)\s+g",    text)
            time_  = re.search(r"Time:\s+([\d:]+)",          text)
            carbon_records.append({
                "label":      label,
                "duration_s": time_.group(1)        if time_  else None,
                "energy_kWh": float(energy.group(1)) if energy else None,
                "co2_g":      float(co2.group(1))    if co2    else None,
            })
            return
        carbon_records.append({
            "label": label, "duration_s": None,
            "energy_kWh": None, "co2_g": None, "note": "no log file found"
        })
    except Exception as e:
        carbon_records.append({"label": label, "error": str(e)})

def save_carbon():
    pd.DataFrame(carbon_records).to_csv(CONFIG["carbon_summary_path"], index=False)
    print(f"[SAVED] Carbon summary → {CONFIG['carbon_summary_path']}", flush=True)

# ==============================================================================
# 3. DATASET
# IMPROVEMENT 2: dataset now returns both past (encoder) and future known
#                (decoder) tensors separately
# ==============================================================================
class ResidualLoadDataset(Dataset):
    """
    Returns:
      x_past  : (N_PAST_CH, encoder_len)   all past features
      x_future: (N_KNOWN_CH, horizon)       known-future features only
      y       : (horizon,)                  future residual load
    """
    def __init__(self, df: pd.DataFrame, encoder_len: int, horizon: int):
        self.encoder_len = encoder_len
        self.horizon     = horizon
        self.past        = df[ALL_PAST_FEATURES].values.astype(np.float32)
        self.known       = df[KNOWN_REALS].values.astype(np.float32)
        self.target      = df[TARGET].values.astype(np.float32)
        self.n           = len(self.past) - encoder_len - horizon + 1

    def __len__(self):
        return max(self.n, 0)

    def __getitem__(self, idx):
        x_past   = self.past[idx : idx + self.encoder_len]                # (enc, C_past)
        x_future = self.known[idx + self.encoder_len :
                               idx + self.encoder_len + self.horizon]      # (H, C_known)
        y        = self.target[idx + self.encoder_len :
                                idx + self.encoder_len + self.horizon]     # (H,)
        return (torch.tensor(x_past.T),      # (C_past,  enc)
                torch.tensor(x_future.T),    # (C_known, H)
                torch.tensor(y))             # (H,)


def make_loaders(train_df, val_df, encoder_len, horizon, batch_size=32):
    tr_ds = ResidualLoadDataset(train_df, encoder_len, horizon)
    va_ds = ResidualLoadDataset(val_df,   encoder_len, horizon)
    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                       num_workers=8, pin_memory=True)
    va_ld = DataLoader(va_ds, batch_size=batch_size, shuffle=False,
                       num_workers=8, pin_memory=True)
    return tr_ld, va_ld

# ==============================================================================
# 4. MODEL COMPONENTS
# ==============================================================================
class PatchEmbedding(nn.Module):
    """
    Univariate patch embedding with learnable positional encoding.
    IMPROVEMENT 3: patch_size and stride are now hyperparameters.
    """
    def __init__(self, seq_len: int, patch_size: int, stride: int,
                 d_model: int, dropout: float = 0.1):
        super().__init__()
        self.patch_size = patch_size
        self.stride     = stride
        self.n_patches  = math.floor((seq_len - patch_size) / stride) + 2
        self.proj       = nn.Linear(patch_size, d_model)
        self.pos_embed  = nn.Parameter(torch.randn(1, self.n_patches, d_model))
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, L)  →  (B, Z, d_model)"""
        B, L = x.shape
        pad  = (self.n_patches - 1) * self.stride + self.patch_size - L
        if pad > 0:
            x = torch.nn.functional.pad(x, (0, pad), mode="replicate")
        patches = x.unfold(-1, self.patch_size, self.stride)
        out     = self.proj(patches) + self.pos_embed[:, :patches.size(1), :]
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
        a, _ = self.attn(x, x, x)
        x    = self.norm1(x + self.drop(a))
        x    = self.norm2(x + self.drop(self.ff(x)))
        return x

# ==============================================================================
# 5. IMPROVED QR-PatchTST MODEL
# Incorporates all 5 improvements:
#   1. Longer encoder (handled at dataset level via encoder_multiplier=4)
#   2. Separate known-future branch fused before the prediction head
#   3. Tunable patch_size and stride
#   4. Tunable n_encoder_layers (2-4)
#   5. MLP prediction head instead of single linear layer
# ==============================================================================
class ImprovedQRPatchTST(nn.Module):
    """
    Channel-independent PatchTST encoder for past features +
    a lightweight linear projection of known future features,
    fused and decoded by an MLP quantile head.
    """
    def __init__(self, n_past_ch: int, n_known_ch: int,
                 encoder_len: int, horizon: int,
                 d_model: int, n_heads: int, n_layers: int, ff_dim: int,
                 patch_size: int, stride: int, dropout: float,
                 quantiles: list):
        super().__init__()
        self.n_past_ch  = n_past_ch
        self.n_known_ch = n_known_ch
        self.horizon    = horizon
        self.quantiles  = quantiles
        self.Q          = len(quantiles)

        # --- Past encoder (channel-independent PatchTST) ---
        self.patch_embed = PatchEmbedding(encoder_len, patch_size, stride,
                                          d_model, dropout)
        n_patches        = self.patch_embed.n_patches
        self.encoder     = nn.Sequential(*[
            TransformerEncoderLayer(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])
        past_dim = n_patches * d_model   # flattened encoder output per channel

        # IMPROVEMENT 2: Known-future branch
        # Projects (C_known * horizon) → d_model
        self.future_proj = nn.Sequential(
            nn.Linear(n_known_ch * horizon, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # IMPROVEMENT 5: MLP head (replaces single linear layer)
        # Input: averaged past encoding (d_model) + future projection (d_model)
        fused_dim = d_model + d_model
        self.head = nn.Sequential(
            nn.Linear(fused_dim, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon * self.Q),
        )

        # Separate flatten for past encoder output
        self.past_flatten = nn.Sequential(
            nn.Flatten(),
            nn.Linear(past_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x_past: torch.Tensor,
                x_future: torch.Tensor) -> torch.Tensor:
        """
        x_past   : (B, C_past,  encoder_len)
        x_future : (B, C_known, horizon)
        returns  : (B, H, Q)
        """
        B, C, L = x_past.shape

        # --- Encode past (channel-independent) ---
        xc  = x_past.reshape(B * C, L)                    # (B*C, L)
        emb = self.patch_embed(xc)                         # (B*C, Z, d)
        enc = self.encoder(emb)                            # (B*C, Z, d)
        enc = enc.reshape(B, C, enc.size(1), enc.size(2))  # (B, C, Z, d)
        # Average over channels, flatten patches, project to d_model
        enc = enc.mean(dim=1)                              # (B, Z, d)
        past_repr = self.past_flatten(enc)                 # (B, d_model)

        # --- Encode known future (IMPROVEMENT 2) ---
        fut_flat  = x_future.reshape(B, -1)               # (B, C_known * H)
        fut_repr  = self.future_proj(fut_flat)             # (B, d_model)

        # --- Fuse and predict ---
        fused = torch.cat([past_repr, fut_repr], dim=-1)  # (B, 2*d_model)
        out   = self.head(fused)                           # (B, H*Q)
        out   = out.reshape(B, self.horizon, self.Q)       # (B, H, Q)
        return out

# ==============================================================================
# 6. PINBALL LOSS
# ==============================================================================
def pinball_loss(preds: torch.Tensor, targets: torch.Tensor,
                 quantiles: list) -> torch.Tensor:
    """preds (B,H,Q), targets (B,H)"""
    q    = torch.tensor(quantiles, dtype=torch.float32, device=preds.device)
    t    = targets.unsqueeze(-1).expand_as(preds)
    err  = t - preds
    loss = torch.max(q * err, (q - 1) * err)
    return loss.mean()

# ==============================================================================
# 7. METRICS
# ==============================================================================
def compute_metrics(preds: torch.Tensor, actuals: torch.Tensor):
    """preds (N,H,Q), actuals (N,H) — median = index 1"""
    median = preds[:, :, 1]
    mae    = torch.mean(torch.abs(median - actuals)).item()
    rmse   = torch.sqrt(torch.mean((median - actuals) ** 2)).item()
    smape  = torch.mean(2 * torch.abs(median - actuals) /
                        (torch.abs(median) + torch.abs(actuals) + 1e-8)).item()
    return mae, rmse, smape

# ==============================================================================
# 8. TRAINING UTILITIES
# ==============================================================================
def train_one_epoch(model, loader, optimizer):
    model.train()
    total = 0.0
    for x_past, x_fut, yb in loader:
        x_past, x_fut, yb = (x_past.to(DEVICE), x_fut.to(DEVICE),
                              yb.to(DEVICE))
        optimizer.zero_grad()
        pred = model(x_past, x_fut)
        loss = pinball_loss(pred, yb, QUANTILES)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * len(x_past)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_acts = [], []
    total_loss = 0.0
    for x_past, x_fut, yb in loader:
        x_past, x_fut, yb = (x_past.to(DEVICE), x_fut.to(DEVICE),
                              yb.to(DEVICE))
        pred        = model(x_past, x_fut)
        total_loss += pinball_loss(pred, yb, QUANTILES).item() * len(x_past)
        all_preds.append(pred.cpu())
        all_acts.append(yb.cpu())
    preds    = torch.cat(all_preds)
    actuals  = torch.cat(all_acts)
    val_loss = total_loss / len(loader.dataset)
    return val_loss, preds, actuals


def train_model(model, tr_loader, va_loader,
                lr: float, max_epochs: int = 30, patience: int = 8,
                label: str = "") -> float:
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                      patience=3, factor=0.5)
    best_val, best_state, wait = float("inf"), None, 0

    tracker = make_tracker(label)
    tracker.epoch_start()

    for epoch in range(max_epochs):
        tr_loss = train_one_epoch(model, tr_loader, optimizer)
        va_loss, _, _ = evaluate(model, va_loader)
        scheduler.step(va_loss)
        print(f"  Epoch {epoch+1:02d} | train={tr_loss:.4f} | "
              f"val={va_loss:.4f}", flush=True)

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


def build_model(params: dict, encoder_len: int, horizon: int
                ) -> ImprovedQRPatchTST:
    d_model = params["d_model"]
    return ImprovedQRPatchTST(
        n_past_ch   = N_PAST_CH,
        n_known_ch  = N_KNOWN_CH,
        encoder_len = encoder_len,
        horizon     = horizon,
        d_model     = d_model,
        n_heads     = CONFIG["n_heads"],
        n_layers    = params["n_layers"],
        ff_dim      = d_model * CONFIG["ff_dim_multiplier"],
        patch_size  = params["patch_size"],
        stride      = params["stride"],
        dropout     = params.get("dropout", 0.1),
        quantiles   = QUANTILES,
    ).to(DEVICE)

# ==============================================================================
# 9. DATA PREP
# ==============================================================================
df = pd.read_csv(CONFIG["dataset_path"])
df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"])
df = df.sort_values("utc_timestamp").reset_index(drop=True)
df["region"] = "DE"

test_cutoff = int(len(df) * 0.90)
df_trainval = df.iloc[:test_cutoff].reset_index(drop=True)
df_test     = df.iloc[test_cutoff:].reset_index(drop=True)
print(f"Train/val size: {len(df_trainval)} | Test size: {len(df_test)}",
      flush=True)


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
# 10. OPTUNA OBJECTIVE
# IMPROVEMENT 3 & 4: patch_size, stride, and n_layers added to search space
# ==============================================================================
def objective(trial, horizon: int) -> float:
    d_model    = trial.suggest_categorical("d_model",    [64, 128, 256])
    lr         = trial.suggest_float("learning_rate",    1e-4, 1e-2, log=True)
    dropout    = trial.suggest_float("dropout",          0.05, 0.3)
    # IMPROVEMENT 3: tunable patch size and stride
    patch_size = trial.suggest_categorical("patch_size", [15, 24, 48])
    stride     = trial.suggest_categorical("stride",     [8, 12, 24])
    # IMPROVEMENT 4: tunable number of encoder layers
    n_layers   = trial.suggest_int("n_layers",           2, 4)

    # Guard: stride must be < patch_size
    if stride >= patch_size:
        raise optuna.exceptions.TrialPruned()

    encoder_len        = horizon * CONFIG["encoder_multiplier"]
    tscv               = TimeSeriesSplit(n_splits=CONFIG["n_splits"])
    train_idx, val_idx = next(tscv.split(df_trainval))

    tr_df, va_df, _ = scale_fold(df_trainval.iloc[train_idx],
                                  df_trainval.iloc[val_idx])
    tr_ld, va_ld    = make_loaders(tr_df, va_df, encoder_len, horizon,
                                    batch_size=64)

    params   = {"d_model": d_model, "dropout": dropout,
                "patch_size": patch_size, "stride": stride,
                "n_layers": n_layers}
    model    = build_model(params, encoder_len, horizon)
    val_loss = train_model(model, tr_ld, va_ld, lr=lr, max_epochs=10,
                           patience=5,
                           label=f"optuna_h{horizon}_trial{trial.number}")
    del model
    torch.cuda.empty_cache()
    return val_loss

# ==============================================================================
# 11. OPTUNA EXECUTION
# ==============================================================================
print("\n--- Starting Optuna Hyperparameter Optimisation ---", flush=True)
best_params_dict = {}

for tune_horizon in [24, 168]:
    print(f"\n>>> Tuning for horizon: {tune_horizon}h", flush=True)
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial, h=tune_horizon: objective(trial, h),
                   n_trials=CONFIG["tuning_trials"])
    best_params_dict[tune_horizon] = study.best_params
    print(f"   Best Params for {tune_horizon}h: {study.best_params}",
          flush=True)

print("\n>>> Applying 168h parameters to 720h horizon", flush=True)
best_params_dict[720] = best_params_dict[168].copy()

serialisable = {str(k): v for k, v in best_params_dict.items()}
with open(CONFIG["best_params_path"], "w") as fh:
    json.dump(serialisable, fh, indent=2)
print(f"[SAVED] Best hyperparameters → {CONFIG['best_params_path']}",
      flush=True)

print("\n===== BEST HYPERPARAMETERS SUMMARY =====", flush=True)
for h, p in best_params_dict.items():
    print(f"  {h}h : {p}", flush=True)
print("=========================================\n", flush=True)

# ==============================================================================
# 12. FEATURE IMPORTANCE via attention weights
# ==============================================================================
importance_records = []

@torch.no_grad()
def extract_attention_importance(model: ImprovedQRPatchTST,
                                 val_df: pd.DataFrame,
                                 encoder_len: int, horizon: int,
                                 label: str, fold: int):
    attn_maps = []

    def make_hook(store):
        def h(module, inp, out):
            if isinstance(out, tuple) and len(out) == 2 and out[1] is not None:
                store.append(out[1].detach().cpu())
        return h

    hook      = model.encoder[0].attn.register_forward_hook(
                    make_hook(attn_maps))
    model_cpu = model.cpu()
    cpu_ds    = ResidualLoadDataset(val_df, encoder_len, horizon)
    cpu_ld    = DataLoader(cpu_ds, batch_size=32, shuffle=False)

    for x_past, x_fut, _ in cpu_ld:
        _ = model_cpu(x_past, x_fut)
        break
    hook.remove()
    model.to(DEVICE)

    if not attn_maps:
        print(f"[WARN] No attention maps for {label}", flush=True)
        return

    avg_attn  = attn_maps[0].mean(dim=[0, 1])
    vals      = avg_attn.numpy().astype(float)

    for i, v in enumerate(vals):
        importance_records.append({
            "label":           label,
            "importance_type": "patch_attention_layer1",
            "patch_index":     i,
            "importance":      float(v),
        })

    plot_path = (f"{RUN_DIR}/importance_{horizon}h_fold{fold}_"
                 f"patch_attention.png")
    try:
        fig, ax = plt.subplots(figsize=(10, max(3, len(vals) * 0.25)))
        ax.bar(range(len(vals)), vals, color="steelblue")
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
        print(f"[SAVED] Feature importance → "
              f"{CONFIG['feature_importance_path']}", flush=True)
    else:
        print("[WARN] No importance records to save.", flush=True)

# ==============================================================================
# 13. CROSS-VALIDATION
# ==============================================================================
horizons = [24, 168, 720]
tscv     = TimeSeriesSplit(n_splits=CONFIG["n_splits"])

for horizon in horizons:
    print(f"\n{'='*40}\n>>> CROSS-VALIDATION: {horizon}h\n{'='*40}",
          flush=True)
    encoder_len = horizon * CONFIG["encoder_multiplier"]
    params      = best_params_dict[horizon]

    for fold, (train_idx, val_idx) in enumerate(tscv.split(df_trainval)):
        tr_df, va_df, _ = scale_fold(df_trainval.iloc[train_idx],
                                      df_trainval.iloc[val_idx])
        tr_ld, va_ld    = make_loaders(tr_df, va_df, encoder_len, horizon)

        for run in range(CONFIG["robustness_runs"]):
            torch.manual_seed(42 + run)
            np.random.seed(42 + run)

            model    = build_model(params, encoder_len, horizon)
            label    = f"cv_h{horizon}_fold{fold+1}_run{run+1}"
            val_loss = train_model(
                model, tr_ld, va_ld,
                lr         = params["learning_rate"],
                max_epochs = 30,
                patience   = 8,
                label      = label,
            )

            _, preds, actuals = evaluate(model, va_ld)
            mae, rmse, smape  = compute_metrics(preds, actuals)

            append_csv({
                "horizon":  horizon, "fold": fold+1, "run": run+1,
                "mae":      mae,     "rmse": rmse,   "smape": smape,
                "val_loss": val_loss,
            }, CONFIG["results_path"])

            if run == 0:
                extract_attention_importance(
                    model, va_df, encoder_len, horizon,
                    label=f"cv_{horizon}h_fold{fold+1}",
                    fold=fold+1,
                )

            del model
            torch.cuda.empty_cache()

save_importance_csv()

# ==============================================================================
# 14. FINAL TEST EVALUATION
# ==============================================================================
print(f"\n{'='*40}\n>>> FINAL TEST EVALUATION\n{'='*40}", flush=True)

for horizon in horizons:
    encoder_len = horizon * CONFIG["encoder_multiplier"]
    params      = best_params_dict[horizon]

    scaler  = StandardScaler()
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
            lr         = params["learning_rate"],
            max_epochs = 30,
            patience   = 8,
            label      = label,
        )

        _, preds, actuals = evaluate(model, te_ld)
        mae, rmse, smape  = compute_metrics(preds, actuals)

        append_csv({
            "horizon":    horizon,  "run":       run+1,
            "test_mae":   mae,      "test_rmse": rmse,
            "test_smape": smape,    "test_loss": val_loss,
        }, CONFIG["test_results_path"])

        del model
        torch.cuda.empty_cache()

# ==============================================================================
# 15. FINAL SAVES
# ==============================================================================
save_carbon()

with open(CONFIG["best_params_path"], "w") as fh:
    json.dump(serialisable, fh, indent=2)

print("\n===== ALL TASKS COMPLETE =====", flush=True)
print(f"  Hyperparameters     → {CONFIG['best_params_path']}", flush=True)
print(f"  Carbon summary      → {CONFIG['carbon_summary_path']}", flush=True)
print(f"  Feature importance  → {CONFIG['feature_importance_path']}", flush=True)
print(f"  CV results          → {CONFIG['results_path']}", flush=True)
print(f"  Test results        → {CONFIG['test_results_path']}", flush=True)
print(f"  Full run log        → {LOG_PATH}", flush=True)
