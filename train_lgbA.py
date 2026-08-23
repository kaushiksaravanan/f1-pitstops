"""
Agent-A: LightGBM with heavy feature engineering for F1 pit-stops prediction.
5-fold CV using pre-built folds.npy.
"""
import os, sys, time, gc, json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

ROOT = r"C:\Users\I587436\f1-pitstops"
T0 = time.time()

# Force flush after every print
class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, x):
        for s in self.streams:
            s.write(x); s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

_logf = open(os.path.join(ROOT, "train_lgbA.log"), "w")
sys.stdout = _Tee(sys.__stdout__, _logf)
sys.stderr = _Tee(sys.__stderr__, _logf)

# ---------- Load ----------
print("[load] reading parquet ...")
train = pd.read_parquet(os.path.join(ROOT, "features_train.parquet"))
test = pd.read_parquet(os.path.join(ROOT, "features_test.parquet"))
folds = np.load(os.path.join(ROOT, "folds.npy"))
print(f"[load] train={train.shape}  test={test.shape}  folds={folds.shape}")

assert len(folds) == len(train)

# ---------- Feature engineering additions ----------
# Add a few interaction / lag-aware features that are cheap and likely useful for pitstop timing
def add_extras(df):
    # Stint progress relative to typical compound stint length
    df["TyreLife_per_compound_max"] = df["TyreLife"] / (df["cr_max_tyre"].replace(0, np.nan))
    df["TyreLife_per_compound_mean"] = df["TyreLife"] / (df["cr_mean_tyre"].replace(0, np.nan))
    # Late race indicator combined with tyre wear
    df["wear_x_progress"] = df["TyreLife"] * df["RaceProgress"]
    df["norm_wear_x_progress"] = df["Norm_TyreLife"] * df["RaceProgress"]
    # Lap-time degradation signals
    df["LapTime_jump_fwd"] = df["LapTime_diff_next"]
    df["LapTime_jump_bwd"] = df["LapTime_diff_prev"]
    df["LapTime_local_change"] = df["LapTime_diff_next"] - df["LapTime_diff_prev"]
    # Position dynamics
    df["pos_change_recent"] = df["Position_diff_prev"].fillna(0) + df["Position_diff_next"].fillna(0)
    # Laps-left vs stint observed span
    df["laps_left_ratio"] = df["laps_left_in_stint"] / (df["stint_observed_span"].replace(0, np.nan))
    # Driver pace deviation today
    df["pace_dev"] = df["LapTime (s)"] - (df["dr_mean_lap"])
    df["pace_dev_norm"] = df["pace_dev"] / (df["dr_std_lap"].replace(0, np.nan))
    # Stint signals interaction
    df["stint_progress_x_wear"] = df["stint_lap_progress"] * df["Norm_TyreLife"]
    return df

print("[fe] adding extra features ...")
train = add_extras(train)
test = add_extras(test)

# Replace inf with nan for safety
for df in (train, test):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

# ---------- Feature selection ----------
DROP = {"id", "PitNextLap", "Driver", "Compound", "Race", "PitStop", "LapTime"}
feature_cols = [c for c in train.columns if c not in DROP]
# Keep only numeric features
feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c])]
print(f"[fe] using {len(feature_cols)} features")

X = train[feature_cols].astype(np.float32)
y = train["PitNextLap"].astype(np.int8).values
X_test = test[feature_cols].astype(np.float32)
print(f"[fe] X={X.shape}  X_test={X_test.shape}  positive rate={y.mean():.4f}")

# ---------- LightGBM ----------
params = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.05,
    num_leaves=127,
    max_depth=-1,
    min_data_in_leaf=200,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    lambda_l2=1.0,
    lambda_l1=0.0,
    min_gain_to_split=0.0,
    num_threads=3,
    verbose=-1,
    seed=42,
    deterministic=True,
)
N_ROUNDS = 3000
EARLY_STOP = 80

oof = np.zeros(len(train), dtype=np.float32)
test_pred = np.zeros(len(test), dtype=np.float32)
fold_aucs = []
best_iters = []

for f in range(5):
    t0 = time.time()
    val_idx = np.where(folds == f)[0]
    tr_idx = np.where(folds != f)[0]
    print(f"\n[fold {f}] train={len(tr_idx)} val={len(val_idx)}")

    dtr = lgb.Dataset(X.iloc[tr_idx], label=y[tr_idx])
    dva = lgb.Dataset(X.iloc[val_idx], label=y[val_idx], reference=dtr)

    model = lgb.train(
        params,
        dtr,
        num_boost_round=N_ROUNDS,
        valid_sets=[dva],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(EARLY_STOP, verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )
    bi = model.best_iteration
    best_iters.append(bi)
    p_val = model.predict(X.iloc[val_idx], num_iteration=bi)
    oof[val_idx] = p_val.astype(np.float32)
    auc = roc_auc_score(y[val_idx], p_val)
    fold_aucs.append(auc)
    print(f"[fold {f}] best_iter={bi}  AUC={auc:.5f}  ({time.time()-t0:.1f}s)")

    p_te = model.predict(X_test, num_iteration=bi)
    test_pred += p_te.astype(np.float32) / 5.0

    del model, dtr, dva, p_val, p_te
    gc.collect()

oof_auc = roc_auc_score(y, oof)
print(f"\n=== overall OOF AUC = {oof_auc:.6f} ===")
print(f"per-fold AUCs: {[round(a,5) for a in fold_aucs]}")
print(f"best_iters: {best_iters}")

# ---------- Save outputs ----------
np.save(os.path.join(ROOT, "oof_lgbA.npy"), oof.astype(np.float32))
np.save(os.path.join(ROOT, "test_lgbA.npy"), test_pred.astype(np.float32))
print(f"saved oof_lgbA.npy ({oof.shape}) and test_lgbA.npy ({test_pred.shape})")

runtime = time.time() - T0
summary = f"""Agent-A LightGBM — F1 Pit Stops (playground-series-s6e5)
================================================================
OOF AUC (5-fold): {oof_auc:.6f}
Per-fold AUCs:    {[round(a,5) for a in fold_aucs]}
Best iters/fold:  {best_iters}
Runtime:          {runtime:.1f}s

Features used: {len(feature_cols)}
Feature drop list: {sorted(DROP)}
Engineered additions on top of base 88 cols:
  - TyreLife_per_compound_max, TyreLife_per_compound_mean
  - wear_x_progress, norm_wear_x_progress
  - LapTime_jump_fwd, LapTime_jump_bwd, LapTime_local_change
  - pos_change_recent
  - laps_left_ratio
  - pace_dev, pace_dev_norm
  - stint_progress_x_wear

LightGBM params:
{json.dumps(params, indent=2)}
num_boost_round={N_ROUNDS}, early_stopping={EARLY_STOP}

Files written:
  oof_lgbA.npy   shape={oof.shape}   dtype=float32
  test_lgbA.npy  shape={test_pred.shape}   dtype=float32
"""
with open(os.path.join(ROOT, "summary_lgbA.txt"), "w") as f:
    f.write(summary)
print("\n" + summary)
