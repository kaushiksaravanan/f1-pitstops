"""Quick vanilla LGB on a sample to verify Agent-D's preds correlate ~0.7-0.95."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import time

T0 = time.time()
ROOT = r"C:\Users\I587436\f1-pitstops"
tr = pd.read_parquet(rf"{ROOT}\features_train.parquet")
folds = np.load(rf"{ROOT}\folds.npy")
TARGET = "PitNextLap"
DROP = ["id", TARGET, "Driver", "Compound", "Race"]
FEATURES = [c for c in tr.columns if c not in DROP]

# Subsample 100k rows for speed
rng = np.random.RandomState(7)
samp = rng.choice(len(tr), 100_000, replace=False)
samp.sort()
X = tr[FEATURES].astype(np.float32).values[samp]
y = tr[TARGET].astype(np.int8).values[samp]
f = folds[samp]

# Single fold split
tr_idx = np.where(f != 0)[0]
va_idx = np.where(f == 0)[0]

params = dict(
    objective="binary", metric="auc", boosting_type="gbdt",
    learning_rate=0.08, num_leaves=63, min_data_in_leaf=200,
    feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
    num_threads=2, verbose=-1, seed=0,
)
dtr = lgb.Dataset(X[tr_idx], label=y[tr_idx])
dva = lgb.Dataset(X[va_idx], label=y[va_idx], reference=dtr)
m = lgb.train(params, dtr, num_boost_round=300, valid_sets=[dva],
              callbacks=[lgb.log_evaluation(period=100)])
vanilla_pred = m.predict(X[va_idx])
print(f"Vanilla AUC = {roc_auc_score(y[va_idx], vanilla_pred):.4f}")
np.save(rf"{ROOT}\_vanilla_sample_pred.npy", vanilla_pred.astype(np.float32))
np.save(rf"{ROOT}\_vanilla_sample_idx.npy", samp[va_idx].astype(np.int64))
print(f"Saved vanilla sample preds ({len(vanilla_pred)} rows). elapsed {time.time()-T0:.1f}s")
