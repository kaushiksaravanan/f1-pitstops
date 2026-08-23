"""Agent-B: CatBoost with native categorical handling, 5-fold OOF."""
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

t_start = time.time()

# Load
tr = pd.read_parquet('features_train.parquet')
te = pd.read_parquet('features_test.parquet')
folds = np.load('folds.npy')
print(f'Loaded train={tr.shape}, test={te.shape}, folds={folds.shape} in {time.time()-t_start:.1f}s', flush=True)

y = tr['PitNextLap'].astype(int).values
# Drop id, target, and the redundant *_code columns (we use string Driver/Race/Compound natively)
drop_cols = ['id', 'PitNextLap', 'Driver_code', 'Race_code', 'Compound_code']
feat_cols = [c for c in tr.columns if c not in drop_cols]
cat_features = ['Driver', 'Compound', 'Race']

X = tr[feat_cols].copy()
Xte = te[feat_cols].copy()

# CatBoost requires string cat columns; fill NA
for c in cat_features:
    X[c] = X[c].fillna('NA').astype(str)
    Xte[c] = Xte[c].fillna('NA').astype(str)

# Replace remaining NaNs in numerics is unnecessary — CatBoost handles them
print(f'n features: {len(feat_cols)}, cat: {cat_features}', flush=True)

PARAMS = dict(
    iterations=350,
    depth=7,
    learning_rate=0.12,
    l2_leaf_reg=3,
    bootstrap_type='Bernoulli',
    subsample=0.85,
    eval_metric='AUC',
    loss_function='Logloss',
    random_seed=42,
    od_type='Iter',
    od_wait=60,
    thread_count=10,
    verbose=50,
    allow_writing_files=False,
    max_ctr_complexity=2,
)

oof = np.zeros(len(tr), dtype=np.float32)
test_pred = np.zeros(len(te), dtype=np.float32)
fold_aucs = []
fold_iters = []

for fold in range(5):
    t_fold = time.time()
    trn_idx = np.where(folds != fold)[0]
    val_idx = np.where(folds == fold)[0]
    X_trn, X_val = X.iloc[trn_idx], X.iloc[val_idx]
    y_trn, y_val = y[trn_idx], y[val_idx]

    model = CatBoostClassifier(**PARAMS)
    model.fit(
        X_trn, y_trn,
        eval_set=(X_val, y_val),
        cat_features=cat_features,
        use_best_model=True,
    )
    val_pred = model.predict_proba(X_val)[:, 1]
    oof[val_idx] = val_pred
    test_pred += model.predict_proba(Xte)[:, 1] / 5.0

    auc = roc_auc_score(y_val, val_pred)
    bi = model.get_best_iteration()
    fold_aucs.append(auc)
    fold_iters.append(bi)
    print(f'Fold {fold}: AUC={auc:.5f} best_iter={bi} time={time.time()-t_fold:.1f}s '
          f'cum={time.time()-t_start:.1f}s', flush=True)
    # Incremental save in case of kill
    np.save('oof_catB.npy', oof.astype(np.float32))
    # Rescale partial test_pred to mean over folds completed so far (fold+1 of them)
    np.save('test_catB.npy', (test_pred * 5.0 / (fold + 1)).astype(np.float32))

oof_auc = roc_auc_score(y, oof)
runtime = time.time() - t_start
print(f'\n=== OOF AUC: {oof_auc:.5f} === total runtime: {runtime:.1f}s', flush=True)
print(f'Fold AUCs: {[f"{a:.5f}" for a in fold_aucs]}')
print(f'Best iters: {fold_iters}')

# Save outputs
np.save('oof_catB.npy', oof.astype(np.float32))
np.save('test_catB.npy', test_pred.astype(np.float32))

with open('summary_catB.txt', 'w') as f:
    f.write('Agent-B: CatBoost with native categorical handling\n')
    f.write('=' * 60 + '\n')
    f.write(f'OOF AUC (5-fold): {oof_auc:.6f}\n')
    f.write(f'Per-fold AUCs:    {[round(a, 6) for a in fold_aucs]}\n')
    f.write(f'Per-fold best_iter: {fold_iters}\n')
    f.write(f'Runtime: {runtime:.1f}s ({runtime/60:.1f} min)\n')
    f.write('\n--- Hyperparameters ---\n')
    for k, v in PARAMS.items():
        f.write(f'{k}: {v}\n')
    f.write('\n--- Feature handling ---\n')
    f.write(f'Total features used: {len(feat_cols)}\n')
    f.write(f'Categorical (passed as strings via cat_features): {cat_features}\n')
    f.write(f'Dropped *_code redundant cols: Driver_code, Race_code, Compound_code\n')
    f.write(f'Dropped non-features: id, PitNextLap\n')
    f.write('\n--- Outputs ---\n')
    f.write('oof_catB.npy   shape=(439140,) float32\n')
    f.write('test_catB.npy  shape=(188165,) float32\n')

print('\nSaved oof_catB.npy, test_catB.npy, summary_catB.txt')
