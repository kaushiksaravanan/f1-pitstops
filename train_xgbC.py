"""Agent-C: XGBoost lean version - fixed n_estimators, lower memory."""
import time
import gc
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

T0 = time.time()

ROOT = 'C:/Users/I587436/f1-pitstops'
SEED = 2026

print('[load] data', flush=True)
train = pd.read_parquet(f'{ROOT}/features_train.parquet')
test = pd.read_parquet(f'{ROOT}/features_test.parquet')
folds = np.load(f'{ROOT}/folds.npy')

# External join
print('[external] joining', flush=True)
ext = pd.read_csv(f'{ROOT}/external/f1_strategy_dataset_v4.csv')
ext_keys = ['Driver', 'Race', 'Year', 'Stint', 'TyreLife']
ext_small = ext[ext_keys + ['Normalized_TyreLife']].drop_duplicates(ext_keys).rename(columns={'Normalized_TyreLife': 'Ext_NormTyreLife'})
train = train.merge(ext_small, on=ext_keys, how='left')
test = test.merge(ext_small, on=ext_keys, how='left')
train['Ext_vs_Internal_Norm'] = train['Ext_NormTyreLife'] - train['Norm_TyreLife']
test['Ext_vs_Internal_Norm'] = test['Ext_NormTyreLife'] - test['Norm_TyreLife']

ext_stint = ext.groupby(['Driver', 'Race', 'Year', 'Stint']).agg(
    Ext_StintMaxTyre=('TyreLife', 'max'),
    Ext_StintMinTyre=('TyreLife', 'min'),
).reset_index()
train = train.merge(ext_stint, on=['Driver', 'Race', 'Year', 'Stint'], how='left')
test = test.merge(ext_stint, on=['Driver', 'Race', 'Year', 'Stint'], how='left')
train['Ext_StintLapsRemaining'] = train['Ext_StintMaxTyre'] - train['TyreLife']
test['Ext_StintLapsRemaining'] = test['Ext_StintMaxTyre'] - test['TyreLife']

drop_cols = ['id', 'PitNextLap', 'Driver', 'Race', 'Compound']
y = train['PitNextLap'].astype(np.int8).values
X = train.drop(columns=drop_cols).select_dtypes(exclude=['object'])
X_test = test.drop(columns=[c for c in drop_cols if c in test.columns]).select_dtypes(exclude=['object'])
common = [c for c in X.columns if c in X_test.columns]
X = X[common].astype(np.float32).values  # convert to ndarray to save memory
X_test_arr = X_test[common].astype(np.float32).values
del train, test, ext, ext_small, ext_stint
gc.collect()
print(f'  feat cols={len(common)}, X={X.shape}, X_test={X_test_arr.shape}', flush=True)

# Build test DMatrix once
dte = xgb.DMatrix(X_test_arr)

params = dict(
    objective='binary:logistic',
    eval_metric='auc',
    tree_method='hist',
    max_depth=7,           # slightly shallower for speed
    learning_rate=0.04,    # slightly higher LR
    subsample=0.8,
    colsample_bytree=0.7,
    reg_lambda=1.0,
    reg_alpha=0.1,
    min_child_weight=4,
    nthread=3,
    seed=SEED,
)

# Use fixed iters based on fold 0's findings: ~1400 best at depth 8/eta 0.03.
# At depth 7/eta 0.04 -> probably ~1100-1300. Use early stop with smaller dval.
NUM_ROUNDS = 1500
ES_ROUNDS = 80

oof = np.zeros(len(y), dtype=np.float32)
test_pred = np.zeros(X_test_arr.shape[0], dtype=np.float32)
fold_aucs = []
best_iters = []

for f in range(5):
    print(f'[fold {f}] start at {time.time()-T0:.0f}s', flush=True)
    tr_idx = np.where(folds != f)[0]
    va_idx = np.where(folds == f)[0]
    dtr = xgb.DMatrix(X[tr_idx], label=y[tr_idx])
    dva = xgb.DMatrix(X[va_idx], label=y[va_idx])

    model = xgb.train(
        params, dtr,
        num_boost_round=NUM_ROUNDS,
        evals=[(dva, 'val')],
        early_stopping_rounds=ES_ROUNDS,
        verbose_eval=300,
    )
    bi = model.best_iteration
    best_iters.append(bi)
    val_pred = model.predict(dva, iteration_range=(0, bi + 1))
    auc = roc_auc_score(y[va_idx], val_pred)
    fold_aucs.append(auc)
    print(f'  fold {f}: AUC={auc:.6f} best_iter={bi} elapsed={time.time()-T0:.0f}s', flush=True)
    oof[va_idx] = val_pred
    test_pred += model.predict(dte, iteration_range=(0, bi + 1)) / 5.0

    del dtr, dva, model
    gc.collect()

oof_auc = roc_auc_score(y, oof)
elapsed = time.time() - T0
print(f'\n[result] OOF AUC = {oof_auc:.6f}', flush=True)
print(f'[result] fold AUCs: {fold_aucs}', flush=True)
print(f'[result] elapsed: {elapsed:.1f}s', flush=True)

np.save(f'{ROOT}/oof_xgbC.npy', oof.astype(np.float32))
np.save(f'{ROOT}/test_xgbC.npy', test_pred.astype(np.float32))
with open(f'{ROOT}/summary_xgbC.txt', 'w') as fout:
    fout.write('Agent-C: XGBoost (hist tree, seed=2026)\n')
    fout.write('==========================================\n\n')
    fout.write(f'OOF AUC: {oof_auc:.6f}\n')
    fout.write(f'Fold AUCs: {fold_aucs}\n')
    fout.write(f'Best iters: {best_iters}\n')
    fout.write(f'Runtime: {elapsed:.1f}s\n\n')
    fout.write(f'Params: {params}\n\n')
    fout.write('Feature additions over base parquet:\n')
    fout.write('  - Ext_NormTyreLife: joined external Normalized_TyreLife on (Driver,Race,Year,Stint,TyreLife) ~3% match\n')
    fout.write('  - Ext_vs_Internal_Norm: difference vs internal Norm_TyreLife\n')
    fout.write('  - Ext_StintMaxTyre, Ext_StintMinTyre: stint-level aggregates from external dataset\n')
    fout.write('  - Ext_StintLapsRemaining: Ext_StintMaxTyre - TyreLife (predicted laps left in stint)\n')
    fout.write(f'Total features: {len(common)}\n')
print('[save] done', flush=True)
