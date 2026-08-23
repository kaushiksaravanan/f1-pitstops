"""Train ONE fold of XGBoost. Args: fold_idx
Saves per-fold OOF segment + test predictions for that fold."""
import sys
import time
import gc
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

f = int(sys.argv[1])
T0 = time.time()
ROOT = 'C:/Users/I587436/f1-pitstops'
SEED = 2026

train = pd.read_parquet(f'{ROOT}/features_train.parquet')
test = pd.read_parquet(f'{ROOT}/features_test.parquet')
folds = np.load(f'{ROOT}/folds.npy')

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
X_arr = X[common].astype(np.float32).values
X_test_arr = X_test[common].astype(np.float32).values
del train, test, ext, ext_small, ext_stint, X, X_test
gc.collect()

params = dict(
    objective='binary:logistic',
    eval_metric='auc',
    tree_method='hist',
    max_depth=7,
    learning_rate=0.04,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_lambda=1.0,
    reg_alpha=0.1,
    min_child_weight=4,
    nthread=3,
    seed=SEED,
)

tr_idx = np.where(folds != f)[0]
va_idx = np.where(folds == f)[0]
dtr = xgb.DMatrix(X_arr[tr_idx], label=y[tr_idx])
dva = xgb.DMatrix(X_arr[va_idx], label=y[va_idx])
del X_arr; gc.collect()

print(f'[fold {f}] training start, n_train={len(tr_idx)} n_val={len(va_idx)}', flush=True)
model = xgb.train(
    params, dtr,
    num_boost_round=1500,
    evals=[(dva, 'val')],
    early_stopping_rounds=80,
    verbose_eval=300,
)
bi = model.best_iteration
val_pred = model.predict(dva, iteration_range=(0, bi + 1))
auc = roc_auc_score(y[va_idx], val_pred)
print(f'[fold {f}] AUC={auc:.6f} best_iter={bi} elapsed={time.time()-T0:.0f}s', flush=True)

del dtr, dva; gc.collect()
dte = xgb.DMatrix(X_test_arr)
test_pred = model.predict(dte, iteration_range=(0, bi + 1))

np.save(f'{ROOT}/oof_xgbC_fold{f}.npy', val_pred.astype(np.float32))
np.save(f'{ROOT}/oof_xgbC_fold{f}_idx.npy', va_idx)
np.save(f'{ROOT}/test_xgbC_fold{f}.npy', test_pred.astype(np.float32))
with open(f'{ROOT}/fold{f}_meta.txt', 'w') as fp:
    fp.write(f'auc={auc:.6f}\nbest_iter={bi}\nelapsed={time.time()-T0:.1f}\nn_features={len(common)}\n')
print(f'[fold {f}] saved', flush=True)
