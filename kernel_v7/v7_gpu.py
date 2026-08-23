import os, gc, time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb

t0 = time.time()
NTHREADS = 16  # use all cores
WD = '/teamspace/studios/this_studio'

tr = pd.read_csv(f'{WD}/train.csv')
te = pd.read_csv(f'{WD}/test.csv')
orig = pd.read_csv(f'{WD}/f1_strategy_dataset_v4.csv')
print(f'train={tr.shape} test={te.shape} orig={orig.shape}')
print(f'orig cols: {list(orig.columns)}')

if 'PitNextLap' not in orig.columns:
    orig = orig.sort_values(['Driver','Race','Year','LapNumber']).reset_index(drop=True)
    g = orig.groupby(['Driver','Race','Year'])
    orig['PitNextLap'] = g['PitStop'].shift(-1).fillna(0).astype(int)
    print(f'Derived PitNextLap: {orig["PitNextLap"].mean():.4f} rate')

common_cols = [c for c in tr.columns if c in orig.columns and c != 'id']
print(f'common cols: {common_cols}')

max_id = max(tr['id'].max(), te['id'].max()) + 1
orig_aligned = orig[common_cols].copy()
orig_aligned['id'] = range(max_id, max_id + len(orig_aligned))
orig_aligned['_is_orig'] = 1
tr['_is_orig'] = 0
tr_combined = pd.concat([tr, orig_aligned], ignore_index=True)
print(f'Combined: {tr_combined.shape}')

def build_features(tr_df, te_df):
    te_df = te_df.copy(); te_df['PitNextLap'] = np.nan; te_df['_is_orig'] = 0
    df = pd.concat([tr_df.assign(_src='train'), te_df.assign(_src='test')], ignore_index=True)
    df['LapTime'] = df['LapTime (s)']
    gkey = ['Driver','Race','Year','Stint']
    g = df.groupby(gkey).agg(stint_max_tyre=('TyreLife','max'),stint_min_tyre=('TyreLife','min'),
        stint_n_obs=('TyreLife','count'),stint_max_lap=('LapNumber','max'),
        stint_min_lap=('LapNumber','min'),stint_pitstop_sum=('PitStop','sum')).reset_index()
    df = df.merge(g, on=gkey, how='left')
    df['Norm_TyreLife'] = df['TyreLife'] / df['stint_max_tyre'].clip(lower=1)
    df['is_max_in_stint'] = (df['TyreLife']==df['stint_max_tyre']).astype(np.int8)
    df['laps_left_in_stint'] = df['stint_max_tyre'] - df['TyreLife']
    df['stint_observed_span'] = df['stint_max_lap'] - df['stint_min_lap'] + 1
    df['stint_lap_offset'] = df['LapNumber'] - df['stint_min_lap']
    df['stint_lap_progress'] = df['stint_lap_offset'] / df['stint_observed_span'].clip(lower=1)
    gkey2 = ['Driver','Race','Year']
    g2 = df.groupby(gkey2).agg(dr_max_lap=('LapNumber','max'),dr_max_stint=('Stint','max'),
        dr_max_tyre=('TyreLife','max'),dr_pitstop_total=('PitStop','sum'),
        dr_mean_lap=('LapTime','mean'),dr_min_lap=('LapTime','min'),dr_std_lap=('LapTime','std')).reset_index()
    df = df.merge(g2, on=gkey2, how='left')
    df['lap_share'] = df['LapNumber'] / df['dr_max_lap'].clip(lower=1)
    gkey3 = ['Race','Year']
    g3 = df.groupby(gkey3).agg(race_max_lap=('LapNumber','max'),race_mean_lap=('LapTime','mean'),
        race_min_lap=('LapTime','min'),race_pitstop_rate=('PitStop','mean')).reset_index()
    df = df.merge(g3, on=gkey3, how='left')
    gkey4 = ['Race','Year','Compound']
    g4 = df.groupby(gkey4).agg(cr_max_tyre=('TyreLife','max'),cr_mean_tyre=('TyreLife','mean'),
        cr_pitstop_rate=('PitStop','mean')).reset_index()
    df = df.merge(g4, on=gkey4, how='left')
    df['tyre_vs_compound_max'] = df['TyreLife'] / df['cr_max_tyre'].clip(lower=1)
    df['tyre_vs_compound_mean'] = df['TyreLife'] / df['cr_mean_tyre'].clip(lower=1)
    df['LapTime_vs_race_min'] = df['LapTime'] - df['race_min_lap']
    df['LapTime_vs_race_mean'] = df['LapTime'] - df['race_mean_lap']
    df['LapTime_vs_dr_min'] = df['LapTime'] - df['dr_min_lap']
    df['LapTime_vs_dr_mean'] = df['LapTime'] - df['dr_mean_lap']
    df['Position_log'] = np.log1p(df['Position'])
    df['LapsToEnd'] = df['race_max_lap'] - df['LapNumber']
    df['StintsToGo'] = df['dr_max_stint'] - df['Stint']
    compound_order = {'SOFT':0,'MEDIUM':1,'HARD':2,'INTERMEDIATE':3,'WET':4}
    df['Compound_ord'] = df['Compound'].map(compound_order).fillna(-1).astype(int)
    for c in ['Driver','Race','Compound']:
        df[f'{c}_code'] = df[c].astype('category').cat.codes
    df['stint_tyre_pct_rank'] = df.groupby(gkey)['TyreLife'].rank(pct=True)
    df['stint_lap_pct_rank'] = df.groupby(gkey)['LapNumber'].rank(pct=True)
    df = df.sort_values(['Driver','Race','Year','LapNumber']).reset_index(drop=True)
    g_dry = df.groupby(['Driver','Race','Year'])
    df['LapTime_prev'] = g_dry['LapTime'].shift(1)
    df['LapTime_diff_prev'] = df['LapTime'] - df['LapTime_prev']
    df['LapTime_roll3_mean'] = g_dry['LapTime'].transform(lambda x: x.rolling(3,min_periods=1).mean())
    df['LapTime_roll5_mean'] = g_dry['LapTime'].transform(lambda x: x.rolling(5,min_periods=1).mean())
    df['LapTime_degradation'] = df['LapTime'] - df['LapTime_roll5_mean']
    df['Position_prev'] = g_dry['Position'].shift(1)
    df['Position_diff_prev'] = df['Position'] - df['Position_prev']
    df['Stint_prev'] = g_dry['Stint'].shift(1)
    df['Stint_changed_prev'] = (df['Stint_prev'] != df['Stint']).astype(np.int8)
    df['TyreLife_prev'] = g_dry['TyreLife'].shift(1)
    df['TyreLife_diff_prev'] = df['TyreLife'] - df['TyreLife_prev']
    df['Compound_prev'] = g_dry['Compound_ord'].shift(1)
    df['Compound_changed_prev'] = (df['Compound_prev'] != df['Compound_ord']).astype(np.int8)
    df['TyreLife_x_Position'] = df['TyreLife'] * df['Position']
    df['Norm_TyreLife_x_LapsToEnd'] = df['Norm_TyreLife'] * df['LapsToEnd']
    df['stint_progress_x_position'] = df['stint_lap_progress'] * df['Position']
    df['degradation_x_tyre'] = df['LapTime_degradation'] * df['TyreLife']
    # TE
    train_mask = df['_src']=='train'
    global_mean = float(np.nanmean(df.loc[train_mask, 'PitNextLap'].values))
    SMOOTH = 50
    for col in ['Race','Driver','Compound']:
        agg = df.loc[train_mask,[col,'PitNextLap']].groupby(col)['PitNextLap'].agg(['mean','count']).reset_index()
        agg[f'{col}_TE'] = (agg['mean']*agg['count']+global_mean*SMOOTH)/(agg['count']+SMOOTH)
        df = df.merge(agg[[col,f'{col}_TE']], on=col, how='left')
        df[f'{col}_TE'] = df[f'{col}_TE'].fillna(global_mean)
    df = df.sort_values('id').reset_index(drop=True)
    train_df = df[df['_src']=='train'].drop(columns=['_src']).reset_index(drop=True)
    test_df = df[df['_src']=='test'].drop(columns=['_src','PitNextLap']).reset_index(drop=True)
    return train_df, test_df

print('=== building features ===')
train_df, test_df = build_features(tr_combined, te)
print(f'train_df={train_df.shape} test_df={test_df.shape}')

y_all = train_df['PitNextLap'].astype(int).values
is_orig = train_df['_is_orig'].values
test_ids = test_df['id'].values

drop_cols = ['id','PitNextLap','Driver','Race','Compound','_is_orig']
feat_cols = [c for c in train_df.columns if c not in drop_cols]
print(f'features: {len(feat_cols)}')

X_all = train_df[feat_cols].astype(np.float32).values
X_te = test_df[feat_cols].astype(np.float32).values

synth_mask = is_orig == 0
X_synth = X_all[synth_mask]
y_synth = y_all[synth_mask]
X_orig = X_all[~synth_mask]
y_orig = y_all[~synth_mask]
print(f'synth={X_synth.shape} orig={X_orig.shape}')
del train_df, test_df, tr_combined, tr, te; gc.collect()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = np.full(len(y_synth), -1, dtype=np.int8)
for f, (_, vidx) in enumerate(skf.split(np.zeros(len(y_synth)), y_synth)):
    folds[vidx] = f

def run_lgb_gpu(name, params, seed):
    oof = np.zeros(len(y_synth), dtype=np.float64)
    test_pred = np.zeros(len(X_te), dtype=np.float64)
    p = dict(params)
    p['objective']='binary'; p['metric']='auc'
    p['verbose']=-1; p['num_threads']=NTHREADS; p['seed']=seed
    # LGB uses CPU (no OpenCL on this machine)
    print(f'\n[{name}] LGB-GPU 5-fold...')
    for f in range(5):
        tr_mask = folds != f; va_mask = folds == f
        X_fold_tr = np.vstack([X_synth[tr_mask], X_orig])
        y_fold_tr = np.concatenate([y_synth[tr_mask], y_orig])
        dtrain = lgb.Dataset(X_fold_tr, y_fold_tr)
        dvalid = lgb.Dataset(X_synth[va_mask], y_synth[va_mask], reference=dtrain)
        booster = lgb.train(p, dtrain, num_boost_round=5000, valid_sets=[dvalid],
                            callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)])
        oof[va_mask] = booster.predict(X_synth[va_mask], num_iteration=booster.best_iteration)
        test_pred += booster.predict(X_te, num_iteration=booster.best_iteration) / 5.0
        print(f'  fold {f}: best_iter={booster.best_iteration}')
        del booster, dtrain, dvalid; gc.collect()
    auc = roc_auc_score(y_synth, oof)
    print(f'[{name}] OOF AUC = {auc:.6f}')
    return oof.astype(np.float32), test_pred.astype(np.float32), auc

def run_xgb_gpu(name, params, seed):
    oof = np.zeros(len(y_synth), dtype=np.float64)
    test_pred = np.zeros(len(X_te), dtype=np.float64)
    p = dict(params)
    p['objective']='binary:logistic'; p['eval_metric']='auc'
    p['tree_method']='hist'; p['device']='cuda'
    p['nthread']=NTHREADS; p['seed']=seed; p['verbosity']=0
    print(f'\n[{name}] XGB-GPU 5-fold...')
    for f in range(5):
        tr_mask = folds != f; va_mask = folds == f
        X_fold_tr = np.vstack([X_synth[tr_mask], X_orig])
        y_fold_tr = np.concatenate([y_synth[tr_mask], y_orig])
        dtr = xgb.DMatrix(X_fold_tr, y_fold_tr)
        dva = xgb.DMatrix(X_synth[va_mask], y_synth[va_mask])
        dte = xgb.DMatrix(X_te)
        booster = xgb.train(p, dtr, num_boost_round=5000, evals=[(dva,'va')],
                            early_stopping_rounds=150, verbose_eval=False)
        oof[va_mask] = booster.predict(dva)
        test_pred += booster.predict(dte) / 5.0
        print(f'  fold {f}: best_iter={booster.best_iteration}')
        del booster, dtr, dva, dte; gc.collect()
    auc = roc_auc_score(y_synth, oof)
    print(f'[{name}] OOF AUC = {auc:.6f}')
    return oof.astype(np.float32), test_pred.astype(np.float32), auc

# Train 4 models on GPU
print(f'\n=== TRAINING ON GPU ({NTHREADS} threads) ===')
oof1, t1, _ = run_lgb_gpu('LGB1', dict(num_leaves=127, learning_rate=0.03, min_data_in_leaf=200,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5, lambda_l2=1.0), 42)
oof2, t2, _ = run_lgb_gpu('LGB2', dict(num_leaves=255, learning_rate=0.03, min_data_in_leaf=150,
    feature_fraction=0.75, bagging_fraction=0.85, bagging_freq=5, lambda_l2=1.5), 2026)
oof3, t3, _ = run_xgb_gpu('XGB1', dict(max_depth=8, eta=0.03, subsample=0.8,
    colsample_bytree=0.7, reg_lambda=1.0, reg_alpha=0.1), 2026)
oof4, t4, _ = run_xgb_gpu('XGB2', dict(max_depth=10, eta=0.02, subsample=0.75,
    colsample_bytree=0.65, reg_lambda=2.0, reg_alpha=0.2, max_bin=512), 777)

# DART on CPU (DART doesn't benefit from GPU as much)
print('\n[DART] LGB-CPU 5-fold...')
dart_oof = np.zeros(len(y_synth), dtype=np.float64)
dart_test = np.zeros(len(X_te), dtype=np.float64)
dp = dict(boosting='dart', num_leaves=63, learning_rate=0.05, min_data_in_leaf=200,
    feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=5, lambda_l2=1.0,
    drop_rate=0.1, skip_drop=0.5, max_drop=50, objective='binary', metric='auc',
    verbose=-1, num_threads=NTHREADS, seed=11)
for f in range(5):
    tr_mask = folds != f; va_mask = folds == f
    X_fold_tr = np.vstack([X_synth[tr_mask], X_orig])
    y_fold_tr = np.concatenate([y_synth[tr_mask], y_orig])
    dtrain = lgb.Dataset(X_fold_tr, y_fold_tr)
    dvalid = lgb.Dataset(X_synth[va_mask], y_synth[va_mask], reference=dtrain)
    booster = lgb.train(dp, dtrain, num_boost_round=2500, valid_sets=[dvalid],
                        callbacks=[lgb.log_evaluation(0)])
    dart_oof[va_mask] = booster.predict(X_synth[va_mask])
    dart_test += booster.predict(X_te) / 5.0
    print(f'  fold {f} done')
    del booster, dtrain, dvalid; gc.collect()
print(f'[DART] OOF AUC = {roc_auc_score(y_synth, dart_oof):.6f}')

# Blend
print('\n=== BLEND ===')
names = ['lgb1','lgb2','xgb1','xgb2','dart']
oofs = [oof1, oof2, oof3, oof4, dart_oof.astype(np.float32)]
tests = [t1, t2, t3, t4, dart_test.astype(np.float32)]
oof_mat = np.stack(oofs, axis=1).astype(np.float64)
test_mat = np.stack(tests, axis=1).astype(np.float64)

for n, o in zip(names, oofs):
    print(f'  {n}: {roc_auc_score(y_synth, o):.6f}')

def neg_auc(w, X, y):
    w = np.abs(w); w /= max(w.sum(), 1e-9)
    return -roc_auc_score(y, X @ w)

w0 = np.ones(len(names))/len(names)
res = minimize(neg_auc, x0=w0, args=(oof_mat, y_synth), method='Nelder-Mead',
               options={'xatol':1e-8,'fatol':1e-9,'maxiter':20000})
w = np.abs(res.x); w /= w.sum()
print(f'weights: {dict(zip(names, w.round(4)))}')
blend_oof = oof_mat @ w
blend_test = test_mat @ w
print(f'\n>>> BLEND OOF AUC = {roc_auc_score(y_synth, blend_oof):.6f}')

avg_test = test_mat.mean(axis=1)
print(f'simple avg OOF AUC = {roc_auc_score(y_synth, oof_mat.mean(axis=1)):.6f}')

# Save best
best_test = blend_test if roc_auc_score(y_synth, blend_oof) > roc_auc_score(y_synth, oof_mat.mean(axis=1)) else avg_test
pd.DataFrame({'id': test_ids, 'PitNextLap': best_test}).to_csv(f'{WD}/submission_v7.csv', index=False)
print(f'\nRuntime: {(time.time()-t0)/60:.1f} min')
print('DONE - submission_v7.csv written')
