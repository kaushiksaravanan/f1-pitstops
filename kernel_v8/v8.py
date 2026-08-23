import subprocess, sys, os
# Install deps
subprocess.run([sys.executable, '-m', 'pip', 'install', 'pytabkit', '-q'], check=True)

# Check RealMLP
from pytabkit.models.sklearn.sklearn_interfaces import RealMLP_TD_Classifier
print('RealMLP_TD_Classifier imported OK')

# Download public submissions for blending
os.environ['KAGGLE_USERNAME'] = 'kaushiksarav'
os.environ['KAGGLE_KEY'] = 'a5026d0acb433da631c87669ace3f8b8'
os.makedirs('/teamspace/studios/this_studio/pub', exist_ok=True)
subprocess.run('kaggle kernels output yekenot/ps-s6-e5-realmlp-pytabkit -p /teamspace/studios/this_studio/pub/realmlp/', shell=True)
subprocess.run('kaggle kernels output kospintr/pitstop-catb-hgbc-xgb-lgbm-realmlp-baseline -p /teamspace/studios/this_studio/pub/baseline/', shell=True)

import gc, time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb

t0 = time.time()
NTHREADS = 16
WD = '/teamspace/studios/this_studio'

tr = pd.read_csv(f'{WD}/train.csv')
te = pd.read_csv(f'{WD}/test.csv')
orig = pd.read_csv(f'{WD}/f1_strategy_dataset_v4.csv')
print(f'train={tr.shape} test={te.shape} orig={orig.shape}')

# Original already has PitNextLap
common_cols = [c for c in tr.columns if c in orig.columns and c != 'id']
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
        stint_max_lap=('LapNumber','max'),stint_min_lap=('LapNumber','min'),
        stint_pitstop_sum=('PitStop','sum')).reset_index()
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
y_all = train_df['PitNextLap'].astype(int).values
is_orig = train_df['_is_orig'].values
test_ids = test_df['id'].values
drop_cols = ['id','PitNextLap','Driver','Race','Compound','_is_orig']
feat_cols = [c for c in train_df.columns if c not in drop_cols]
print(f'features: {len(feat_cols)}')

X_all = train_df[feat_cols].astype(np.float32).values
X_te = test_df[feat_cols].astype(np.float32).values
synth_mask = is_orig == 0
X_synth = X_all[synth_mask]; y_synth = y_all[synth_mask]
X_orig = X_all[~synth_mask]; y_orig = y_all[~synth_mask]
print(f'synth={X_synth.shape} orig={X_orig.shape}')
del train_df, test_df, tr_combined; gc.collect()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = np.full(len(y_synth), -1, dtype=np.int8)
for f, (_, vidx) in enumerate(skf.split(np.zeros(len(y_synth)), y_synth)):
    folds[vidx] = f

# ===== MODEL 1: LGB =====
print('\n[LGB] training...')
oof_lgb = np.zeros(len(y_synth), dtype=np.float64)
test_lgb = np.zeros(len(X_te), dtype=np.float64)
for f in range(5):
    tr_mask = folds != f; va_mask = folds == f
    X_fold_tr = np.vstack([X_synth[tr_mask], X_orig])
    y_fold_tr = np.concatenate([y_synth[tr_mask], y_orig])
    dtrain = lgb.Dataset(X_fold_tr, y_fold_tr)
    dvalid = lgb.Dataset(X_synth[va_mask], y_synth[va_mask], reference=dtrain)
    booster = lgb.train(dict(num_leaves=127, learning_rate=0.03, min_data_in_leaf=200,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5, lambda_l2=1.0,
        objective='binary', metric='auc', verbose=-1, num_threads=NTHREADS, seed=42),
        dtrain, num_boost_round=5000, valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)])
    oof_lgb[va_mask] = booster.predict(X_synth[va_mask], num_iteration=booster.best_iteration)
    test_lgb += booster.predict(X_te, num_iteration=booster.best_iteration) / 5.0
    del booster, dtrain, dvalid; gc.collect()
print(f'[LGB] OOF AUC = {roc_auc_score(y_synth, oof_lgb):.6f}')

# ===== MODEL 2: XGB on GPU =====
print('\n[XGB] training on GPU...')
oof_xgb = np.zeros(len(y_synth), dtype=np.float64)
test_xgb = np.zeros(len(X_te), dtype=np.float64)
for f in range(5):
    tr_mask = folds != f; va_mask = folds == f
    X_fold_tr = np.vstack([X_synth[tr_mask], X_orig])
    y_fold_tr = np.concatenate([y_synth[tr_mask], y_orig])
    dtr = xgb.DMatrix(X_fold_tr, y_fold_tr)
    dva = xgb.DMatrix(X_synth[va_mask], y_synth[va_mask])
    dte = xgb.DMatrix(X_te)
    booster = xgb.train(dict(max_depth=8, eta=0.03, subsample=0.8, colsample_bytree=0.7,
        reg_lambda=1.0, reg_alpha=0.1, objective='binary:logistic', eval_metric='auc',
        tree_method='hist', device='cuda', nthread=NTHREADS, seed=2026, verbosity=0),
        dtr, num_boost_round=5000, evals=[(dva,'va')], early_stopping_rounds=150, verbose_eval=False)
    oof_xgb[va_mask] = booster.predict(dva)
    test_xgb += booster.predict(dte) / 5.0
    del booster, dtr, dva, dte; gc.collect()
print(f'[XGB] OOF AUC = {roc_auc_score(y_synth, oof_xgb):.6f}')

# ===== MODEL 3: HGBC (HistGradientBoosting) =====
print('\n[HGBC] training...')
oof_hgb = np.zeros(len(y_synth), dtype=np.float64)
test_hgb = np.zeros(len(X_te), dtype=np.float64)
for f in range(5):
    tr_mask = folds != f; va_mask = folds == f
    X_fold_tr = np.vstack([X_synth[tr_mask], X_orig])
    y_fold_tr = np.concatenate([y_synth[tr_mask], y_orig])
    hgb = HistGradientBoostingClassifier(max_iter=2000, learning_rate=0.03, max_leaf_nodes=127,
        min_samples_leaf=200, l2_regularization=1.0, max_bins=255, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=100, random_state=99)
    hgb.fit(X_fold_tr, y_fold_tr)
    oof_hgb[va_mask] = hgb.predict_proba(X_synth[va_mask])[:, 1]
    test_hgb += hgb.predict_proba(X_te)[:, 1] / 5.0
    del hgb; gc.collect()
print(f'[HGBC] OOF AUC = {roc_auc_score(y_synth, oof_hgb):.6f}')

# ===== MODEL 4: RealMLP =====
print('\n[RealMLP] training...')
oof_mlp = np.zeros(len(y_synth), dtype=np.float64)
test_mlp = np.zeros(len(X_te), dtype=np.float64)
try:
    for f in range(5):
        tr_mask = folds != f; va_mask = folds == f
        X_fold_tr = np.vstack([X_synth[tr_mask], X_orig])
        y_fold_tr = np.concatenate([y_synth[tr_mask], y_orig])
        mlp = RealMLP_TD_Classifier(n_epochs=100, device='cuda', verbosity=0)
        mlp.fit(X_fold_tr, y_fold_tr)
        oof_mlp[va_mask] = mlp.predict_proba(X_synth[va_mask])[:, 1]
        test_mlp += mlp.predict_proba(X_te)[:, 1] / 5.0
        del mlp; gc.collect()
    print(f'[RealMLP] OOF AUC = {roc_auc_score(y_synth, oof_mlp):.6f}')
    has_mlp = True
except Exception as e:
    print(f'[RealMLP] FAILED: {e}')
    has_mlp = False

# ===== BLEND our models =====
print('\n=== OUR MODEL BLEND ===')
names = ['lgb', 'xgb', 'hgbc']
oofs = [oof_lgb, oof_xgb, oof_hgb]
tests = [test_lgb, test_xgb, test_hgb]
if has_mlp:
    names.append('realmlp'); oofs.append(oof_mlp); tests.append(test_mlp)

for n, o in zip(names, oofs):
    print(f'  {n}: {roc_auc_score(y_synth, o):.6f}')

oof_mat = np.stack(oofs, axis=1).astype(np.float64)
test_mat = np.stack(tests, axis=1).astype(np.float64)

def neg_auc(w, X, y):
    w = np.abs(w); w /= max(w.sum(), 1e-9)
    return -roc_auc_score(y, X @ w)

w0 = np.ones(len(names))/len(names)
res = minimize(neg_auc, x0=w0, args=(oof_mat, y_synth), method='Nelder-Mead',
               options={'xatol':1e-8,'fatol':1e-9,'maxiter':20000})
w = np.abs(res.x); w /= w.sum()
our_blend = test_mat @ w
our_oof = oof_mat @ w
print(f'Our blend weights: {dict(zip(names, w.round(4)))}')
print(f'Our blend OOF AUC = {roc_auc_score(y_synth, our_oof):.6f}')

# ===== LOAD PUBLIC SUBMISSIONS =====
print('\n=== LOADING PUBLIC SUBMISSIONS ===')
pub_preds = {}
for root, dirs, files in os.walk(f'{WD}/pub'):
    for fn in files:
        if fn.endswith('.csv') and fn != 'submission.csv':
            continue
        if fn == 'submission.csv':
            path = os.path.join(root, fn)
            df = pd.read_csv(path)
            if 'PitNextLap' in df.columns and len(df) == len(X_te):
                label = os.path.basename(os.path.dirname(path))
                pub_preds[label] = df['PitNextLap'].values
                print(f'  loaded {label}: mean={df["PitNextLap"].mean():.4f}')

# ===== FINAL MEGA-BLEND (our + public) =====
print('\n=== MEGA BLEND ===')
all_test_preds = {'ours': our_blend}
all_test_preds.update(pub_preds)

if len(all_test_preds) > 1:
    # Rank-average all available predictions
    def rank_avg(preds_dict):
        arrays = list(preds_dict.values())
        ranks = [pd.Series(a).rank(pct=True).values for a in arrays]
        return np.mean(ranks, axis=0)

    mega_rank = rank_avg(all_test_preds)
    print(f'  Mega rank-avg of {len(all_test_preds)} submissions')
    # Also try weighted toward ours
    best_test = mega_rank
else:
    best_test = our_blend
    print('  No public subs found, using our blend only')

# Save submissions
pd.DataFrame({'id': test_ids, 'PitNextLap': our_blend}).to_csv(f'{WD}/submission_v8_ours.csv', index=False)
pd.DataFrame({'id': test_ids, 'PitNextLap': best_test}).to_csv(f'{WD}/submission_v8_mega.csv', index=False)
print(f'\nRuntime: {(time.time()-t0)/60:.1f} min')
print('DONE')
