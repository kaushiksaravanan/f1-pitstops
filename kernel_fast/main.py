"""
F1 Pit Stops — Fast safety-net submission.
Tighter iter budget so it finishes inside 25-30 min worst case.
"""
import os, gc, time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

t0 = time.time()
OUT = '/kaggle/working'
os.makedirs(OUT, exist_ok=True)
NTHREADS = 4

def find_input():
    base = '/kaggle/input'
    for root, dirs, files in os.walk(base):
        if 'train.csv' in files:
            cand = os.path.join(root, 'train.csv')
            try:
                head = pd.read_csv(cand, nrows=1)
                if 'PitNextLap' in head.columns:
                    return root
            except Exception:
                continue
    raise SystemExit('No input found')

INP = find_input()
print(f'INP={INP}')

tr = pd.read_csv(f'{INP}/train.csv')
te = pd.read_csv(f'{INP}/test.csv')
print(f'train={tr.shape}  test={te.shape}')

def build_features(tr, te):
    te = te.copy(); te['PitNextLap'] = np.nan
    df = pd.concat([tr.assign(_src='train'), te.assign(_src='test')], ignore_index=True)
    df['LapTime'] = df['LapTime (s)']
    gkey = ['Driver', 'Race', 'Year', 'Stint']
    g = df.groupby(gkey).agg(
        stint_max_tyre=('TyreLife', 'max'),
        stint_min_tyre=('TyreLife', 'min'),
        stint_max_lap=('LapNumber', 'max'),
        stint_min_lap=('LapNumber', 'min'),
    ).reset_index()
    df = df.merge(g, on=gkey, how='left')
    df['Norm_TyreLife'] = df['TyreLife'] / df['stint_max_tyre']
    df['laps_left_in_stint'] = df['stint_max_tyre'] - df['TyreLife']
    df['stint_observed_span'] = df['stint_max_lap'] - df['stint_min_lap'] + 1
    df['stint_lap_offset'] = df['LapNumber'] - df['stint_min_lap']
    df['stint_lap_progress'] = df['stint_lap_offset'] / df['stint_observed_span'].clip(lower=1)

    gkey2 = ['Driver', 'Race', 'Year']
    g2 = df.groupby(gkey2).agg(
        dr_max_lap=('LapNumber', 'max'),
        dr_max_stint=('Stint', 'max'),
        dr_max_tyre=('TyreLife', 'max'),
        dr_mean_lap=('LapTime', 'mean'),
        dr_min_lap=('LapTime', 'min'),
    ).reset_index()
    df = df.merge(g2, on=gkey2, how='left')
    df['lap_share_in_drrace'] = df['LapNumber'] / df['dr_max_lap'].clip(lower=1)

    gkey3 = ['Race', 'Year']
    g3 = df.groupby(gkey3).agg(
        race_max_lap=('LapNumber', 'max'),
        race_mean_lap=('LapTime', 'mean'),
        race_min_lap=('LapTime', 'min'),
        race_pitstop_rate=('PitStop', 'mean'),
    ).reset_index()
    df = df.merge(g3, on=gkey3, how='left')

    gkey4 = ['Race', 'Year', 'Compound']
    g4 = df.groupby(gkey4).agg(
        cr_max_tyre=('TyreLife', 'max'),
        cr_mean_tyre=('TyreLife', 'mean'),
        cr_pitstop_rate=('PitStop', 'mean'),
    ).reset_index()
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

    compound_order = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2, 'INTERMEDIATE': 3, 'WET': 4}
    df['Compound_ord'] = df['Compound'].map(compound_order).fillna(-1).astype(int)

    train_mask = (df['_src'] == 'train').values
    global_mean = float(np.nanmean(df.loc[train_mask, 'PitNextLap'].values))
    SMOOTH = 50
    for col in ['Race', 'Driver', 'Compound']:
        agg = df.loc[train_mask, [col, 'PitNextLap']].groupby(col)['PitNextLap'].agg(['mean', 'count']).reset_index()
        agg[f'{col}_TE'] = (agg['mean'] * agg['count'] + global_mean * SMOOTH) / (agg['count'] + SMOOTH)
        df = df.merge(agg[[col, f'{col}_TE']], on=col, how='left')
        df[f'{col}_TE'] = df[f'{col}_TE'].fillna(global_mean)

    for c in ['Driver', 'Race', 'Compound']:
        df[f'{c}_code'] = df[c].astype('category').cat.codes

    # lag features
    df = df.sort_values(['Driver', 'Race', 'Year', 'LapNumber']).reset_index(drop=True)
    g_dry = df.groupby(['Driver', 'Race', 'Year'])
    df['LapTime_prev'] = g_dry['LapTime'].shift(1)
    df['LapTime_diff_prev'] = df['LapTime'] - df['LapTime_prev']
    df['Position_prev'] = g_dry['Position'].shift(1)
    df['Position_diff_prev'] = df['Position'] - df['Position_prev']
    df['Stint_prev'] = g_dry['Stint'].shift(1)
    df['Stint_changed_prev'] = (df['Stint_prev'] != df['Stint']).astype(np.int8)
    df['TyreLife_prev'] = g_dry['TyreLife'].shift(1)
    df['TyreLife_diff_prev'] = df['TyreLife'] - df['TyreLife_prev']
    df['Compound_prev'] = g_dry['Compound_ord'].shift(1)
    df['Compound_changed_prev'] = (df['Compound_prev'] != df['Compound_ord']).astype(np.int8)

    df = df.sort_values('id').reset_index(drop=True)
    train_df = df[df['_src'] == 'train'].drop(columns=['_src']).reset_index(drop=True)
    test_df = df[df['_src'] == 'test'].drop(columns=['_src', 'PitNextLap']).reset_index(drop=True)
    return train_df, test_df

print('=== building features ===')
train_df, test_df = build_features(tr, te)
y = train_df['PitNextLap'].astype(int).values
test_ids = test_df['id'].values
drop_cols = ['id', 'PitNextLap', 'Driver', 'Race', 'Compound']
feat_cols = [c for c in train_df.columns if c not in drop_cols]
X_tr = train_df[feat_cols].astype(np.float32).values
X_te = test_df[feat_cols].astype(np.float32).values
del train_df, test_df, tr, te; gc.collect()
print(f'X_tr={X_tr.shape}  X_te={X_te.shape}  feat={len(feat_cols)}')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = np.full(len(y), -1, dtype=np.int8)
for f, (_, vidx) in enumerate(skf.split(np.zeros(len(y)), y)):
    folds[vidx] = f

def run_lgb(name, params, seed):
    import lightgbm as lgb
    oof = np.zeros(len(y), dtype=np.float64)
    test_pred = np.zeros(len(X_te), dtype=np.float64)
    p = dict(params); p.setdefault('objective','binary'); p.setdefault('metric','auc')
    p.setdefault('verbose',-1); p.setdefault('num_threads', NTHREADS); p.setdefault('seed', seed)
    print(f'\n[{name}] LGB 5-fold...')
    for f in range(5):
        tr_mask = folds != f; va_mask = folds == f
        dtrain = lgb.Dataset(X_tr[tr_mask], y[tr_mask])
        dvalid = lgb.Dataset(X_tr[va_mask], y[va_mask], reference=dtrain)
        booster = lgb.train(p, dtrain, num_boost_round=2000,
                            valid_sets=[dvalid],
                            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
        oof[va_mask] = booster.predict(X_tr[va_mask], num_iteration=booster.best_iteration)
        test_pred += booster.predict(X_te, num_iteration=booster.best_iteration) / 5.0
        del booster, dtrain, dvalid; gc.collect()
    print(f'[{name}] OOF AUC = {roc_auc_score(y, oof):.6f}')
    return oof.astype(np.float32), test_pred.astype(np.float32)

def run_xgb(name, seed):
    import xgboost as xgb
    oof = np.zeros(len(y), dtype=np.float64)
    test_pred = np.zeros(len(X_te), dtype=np.float64)
    p = dict(max_depth=7, eta=0.05, subsample=0.8, colsample_bytree=0.7,
             reg_lambda=1.0, reg_alpha=0.1, tree_method='hist',
             objective='binary:logistic', eval_metric='auc', nthread=NTHREADS,
             seed=seed, verbosity=0)
    print(f'\n[{name}] XGB 5-fold...')
    for f in range(5):
        tr_mask = folds != f; va_mask = folds == f
        dtr = xgb.DMatrix(X_tr[tr_mask], y[tr_mask])
        dva = xgb.DMatrix(X_tr[va_mask], y[va_mask])
        dte = xgb.DMatrix(X_te)
        booster = xgb.train(p, dtr, num_boost_round=2000,
                            evals=[(dva,'va')], early_stopping_rounds=80, verbose_eval=False)
        oof[va_mask] = booster.predict(dva)
        test_pred += booster.predict(dte) / 5.0
        del booster, dtr, dva, dte; gc.collect()
    print(f'[{name}] OOF AUC = {roc_auc_score(y, oof):.6f}')
    return oof.astype(np.float32), test_pred.astype(np.float32)

oof1, t1 = run_lgb('LGB', dict(num_leaves=127, learning_rate=0.05, min_data_in_leaf=200,
                                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5, lambda_l2=1.0), 42)
oof2, t2 = run_xgb('XGB', 2026)

oof_avg = (oof1 + oof2) / 2.0
test_avg = (t1 + t2) / 2.0
print(f'\nblend OOF AUC = {roc_auc_score(y, oof_avg):.6f}')
print(f'lgb:  {roc_auc_score(y, oof1):.6f}')
print(f'xgb:  {roc_auc_score(y, oof2):.6f}')

pd.DataFrame({'id': test_ids, 'PitNextLap': test_avg}).to_csv(f'{OUT}/submission.csv', index=False)
print(f'wrote submission.csv  total runtime = {(time.time()-t0)/60:.1f} min')
