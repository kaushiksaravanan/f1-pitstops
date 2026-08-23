"""
F1 Pit Stops v4 — adds external aadigupta1601 dataset join to recover
true Normalized_TyreLife and any other latent ground-truth columns.
Trains LGB + XGB on extended features and writes submission.
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

print('=== probing /kaggle/input ===')
def find_comp():
    for root, dirs, files in os.walk('/kaggle/input'):
        if 'train.csv' in files:
            try:
                head = pd.read_csv(os.path.join(root, 'train.csv'), nrows=1)
                if 'PitNextLap' in head.columns:
                    return root
            except Exception:
                pass
    raise SystemExit('comp data not found')

def find_external():
    cands = []
    for root, dirs, files in os.walk('/kaggle/input'):
        for fn in files:
            if fn.endswith('.csv') or fn.endswith('.parquet'):
                cands.append(os.path.join(root, fn))
    return cands

INP = find_comp()
print(f'comp INP={INP}')
ext_files = [f for f in find_external() if 'aadigupta' in f.lower() or 'strategy' in f.lower() or 'f1_strategy' in f.lower()]
print(f'external candidates: {ext_files[:10]}')

tr = pd.read_csv(f'{INP}/train.csv')
te = pd.read_csv(f'{INP}/test.csv')
print(f'train={tr.shape}  test={te.shape}')

# Try to load external dataset
ext = None
for f in ext_files:
    try:
        if f.endswith('.parquet'):
            cand = pd.read_parquet(f)
        else:
            cand = pd.read_csv(f)
        if 'TyreLife' in cand.columns and 'Driver' in cand.columns:
            ext = cand
            print(f'loaded external: {f}  shape={ext.shape}')
            print(f'cols: {list(ext.columns)}')
            break
    except Exception as e:
        print(f'skip {f}: {e}')

# Build base feature set
def build_features(tr, te, ext=None):
    n_tr = len(tr)
    te = te.copy(); te['PitNextLap'] = np.nan
    df = pd.concat([tr.assign(_src='train'), te.assign(_src='test')], ignore_index=True)
    df['LapTime'] = df['LapTime (s)']

    # External join on (Driver, Race, Year, LapNumber, TyreLife, Compound)
    if ext is not None:
        ext_keep_cols = []
        for cand in ['Normalized_TyreLife', 'normalized_tyrelife', 'NormTyreLife',
                     'TrueStint', 'StintTrue', 'Stint_true']:
            if cand in ext.columns:
                ext_keep_cols.append(cand)
        # Always keep Stint as ext_Stint to compare
        if 'Stint' in ext.columns:
            ext_keep_cols.append('Stint')
        join_keys = [k for k in ['Driver', 'Race', 'Year', 'LapNumber', 'TyreLife', 'Compound']
                     if k in ext.columns and k in df.columns]
        print(f'join keys: {join_keys}  ext cols to bring: {ext_keep_cols}')
        if join_keys and ext_keep_cols:
            ext_sub = ext[join_keys + ext_keep_cols].copy()
            ext_sub = ext_sub.drop_duplicates(subset=join_keys, keep='first')
            ext_sub = ext_sub.rename(columns={c: f'ext_{c}' for c in ext_keep_cols})
            df = df.merge(ext_sub, on=join_keys, how='left')
            for c in ext_keep_cols:
                col = f'ext_{c}'
                cov = df[col].notna().mean()
                print(f'  {col}: coverage = {cov:.3%}')

    # Stint reconstruction
    gkey = ['Driver', 'Race', 'Year', 'Stint']
    g = df.groupby(gkey).agg(
        stint_max_tyre=('TyreLife', 'max'),
        stint_min_tyre=('TyreLife', 'min'),
        stint_n_obs=('TyreLife', 'count'),
        stint_max_lap=('LapNumber', 'max'),
        stint_min_lap=('LapNumber', 'min'),
        stint_pitstop_sum=('PitStop', 'sum'),
    ).reset_index()
    df = df.merge(g, on=gkey, how='left')
    df['Norm_TyreLife'] = df['TyreLife'] / df['stint_max_tyre']
    df['is_max_in_stint'] = (df['TyreLife'] == df['stint_max_tyre']).astype(np.int8)
    df['laps_left_in_stint'] = df['stint_max_tyre'] - df['TyreLife']
    df['stint_observed_span'] = df['stint_max_lap'] - df['stint_min_lap'] + 1
    df['stint_lap_offset'] = df['LapNumber'] - df['stint_min_lap']
    df['stint_lap_progress'] = df['stint_lap_offset'] / df['stint_observed_span'].clip(lower=1)

    gkey2 = ['Driver', 'Race', 'Year']
    g2 = df.groupby(gkey2).agg(
        dr_total_laps=('LapNumber', 'count'),
        dr_max_lap=('LapNumber', 'max'),
        dr_max_stint=('Stint', 'max'),
        dr_unique_stints=('Stint', 'nunique'),
        dr_max_tyre=('TyreLife', 'max'),
        dr_pitstop_total=('PitStop', 'sum'),
        dr_mean_lap=('LapTime', 'mean'),
        dr_min_lap=('LapTime', 'min'),
        dr_std_lap=('LapTime', 'std'),
    ).reset_index()
    df = df.merge(g2, on=gkey2, how='left')
    df['lap_share_in_drrace'] = df['LapNumber'] / df['dr_max_lap'].clip(lower=1)

    gkey3 = ['Race', 'Year']
    g3 = df.groupby(gkey3).agg(
        race_total_laps=('LapNumber', 'count'),
        race_max_lap=('LapNumber', 'max'),
        race_drivers=('Driver', 'nunique'),
        race_mean_lap=('LapTime', 'mean'),
        race_min_lap=('LapTime', 'min'),
        race_pitstop_rate=('PitStop', 'mean'),
    ).reset_index()
    df = df.merge(g3, on=gkey3, how='left')

    gkey4 = ['Race', 'Year', 'Compound']
    g4 = df.groupby(gkey4).agg(
        cr_count=('Compound', 'count'),
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

    df['stint_tyre_rank'] = df.groupby(gkey)['TyreLife'].rank(method='dense')
    df['stint_tyre_pct_rank'] = df.groupby(gkey)['TyreLife'].rank(pct=True)
    df['stint_lap_rank'] = df.groupby(gkey)['LapNumber'].rank(method='dense')
    df['stint_lap_pct_rank'] = df.groupby(gkey)['LapNumber'].rank(pct=True)

    df = df.sort_values(['Driver', 'Race', 'Year', 'LapNumber']).reset_index(drop=True)
    g_dry = df.groupby(['Driver', 'Race', 'Year'])
    df['LapTime_prev'] = g_dry['LapTime'].shift(1)
    df['LapTime_next'] = g_dry['LapTime'].shift(-1)
    df['LapTime_diff_prev'] = df['LapTime'] - df['LapTime_prev']
    df['LapTime_diff_next'] = df['LapTime_next'] - df['LapTime']
    df['Position_prev'] = g_dry['Position'].shift(1)
    df['Position_next'] = g_dry['Position'].shift(-1)
    df['Position_diff_prev'] = df['Position'] - df['Position_prev']
    df['Position_diff_next'] = df['Position_next'] - df['Position']
    df['Stint_prev'] = g_dry['Stint'].shift(1)
    df['Stint_next'] = g_dry['Stint'].shift(-1)
    df['Stint_changes_next'] = (df['Stint_next'] != df['Stint']).astype(np.int8)
    df['Stint_changed_prev'] = (df['Stint_prev'] != df['Stint']).astype(np.int8)
    df['TyreLife_prev'] = g_dry['TyreLife'].shift(1)
    df['TyreLife_next'] = g_dry['TyreLife'].shift(-1)
    df['TyreLife_diff_prev'] = df['TyreLife'] - df['TyreLife_prev']
    df['Compound_prev'] = g_dry['Compound_ord'].shift(1)
    df['Compound_changed_prev'] = (df['Compound_prev'] != df['Compound_ord']).astype(np.int8)
    df['LapNumber_prev'] = g_dry['LapNumber'].shift(1)
    df['LapNumber_gap'] = df['LapNumber'] - df['LapNumber_prev']

    df = df.sort_values('id').reset_index(drop=True)
    train_df = df[df['_src'] == 'train'].drop(columns=['_src']).reset_index(drop=True)
    test_df = df[df['_src'] == 'test'].drop(columns=['_src', 'PitNextLap']).reset_index(drop=True)
    return train_df, test_df

print('=== building features ===')
train_df, test_df = build_features(tr, te, ext)
print(f'  train_df={train_df.shape}  test_df={test_df.shape}')

y = train_df['PitNextLap'].astype(int).values
test_ids = test_df['id'].values
drop_cols = ['id', 'PitNextLap', 'Driver', 'Race', 'Compound']
feat_cols = [c for c in train_df.columns if c not in drop_cols]
print(f'  features: {len(feat_cols)}')

X_tr = train_df[feat_cols].astype(np.float32).values
X_te = test_df[feat_cols].astype(np.float32).values
del train_df, test_df, tr, te; gc.collect()

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
        booster = lgb.train(p, dtrain, num_boost_round=4000,
                            valid_sets=[dvalid],
                            callbacks=[lgb.early_stopping(120, verbose=False), lgb.log_evaluation(0)])
        oof[va_mask] = booster.predict(X_tr[va_mask], num_iteration=booster.best_iteration)
        test_pred += booster.predict(X_te, num_iteration=booster.best_iteration) / 5.0
        del booster, dtrain, dvalid; gc.collect()
    auc = roc_auc_score(y, oof)
    print(f'[{name}] OOF AUC = {auc:.6f}')
    return oof.astype(np.float32), test_pred.astype(np.float32), auc

def run_xgb(name, seed):
    import xgboost as xgb
    oof = np.zeros(len(y), dtype=np.float64)
    test_pred = np.zeros(len(X_te), dtype=np.float64)
    p = dict(max_depth=8, eta=0.04, subsample=0.8, colsample_bytree=0.7,
             reg_lambda=1.0, reg_alpha=0.1, tree_method='hist',
             objective='binary:logistic', eval_metric='auc', nthread=NTHREADS,
             seed=seed, verbosity=0)
    print(f'\n[{name}] XGB 5-fold...')
    for f in range(5):
        tr_mask = folds != f; va_mask = folds == f
        dtr = xgb.DMatrix(X_tr[tr_mask], y[tr_mask])
        dva = xgb.DMatrix(X_tr[va_mask], y[va_mask])
        dte = xgb.DMatrix(X_te)
        booster = xgb.train(p, dtr, num_boost_round=4000,
                            evals=[(dva,'va')], early_stopping_rounds=120, verbose_eval=False)
        oof[va_mask] = booster.predict(dva)
        test_pred += booster.predict(dte) / 5.0
        del booster, dtr, dva, dte; gc.collect()
    auc = roc_auc_score(y, oof)
    print(f'[{name}] OOF AUC = {auc:.6f}')
    return oof.astype(np.float32), test_pred.astype(np.float32), auc

oof_lgb, test_lgb, auc_lgb = run_lgb('LGB-v4',
    dict(num_leaves=255, learning_rate=0.04, min_data_in_leaf=150,
         feature_fraction=0.75, bagging_fraction=0.85, bagging_freq=5, lambda_l2=1.0), 42)

oof_xgb, test_xgb, auc_xgb = run_xgb('XGB-v4', 2026)

# Save individual + simple blend
np.save(f'{OUT}/oof_lgb_v4.npy', oof_lgb)
np.save(f'{OUT}/oof_xgb_v4.npy', oof_xgb)
np.save(f'{OUT}/test_lgb_v4.npy', test_lgb)
np.save(f'{OUT}/test_xgb_v4.npy', test_xgb)

avg_oof = (oof_lgb.astype(np.float64) + oof_xgb.astype(np.float64)) / 2.0
avg_test = (test_lgb.astype(np.float64) + test_xgb.astype(np.float64)) / 2.0
print(f'\n[v4 blend] OOF AUC = {roc_auc_score(y, avg_oof):.6f}')

pd.DataFrame({'id': test_ids, 'PitNextLap': avg_test}).to_csv(f'{OUT}/submission.csv', index=False)
print(f'wrote submission.csv  total runtime = {(time.time()-t0)/60:.1f} min')
