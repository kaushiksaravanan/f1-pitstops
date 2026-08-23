"""
F1 Pit Stops — End-to-end blended GBM ensemble.
Runs on Kaggle Notebook (CPU). Single process, sequential model training.
Outputs /kaggle/working/submission.csv
"""
import os, gc, time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize

t0 = time.time()
OUT = '/kaggle/working'
os.makedirs(OUT, exist_ok=True)

NTHREADS = 4  # kaggle CPU has 4 vcpus

# --- robust input discovery ---
def find_input():
    base = '/kaggle/input'
    print(f'=== probing {base} ===')
    if not os.path.isdir(base):
        raise SystemExit(f'No {base} directory!')
    # Walk to find train.csv with PitNextLap column anywhere
    for root, dirs, files in os.walk(base):
        if 'train.csv' in files:
            cand = os.path.join(root, 'train.csv')
            try:
                head = pd.read_csv(cand, nrows=1)
                if 'PitNextLap' in head.columns:
                    print(f'  --> using {root} as competition root')
                    return root
            except Exception as e:
                print(f'  skipped {cand}: {e}')
    raise SystemExit('Could not find competition train.csv with PitNextLap column')

INP = find_input()

print('=== loading raw ===')
tr = pd.read_csv(f'{INP}/train.csv')
te = pd.read_csv(f'{INP}/test.csv')
print(f'train={tr.shape}  test={te.shape}')

# ---------------- FEATURE ENGINEERING ----------------
def build_features(tr, te):
    n_tr = len(tr)
    te = te.copy()
    te['PitNextLap'] = np.nan
    df = pd.concat([tr.assign(_src='train'), te.assign(_src='test')], ignore_index=True)
    df['LapTime'] = df['LapTime (s)']

    # Stint reconstruction (the dropped Normalized_TyreLife)
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

    # (Driver, Race, Year)
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

    # (Race, Year)
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

    # Compound x Race
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

    # Target encoding (train-only, with smoothing)
    train_mask = (df['_src'] == 'train').values
    y_tr = df.loc[train_mask, 'PitNextLap'].values
    global_mean = float(np.nanmean(y_tr))
    SMOOTH = 50

    for col in ['Race', 'Driver', 'Compound']:
        agg = df.loc[train_mask, [col, 'PitNextLap']].groupby(col)['PitNextLap'].agg(['mean', 'count']).reset_index()
        agg[f'{col}_TE'] = (agg['mean'] * agg['count'] + global_mean * SMOOTH) / (agg['count'] + SMOOTH)
        df = df.merge(agg[[col, f'{col}_TE']], on=col, how='left')
        df[f'{col}_TE'] = df[f'{col}_TE'].fillna(global_mean)

    # Categorical codes
    for c in ['Driver', 'Race', 'Compound']:
        df[f'{c}_code'] = df[c].astype('category').cat.codes

    # Stint position rank
    df['stint_tyre_rank'] = df.groupby(gkey)['TyreLife'].rank(method='dense')
    df['stint_tyre_pct_rank'] = df.groupby(gkey)['TyreLife'].rank(pct=True)
    df['stint_lap_rank'] = df.groupby(gkey)['LapNumber'].rank(method='dense')
    df['stint_lap_pct_rank'] = df.groupby(gkey)['LapNumber'].rank(pct=True)

    # Lag features within (D,R,Y) ordered by LapNumber
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
train_df, test_df = build_features(tr, te)
print(f'  train_df={train_df.shape}  test_df={test_df.shape}')

y = train_df['PitNextLap'].astype(int).values
test_ids = test_df['id'].values

drop_cols = ['id', 'PitNextLap', 'Driver', 'Race', 'Compound']
feat_cols = [c for c in train_df.columns if c not in drop_cols]
print(f'  features used: {len(feat_cols)}')

X_tr = train_df[feat_cols].astype(np.float32).values
X_te = test_df[feat_cols].astype(np.float32).values
del train_df, test_df, tr, te; gc.collect()
print(f'  X_tr={X_tr.shape}  X_te={X_te.shape}  RAM-ish freed')

# Folds
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = np.full(len(y), -1, dtype=np.int8)
for f, (_, vidx) in enumerate(skf.split(np.zeros(len(y)), y)):
    folds[vidx] = f


# ---------------- MODELS ----------------
def run_lgb(params, name, X_tr, y, X_te, folds, seed=42):
    import lightgbm as lgb
    oof = np.zeros(len(y), dtype=np.float64)
    test_pred = np.zeros(len(X_te), dtype=np.float64)
    p = dict(params); p.setdefault('objective','binary'); p.setdefault('metric','auc')
    p.setdefault('verbose',-1); p.setdefault('num_threads', NTHREADS); p.setdefault('seed', seed)
    print(f'\n[{name}] LightGBM 5-fold...')
    for f in range(5):
        tr_mask = folds != f; va_mask = folds == f
        dtrain = lgb.Dataset(X_tr[tr_mask], y[tr_mask])
        dvalid = lgb.Dataset(X_tr[va_mask], y[va_mask], reference=dtrain)
        booster = lgb.train(p, dtrain,
                            num_boost_round=p.get('num_boost_round', 5000),
                            valid_sets=[dvalid],
                            callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)])
        oof[va_mask] = booster.predict(X_tr[va_mask], num_iteration=booster.best_iteration)
        test_pred += booster.predict(X_te, num_iteration=booster.best_iteration) / 5.0
        del booster, dtrain, dvalid; gc.collect()
    auc = roc_auc_score(y, oof)
    print(f'[{name}] OOF AUC = {auc:.6f}')
    return oof.astype(np.float32), test_pred.astype(np.float32), auc


def run_xgb(params, name, X_tr, y, X_te, folds, seed=2026):
    import xgboost as xgb
    oof = np.zeros(len(y), dtype=np.float64)
    test_pred = np.zeros(len(X_te), dtype=np.float64)
    p = dict(params); p.setdefault('objective','binary:logistic'); p.setdefault('eval_metric','auc')
    p.setdefault('tree_method','hist'); p.setdefault('nthread', NTHREADS); p.setdefault('seed', seed)
    p.setdefault('verbosity', 0)
    print(f'\n[{name}] XGBoost 5-fold...')
    for f in range(5):
        tr_mask = folds != f; va_mask = folds == f
        dtr = xgb.DMatrix(X_tr[tr_mask], y[tr_mask])
        dva = xgb.DMatrix(X_tr[va_mask], y[va_mask])
        dte = xgb.DMatrix(X_te)
        booster = xgb.train(p, dtr,
                            num_boost_round=p.get('num_boost_round', 5000),
                            evals=[(dva,'va')],
                            early_stopping_rounds=150,
                            verbose_eval=False)
        oof[va_mask] = booster.predict(dva)
        test_pred += booster.predict(dte) / 5.0
        del booster, dtr, dva, dte; gc.collect()
    auc = roc_auc_score(y, oof)
    print(f'[{name}] OOF AUC = {auc:.6f}')
    return oof.astype(np.float32), test_pred.astype(np.float32), auc


def run_cat(params, name, X_tr, y, X_te, folds, seed=7):
    from catboost import CatBoostClassifier, Pool
    oof = np.zeros(len(y), dtype=np.float64)
    test_pred = np.zeros(len(X_te), dtype=np.float64)
    print(f'\n[{name}] CatBoost 5-fold...')
    p = dict(params); p.setdefault('thread_count', NTHREADS); p.setdefault('random_seed', seed)
    p.setdefault('eval_metric', 'AUC'); p.setdefault('verbose', 0)
    for f in range(5):
        tr_mask = folds != f; va_mask = folds == f
        m = CatBoostClassifier(**p)
        m.fit(X_tr[tr_mask], y[tr_mask], eval_set=(X_tr[va_mask], y[va_mask]),
              early_stopping_rounds=200, verbose=0)
        oof[va_mask] = m.predict_proba(X_tr[va_mask])[:, 1]
        test_pred += m.predict_proba(X_te)[:, 1] / 5.0
        del m; gc.collect()
    auc = roc_auc_score(y, oof)
    print(f'[{name}] OOF AUC = {auc:.6f}')
    return oof.astype(np.float32), test_pred.astype(np.float32), auc


# ---- Train each model ----
lgb_params = dict(num_leaves=127, learning_rate=0.03, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
                  lambda_l2=1.0, num_boost_round=5000)
oof_lgb, test_lgb, auc_lgb = run_lgb(lgb_params, 'LGB', X_tr, y, X_te, folds, seed=42)
np.save(f'{OUT}/oof_lgb.npy', oof_lgb)
np.save(f'{OUT}/test_lgb.npy', test_lgb)

xgb_params = dict(max_depth=8, eta=0.03, subsample=0.8, colsample_bytree=0.7,
                  reg_lambda=1.0, reg_alpha=0.1, num_boost_round=5000)
oof_xgb, test_xgb, auc_xgb = run_xgb(xgb_params, 'XGB', X_tr, y, X_te, folds, seed=2026)
np.save(f'{OUT}/oof_xgb.npy', oof_xgb)
np.save(f'{OUT}/test_xgb.npy', test_xgb)

cat_params = dict(depth=8, learning_rate=0.03, iterations=4000, l2_leaf_reg=3,
                  bootstrap_type='Bernoulli', subsample=0.8)
oof_cat, test_cat, auc_cat = run_cat(cat_params, 'CAT', X_tr, y, X_te, folds, seed=7)
np.save(f'{OUT}/oof_cat.npy', oof_cat)
np.save(f'{OUT}/test_cat.npy', test_cat)

# DART for diversity (different residual structure)
dart_params = dict(boosting='dart', num_leaves=63, learning_rate=0.05, min_data_in_leaf=200,
                   feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=5, lambda_l2=1.0,
                   num_boost_round=2500, drop_rate=0.1, skip_drop=0.5, max_drop=50,
                   uniform_drop=False)
# DART has no early stopping - run fixed
def run_dart(params, name, X_tr, y, X_te, folds, seed=11):
    import lightgbm as lgb
    oof = np.zeros(len(y), dtype=np.float64)
    test_pred = np.zeros(len(X_te), dtype=np.float64)
    p = dict(params); p.setdefault('objective','binary'); p.setdefault('metric','auc')
    p.setdefault('verbose',-1); p.setdefault('num_threads', NTHREADS); p.setdefault('seed', seed)
    print(f'\n[{name}] LightGBM-DART 5-fold...')
    nbr = p.pop('num_boost_round', 2500)
    for f in range(5):
        tr_mask = folds != f; va_mask = folds == f
        dtrain = lgb.Dataset(X_tr[tr_mask], y[tr_mask])
        dvalid = lgb.Dataset(X_tr[va_mask], y[va_mask], reference=dtrain)
        booster = lgb.train(p, dtrain, num_boost_round=nbr,
                            valid_sets=[dvalid], callbacks=[lgb.log_evaluation(0)])
        oof[va_mask] = booster.predict(X_tr[va_mask])
        test_pred += booster.predict(X_te) / 5.0
        del booster, dtrain, dvalid; gc.collect()
    auc = roc_auc_score(y, oof)
    print(f'[{name}] OOF AUC = {auc:.6f}')
    return oof.astype(np.float32), test_pred.astype(np.float32), auc

oof_dart, test_dart, auc_dart = run_dart(dart_params, 'DART', X_tr, y, X_te, folds, seed=11)
np.save(f'{OUT}/oof_dart.npy', oof_dart)
np.save(f'{OUT}/test_dart.npy', test_dart)


# ---------------- BLEND ----------------
print('\n=== blend ===')
oofs = [oof_lgb, oof_xgb, oof_cat, oof_dart]
tests = [test_lgb, test_xgb, test_cat, test_dart]
names = ['lgb', 'xgb', 'cat', 'dart']
oof_mat = np.stack(oofs, axis=1).astype(np.float64)
test_mat = np.stack(tests, axis=1).astype(np.float64)

for n, o in zip(names, oofs):
    print(f'  {n}: OOF AUC = {roc_auc_score(y, o):.6f}')

# Pearson correlations
print('\nOOF correlations:')
print('       ' + ' '.join(f'{n:>8s}' for n in names))
corr = np.corrcoef(oof_mat.T)
for i, n in enumerate(names):
    print(f'{n:>4s}   ' + ' '.join(f'{corr[i,j]:8.4f}' for j in range(len(names))))

# Rank-average
def rankavg(M):
    return np.mean(np.stack([pd.Series(M[:, i]).rank(pct=True).values for i in range(M.shape[1])], axis=1), axis=1)

avg_oof = oof_mat.mean(axis=1)
avg_test = test_mat.mean(axis=1)
rk_oof = rankavg(oof_mat)
rk_test = rankavg(test_mat)
print(f'\nsimple avg OOF AUC = {roc_auc_score(y, avg_oof):.6f}')
print(f'rank avg   OOF AUC = {roc_auc_score(y, rk_oof):.6f}')

# Optimised weighted blend (Nelder-Mead)
def neg_auc(w, X, y):
    w = np.abs(w); w /= max(w.sum(), 1e-9)
    return -roc_auc_score(y, X @ w)

w0 = np.ones(len(names))/len(names)
res = minimize(neg_auc, x0=w0, args=(oof_mat, y), method='Nelder-Mead',
               options={'xatol':1e-7,'fatol':1e-8,'maxiter':10000})
w_opt = np.abs(res.x); w_opt /= w_opt.sum()
print(f'\nweighted blend weights: {dict(zip(names, w_opt.round(4)))}')
w_oof = oof_mat @ w_opt
w_test = test_mat @ w_opt
print(f'weighted blend OOF AUC = {roc_auc_score(y, w_oof):.6f}')

# Same on ranks
oof_rank = np.stack([pd.Series(oof_mat[:, i]).rank(pct=True).values for i in range(len(names))], axis=1)
test_rank = np.stack([pd.Series(test_mat[:, i]).rank(pct=True).values for i in range(len(names))], axis=1)
res2 = minimize(neg_auc, x0=w0, args=(oof_rank, y), method='Nelder-Mead',
                options={'xatol':1e-7,'fatol':1e-8,'maxiter':10000})
w2 = np.abs(res2.x); w2 /= w2.sum()
print(f'weighted-rank weights:  {dict(zip(names, w2.round(4)))}')
wr_oof = oof_rank @ w2
wr_test = test_rank @ w2
print(f'weighted-rank OOF AUC = {roc_auc_score(y, wr_oof):.6f}')

# LR meta-stack
stack_oof = np.zeros(len(y))
for f in range(5):
    tr_mask = folds != f; va_mask = folds == f
    lr = LogisticRegression(C=1.0, max_iter=2000)
    lr.fit(oof_mat[tr_mask], y[tr_mask])
    stack_oof[va_mask] = lr.predict_proba(oof_mat[va_mask])[:, 1]
lr_full = LogisticRegression(C=1.0, max_iter=2000)
lr_full.fit(oof_mat, y)
stack_test = lr_full.predict_proba(test_mat)[:, 1]
print(f'LR-stack       OOF AUC = {roc_auc_score(y, stack_oof):.6f}')

# Pick best
candidates = {
    'simple_avg': (roc_auc_score(y, avg_oof), avg_test),
    'rank_avg': (roc_auc_score(y, rk_oof), rk_test),
    'weighted_mean': (roc_auc_score(y, w_oof), w_test),
    'weighted_rank': (roc_auc_score(y, wr_oof), wr_test),
    'lr_stack': (roc_auc_score(y, stack_oof), stack_test),
}
best = max(candidates, key=lambda k: candidates[k][0])
best_auc, best_test = candidates[best]
print(f'\n>>> BEST = {best}  OOF AUC = {best_auc:.6f}')

# Save submissions
for k, (a, t) in candidates.items():
    pd.DataFrame({'id': test_ids, 'PitNextLap': t}).to_csv(f'{OUT}/sub_{k}.csv', index=False)
pd.DataFrame({'id': test_ids, 'PitNextLap': best_test}).to_csv(f'{OUT}/submission.csv', index=False)

print(f'\nTotal runtime: {(time.time()-t0)/60:.1f} min')
print('Wrote:', os.listdir(OUT))
