"""
F1 Pit Stops v5 — Pseudo-labeling + multi-seed diversity + feature interactions.
Strategy:
  1. Build strong features (same as v3 + interactions)
  2. Train a 'teacher' LGB → predict test probabilities
  3. Select high-confidence test rows (>0.9 or <0.1) as pseudo-labels
  4. Retrain with train + pseudo-labels
  5. Multi-seed average (3 seeds each for LGB + XGB)
  6. Blend all models via Nelder-Mead optimization
"""
import os, gc, time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize

t0 = time.time()
OUT = '/kaggle/working'
os.makedirs(OUT, exist_ok=True)
NTHREADS = 4

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

INP = find_comp()
print(f'INP={INP}')

tr = pd.read_csv(f'{INP}/train.csv')
te = pd.read_csv(f'{INP}/test.csv')
print(f'train={tr.shape}  test={te.shape}')

def build_features(tr, te):
    n_tr = len(tr)
    te = te.copy(); te['PitNextLap'] = np.nan
    df = pd.concat([tr.assign(_src='train'), te.assign(_src='test')], ignore_index=True)
    df['LapTime'] = df['LapTime (s)']

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

    # Target encoding
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

    # Lag features
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

    # --- NEW: Feature interactions ---
    df['TyreLife_x_Position'] = df['TyreLife'] * df['Position']
    df['TyreLife_x_LapTime_vs_dr'] = df['TyreLife'] * df['LapTime_vs_dr_mean']
    df['Norm_TyreLife_x_LapsToEnd'] = df['Norm_TyreLife'] * df['LapsToEnd']
    df['stint_progress_x_position'] = df['stint_lap_progress'] * df['Position']
    df['laps_left_x_compound'] = df['laps_left_in_stint'] * df['Compound_ord']
    df['TyreLife_x_stint_max'] = df['TyreLife'] * df['stint_max_tyre']
    df['Position_x_LapsToEnd'] = df['Position'] * df['LapsToEnd']
    df['LapTime_diff_x_TyreLife'] = df['LapTime_diff_prev'] * df['TyreLife']
    df['pitstop_rate_x_tyre_progress'] = df['race_pitstop_rate'] * df['Norm_TyreLife']
    df['dr_pitstop_total_x_stint'] = df['dr_pitstop_total'] * df['Stint']

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
print(f'  features: {len(feat_cols)}')
X_tr = train_df[feat_cols].astype(np.float32).values
X_te = test_df[feat_cols].astype(np.float32).values
del train_df, test_df, tr, te; gc.collect()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = np.full(len(y), -1, dtype=np.int8)
for f, (_, vidx) in enumerate(skf.split(np.zeros(len(y)), y)):
    folds[vidx] = f

# ====== PHASE 1: Teacher model for pseudo-labels ======
import lightgbm as lgb

print('\n=== PHASE 1: Teacher LGB for pseudo-labels ===')
teacher_params = dict(num_leaves=127, learning_rate=0.03, min_data_in_leaf=200,
                      feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
                      lambda_l2=1.0, objective='binary', metric='auc',
                      verbose=-1, num_threads=NTHREADS, seed=42)
teacher_test = np.zeros(len(X_te), dtype=np.float64)
for f in range(5):
    tr_mask = folds != f
    dtrain = lgb.Dataset(X_tr[tr_mask], y[tr_mask])
    booster = lgb.train(teacher_params, dtrain, num_boost_round=3000,
                        callbacks=[lgb.log_evaluation(0)])
    teacher_test += booster.predict(X_te) / 5.0
    del booster, dtrain; gc.collect()
print(f'  teacher test mean={teacher_test.mean():.4f}  min={teacher_test.min():.4f}  max={teacher_test.max():.4f}')

# Select pseudo-labels (high confidence)
THRESH_HIGH = 0.90
THRESH_LOW = 0.10
pseudo_pos = teacher_test >= THRESH_HIGH
pseudo_neg = teacher_test <= THRESH_LOW
n_pseudo = pseudo_pos.sum() + pseudo_neg.sum()
print(f'  pseudo-labels: {pseudo_pos.sum()} pos + {pseudo_neg.sum()} neg = {n_pseudo} ({n_pseudo/len(X_te)*100:.1f}% of test)')

pseudo_X = X_te[pseudo_pos | pseudo_neg]
pseudo_y = np.where(teacher_test[pseudo_pos | pseudo_neg] >= THRESH_HIGH, 1, 0)

# Augmented training set
X_aug = np.concatenate([X_tr, pseudo_X], axis=0)
y_aug = np.concatenate([y, pseudo_y], axis=0)
print(f'  augmented train: {X_aug.shape[0]} rows ({len(X_tr)} real + {len(pseudo_X)} pseudo)')

# Build folds for augmented data (only real data gets OOF evaluation)
folds_aug = np.full(len(y_aug), -1, dtype=np.int8)
folds_aug[:len(y)] = folds
folds_aug[len(y):] = -1  # pseudo-labels always in training

# ====== PHASE 2: Multi-seed LGB on augmented data ======
print('\n=== PHASE 2: Multi-seed LGB (3 seeds) ===')
lgb_seeds = [42, 2026, 777]
lgb_oofs = []
lgb_tests = []
lgb_base = dict(num_leaves=191, learning_rate=0.03, min_data_in_leaf=150,
                feature_fraction=0.75, bagging_fraction=0.85, bagging_freq=5,
                lambda_l2=1.5, objective='binary', metric='auc',
                verbose=-1, num_threads=NTHREADS)

for seed in lgb_seeds:
    p = dict(lgb_base); p['seed'] = seed
    oof = np.zeros(len(y), dtype=np.float64)
    test_pred = np.zeros(len(X_te), dtype=np.float64)
    for f in range(5):
        tr_mask_real = folds != f
        tr_mask_aug = np.concatenate([tr_mask_real, np.ones(len(pseudo_X), dtype=bool)])
        va_mask_real = folds == f
        dtrain = lgb.Dataset(X_aug[tr_mask_aug], y_aug[tr_mask_aug])
        dvalid = lgb.Dataset(X_tr[va_mask_real], y[va_mask_real], reference=dtrain)
        booster = lgb.train(p, dtrain, num_boost_round=4000,
                            valid_sets=[dvalid],
                            callbacks=[lgb.early_stopping(120, verbose=False), lgb.log_evaluation(0)])
        oof[va_mask_real] = booster.predict(X_tr[va_mask_real], num_iteration=booster.best_iteration)
        test_pred += booster.predict(X_te, num_iteration=booster.best_iteration) / 5.0
        del booster, dtrain, dvalid; gc.collect()
    auc = roc_auc_score(y, oof)
    print(f'  LGB seed={seed} OOF AUC = {auc:.6f}')
    lgb_oofs.append(oof.astype(np.float32))
    lgb_tests.append(test_pred.astype(np.float32))

# ====== PHASE 3: Multi-seed XGB on augmented data ======
print('\n=== PHASE 3: Multi-seed XGB (3 seeds) ===')
import xgboost as xgb
xgb_seeds = [2026, 1337, 99]
xgb_oofs = []
xgb_tests = []

for seed in xgb_seeds:
    p = dict(max_depth=8, eta=0.03, subsample=0.8, colsample_bytree=0.7,
             reg_lambda=1.0, reg_alpha=0.1, tree_method='hist',
             objective='binary:logistic', eval_metric='auc', nthread=NTHREADS,
             seed=seed, verbosity=0)
    oof = np.zeros(len(y), dtype=np.float64)
    test_pred = np.zeros(len(X_te), dtype=np.float64)
    for f in range(5):
        tr_mask_real = folds != f
        tr_mask_aug = np.concatenate([tr_mask_real, np.ones(len(pseudo_X), dtype=bool)])
        va_mask_real = folds == f
        dtr = xgb.DMatrix(X_aug[tr_mask_aug], y_aug[tr_mask_aug])
        dva = xgb.DMatrix(X_tr[va_mask_real], y[va_mask_real])
        dte = xgb.DMatrix(X_te)
        booster = xgb.train(p, dtr, num_boost_round=4000,
                            evals=[(dva,'va')], early_stopping_rounds=120, verbose_eval=False)
        oof[va_mask_real] = booster.predict(dva)
        test_pred += booster.predict(dte) / 5.0
        del booster, dtr, dva, dte; gc.collect()
    auc = roc_auc_score(y, oof)
    print(f'  XGB seed={seed} OOF AUC = {auc:.6f}')
    xgb_oofs.append(oof.astype(np.float32))
    xgb_tests.append(test_pred.astype(np.float32))

# ====== PHASE 4: DART for diversity ======
print('\n=== PHASE 4: DART (augmented) ===')
dart_params = dict(boosting='dart', num_leaves=63, learning_rate=0.05, min_data_in_leaf=200,
                   feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=5, lambda_l2=1.0,
                   drop_rate=0.1, skip_drop=0.5, max_drop=50,
                   objective='binary', metric='auc', verbose=-1, num_threads=NTHREADS, seed=11)
dart_oof = np.zeros(len(y), dtype=np.float64)
dart_test = np.zeros(len(X_te), dtype=np.float64)
for f in range(5):
    tr_mask_real = folds != f
    tr_mask_aug = np.concatenate([tr_mask_real, np.ones(len(pseudo_X), dtype=bool)])
    va_mask_real = folds == f
    dtrain = lgb.Dataset(X_aug[tr_mask_aug], y_aug[tr_mask_aug])
    dvalid = lgb.Dataset(X_tr[va_mask_real], y[va_mask_real], reference=dtrain)
    booster = lgb.train(dart_params, dtrain, num_boost_round=2000,
                        valid_sets=[dvalid], callbacks=[lgb.log_evaluation(0)])
    dart_oof[va_mask_real] = booster.predict(X_tr[va_mask_real])
    dart_test += booster.predict(X_te) / 5.0
    del booster, dtrain, dvalid; gc.collect()
print(f'  DART OOF AUC = {roc_auc_score(y, dart_oof):.6f}')

# ====== PHASE 5: Blend ======
print('\n=== PHASE 5: Optimized blend ===')
all_oofs = lgb_oofs + xgb_oofs + [dart_oof.astype(np.float32)]
all_tests = lgb_tests + xgb_tests + [dart_test.astype(np.float32)]
names = [f'lgb_s{s}' for s in lgb_seeds] + [f'xgb_s{s}' for s in xgb_seeds] + ['dart']

oof_mat = np.stack(all_oofs, axis=1).astype(np.float64)
test_mat = np.stack(all_tests, axis=1).astype(np.float64)

for n, o in zip(names, all_oofs):
    print(f'  {n}: {roc_auc_score(y, o):.6f}')

# Simple avg
avg_oof = oof_mat.mean(axis=1)
avg_test = test_mat.mean(axis=1)
print(f'\nsimple avg OOF AUC = {roc_auc_score(y, avg_oof):.6f}')

# Nelder-Mead weighted
def neg_auc(w, X, y):
    w = np.abs(w); w /= max(w.sum(), 1e-9)
    return -roc_auc_score(y, X @ w)

w0 = np.ones(len(names)) / len(names)
res = minimize(neg_auc, x0=w0, args=(oof_mat, y), method='Nelder-Mead',
               options={'xatol':1e-8, 'fatol':1e-9, 'maxiter':30000})
w = np.abs(res.x); w /= w.sum()
print(f'weighted blend weights: {dict(zip(names, w.round(4)))}')
w_oof = oof_mat @ w
w_test = test_mat @ w
print(f'weighted blend OOF AUC = {roc_auc_score(y, w_oof):.6f}')

# Rank-weighted
oof_rank = np.stack([pd.Series(oof_mat[:, i]).rank(pct=True).values for i in range(len(names))], axis=1)
test_rank = np.stack([pd.Series(test_mat[:, i]).rank(pct=True).values for i in range(len(names))], axis=1)
res2 = minimize(neg_auc, x0=w0, args=(oof_rank, y), method='Nelder-Mead',
                options={'xatol':1e-8, 'fatol':1e-9, 'maxiter':30000})
w2 = np.abs(res2.x); w2 /= w2.sum()
wr_oof = oof_rank @ w2
wr_test = test_rank @ w2
print(f'weighted-rank OOF AUC = {roc_auc_score(y, wr_oof):.6f}')

# Pick best
candidates = {
    'simple_avg': (roc_auc_score(y, avg_oof), avg_test),
    'weighted_mean': (roc_auc_score(y, w_oof), w_test),
    'weighted_rank': (roc_auc_score(y, wr_oof), wr_test),
}
best = max(candidates, key=lambda k: candidates[k][0])
best_auc, best_test = candidates[best]
print(f'\n>>> BEST = {best}  OOF AUC = {best_auc:.6f}')

pd.DataFrame({'id': test_ids, 'PitNextLap': best_test}).to_csv(f'{OUT}/submission.csv', index=False)
for k, (a, t) in candidates.items():
    pd.DataFrame({'id': test_ids, 'PitNextLap': t}).to_csv(f'{OUT}/sub_{k}.csv', index=False)

print(f'\nTotal runtime: {(time.time()-t0)/60:.1f} min')
print('Done.')
