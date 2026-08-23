"""Blend OOF and test predictions from 4 adversarial agents.

Strategies:
1. Simple average
2. Rank average
3. Weighted blend (weights chosen to maximize OOF AUC via grid + scipy.optimize)
4. Logistic meta-stack (LR on OOF preds)

Pick the best by OOF AUC, generate submission file.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize
import os, sys

DATA = Path('.')

# Load truth
train = pd.read_csv(DATA / 'train.csv')
y = train['PitNextLap'].astype(int).values
test = pd.read_csv(DATA / 'test.csv')

models = ['lgbA', 'catB', 'xgbC', 'divD']
oofs, tests, names = {}, {}, []
for m in models:
    of = DATA / f'oof_{m}.npy'
    tf = DATA / f'test_{m}.npy'
    if of.exists() and tf.exists():
        oof = np.load(of).astype(np.float64)
        tst = np.load(tf).astype(np.float64)
        if oof.shape != y.shape:
            print(f'[blend] SKIP {m}: oof shape {oof.shape}')
            continue
        if tst.shape != (len(test),):
            print(f'[blend] SKIP {m}: test shape {tst.shape}')
            continue
        oofs[m] = oof
        tests[m] = tst
        names.append(m)
        auc = roc_auc_score(y, oof)
        print(f'[blend] {m:8s} OOF AUC = {auc:.6f}')
    else:
        print(f'[blend] {m:8s} MISSING ({of.exists()=}, {tf.exists()=})')

if not names:
    print('No model outputs available. Aborting.')
    sys.exit(1)

# Pairwise correlation of OOF preds
print('\n[blend] OOF correlations (Pearson):')
oof_mat = np.stack([oofs[n] for n in names], axis=1)
corr = np.corrcoef(oof_mat.T)
print('  ' + ' '.join(f'{n:>8s}' for n in names))
for i, n in enumerate(names):
    print(f'{n:>4s} ' + ' '.join(f'{corr[i,j]:8.4f}' for j in range(len(names))))

# --- 1. Simple average ---
avg_oof = np.mean(oof_mat, axis=1)
avg_test = np.mean(np.stack([tests[n] for n in names], axis=1), axis=1)
print(f'\n[blend] simple avg OOF AUC = {roc_auc_score(y, avg_oof):.6f}')

# --- 2. Rank average ---
def rankavg(arr):
    return np.mean(np.stack([pd.Series(arr[:, i]).rank(pct=True).values for i in range(arr.shape[1])], axis=1), axis=1)

rk_oof = rankavg(oof_mat)
rk_test = rankavg(np.stack([tests[n] for n in names], axis=1))
print(f'[blend] rank avg OOF AUC = {roc_auc_score(y, rk_oof):.6f}')

# --- 3. Weighted blend via scipy minimize (max AUC) ---
def neg_auc(w, X, y):
    w = np.abs(w)
    s = w.sum()
    if s < 1e-9:
        return 0.0
    w = w / s
    pred = X @ w
    return -roc_auc_score(y, pred)

res = minimize(neg_auc, x0=np.ones(len(names))/len(names), args=(oof_mat, y), method='Nelder-Mead', options={'xatol':1e-6,'fatol':1e-7,'maxiter':5000})
w_opt = np.abs(res.x); w_opt = w_opt / w_opt.sum()
print(f'\n[blend] weighted-mean weights: {dict(zip(names, w_opt.round(4)))}')
w_oof = oof_mat @ w_opt
test_mat = np.stack([tests[n] for n in names], axis=1)
w_test = test_mat @ w_opt
print(f'[blend] weighted-mean OOF AUC = {roc_auc_score(y, w_oof):.6f}')

# --- 3b. Weighted blend on RANKS ---
oof_rank = np.stack([pd.Series(oof_mat[:, i]).rank(pct=True).values for i in range(len(names))], axis=1)
test_rank = np.stack([pd.Series(test_mat[:, i]).rank(pct=True).values for i in range(len(names))], axis=1)
res2 = minimize(neg_auc, x0=np.ones(len(names))/len(names), args=(oof_rank, y), method='Nelder-Mead', options={'xatol':1e-6,'fatol':1e-7,'maxiter':5000})
w_opt2 = np.abs(res2.x); w_opt2 = w_opt2 / w_opt2.sum()
print(f'[blend] weighted-rank weights: {dict(zip(names, w_opt2.round(4)))}')
wr_oof = oof_rank @ w_opt2
wr_test = test_rank @ w_opt2
print(f'[blend] weighted-rank OOF AUC = {roc_auc_score(y, wr_oof):.6f}')

# --- 4. Logistic meta-stack (5-fold) ---
folds = np.load(DATA / 'folds.npy')
stack_oof = np.zeros(len(y))
stack_test = np.zeros(len(test))
for f in range(5):
    tr_mask = folds != f
    va_mask = folds == f
    lr = LogisticRegression(C=1.0, max_iter=2000)
    lr.fit(oof_mat[tr_mask], y[tr_mask])
    stack_oof[va_mask] = lr.predict_proba(oof_mat[va_mask])[:, 1]
# Final test-stack: fit on full
lr_full = LogisticRegression(C=1.0, max_iter=2000)
lr_full.fit(oof_mat, y)
stack_test = lr_full.predict_proba(test_mat)[:, 1]
print(f'[blend] LR-stack OOF AUC = {roc_auc_score(y, stack_oof):.6f}')
print(f'[blend] LR coefs = {dict(zip(names, lr_full.coef_[0].round(4)))}, intercept={lr_full.intercept_[0]:.4f}')

# Pick best
candidates = {
    'simple_avg': (roc_auc_score(y, avg_oof), avg_test),
    'rank_avg': (roc_auc_score(y, rk_oof), rk_test),
    'weighted_mean': (roc_auc_score(y, w_oof), w_test),
    'weighted_rank': (roc_auc_score(y, wr_oof), wr_test),
    'lr_stack': (roc_auc_score(y, stack_oof), stack_test),
}
best_name = max(candidates, key=lambda k: candidates[k][0])
best_auc, best_test = candidates[best_name]
print(f'\n[blend] BEST = {best_name} OOF AUC = {best_auc:.6f}')

# Build submissions: best blend + safety (rank average across all available)
sub_best = pd.DataFrame({'id': test['id'].values, 'PitNextLap': best_test})
sub_best.to_csv(DATA / f'submission_{best_name}.csv', index=False)
print(f'[blend] wrote submission_{best_name}.csv')

sub_rank = pd.DataFrame({'id': test['id'].values, 'PitNextLap': rk_test})
sub_rank.to_csv(DATA / 'submission_rank_avg.csv', index=False)

# Always write a "submission.csv" pointing at the best blend
sub_best.to_csv(DATA / 'submission.csv', index=False)
print(f'[blend] wrote submission.csv -> {best_name}')
