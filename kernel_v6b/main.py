"""
F1 Pit Stops v6b — LSTM + 1D-CNN on GPU, fixed O(N) lookup bug.
Blends with GBM OOF from v3.
"""
import os, gc, time, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize

warnings.filterwarnings('ignore')
t0 = time.time()
OUT = '/kaggle/working'
os.makedirs(OUT, exist_ok=True)

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

te_copy = te.copy(); te_copy['PitNextLap'] = np.nan
df = pd.concat([tr.assign(_src='train'), te_copy.assign(_src='test')], ignore_index=True)
df['LapTime'] = df['LapTime (s)']

compound_order = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2, 'INTERMEDIATE': 3, 'WET': 4}
df['Compound_ord'] = df['Compound'].map(compound_order).fillna(-1).astype(int)

gkey = ['Driver', 'Race', 'Year', 'Stint']
g = df.groupby(gkey).agg(stint_max_tyre=('TyreLife', 'max')).reset_index()
df = df.merge(g, on=gkey, how='left')
df['Norm_TyreLife'] = df['TyreLife'] / df['stint_max_tyre'].clip(lower=1)

gkey3 = ['Race', 'Year']
g3 = df.groupby(gkey3).agg(race_max_lap=('LapNumber', 'max'), race_mean_lap=('LapTime', 'mean')).reset_index()
df = df.merge(g3, on=gkey3, how='left')
df['lap_progress'] = df['LapNumber'] / df['race_max_lap'].clip(lower=1)
df['LapTime_vs_mean'] = df['LapTime'] - df['race_mean_lap']

df = df.sort_values(['Driver', 'Race', 'Year', 'LapNumber']).reset_index(drop=True)

SEQ_COLS = ['LapNumber', 'Position', 'TyreLife', 'Stint', 'PitStop',
            'LapTime', 'Compound_ord', 'Norm_TyreLife', 'lap_progress', 'LapTime_vs_mean']
N_FEAT = len(SEQ_COLS)

# Pre-build id->target and id->fold lookups (FIX: O(1) instead of O(N))
print('=== building lookups ===')
train_mask_global = df['_src'] == 'train'
id_to_target = dict(zip(df.loc[train_mask_global, 'id'].values,
                         df.loc[train_mask_global, 'PitNextLap'].values))

y_full = df.loc[train_mask_global, 'PitNextLap'].astype(int).values
n_train = train_mask_global.sum()
n_test = (~train_mask_global & (df['_src'] == 'test')).sum()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_ids_arr = df.loc[train_mask_global, 'id'].values
id_to_fold = {}
for f, (_, vidx) in enumerate(skf.split(np.zeros(n_train), y_full)):
    for i in vidx:
        id_to_fold[train_ids_arr[i]] = f

# Build sequences
print('=== building sequences ===')
groups = df.groupby(['Driver', 'Race', 'Year'])
sequences = []
seq_meta = []  # list of (feat_array, id_array, src_array, target_array)
for (drv, race, yr), grp in groups:
    grp = grp.sort_values('LapNumber')
    feat_arr = grp[SEQ_COLS].values.astype(np.float32)
    ids = grp['id'].values.astype(np.int64)
    srcs = grp['_src'].values
    targets = np.array([id_to_target.get(i, 0.0) for i in ids], dtype=np.float32)
    sequences.append((feat_arr, ids, srcs, targets))

print(f'  total sequences: {len(sequences)}')
print(f'  seq lengths: min={min(len(s[0]) for s in sequences)} max={max(len(s[0]) for s in sequences)}')

# Align to original id order for final output
df_sorted = df.sort_values('id').reset_index(drop=True)
train_ids_ordered = df_sorted.loc[df_sorted['_src'] == 'train', 'id'].values
test_ids_ordered = df_sorted.loc[df_sorted['_src'] == 'test', 'id'].values

# ====== PYTORCH ======
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device={device}')


class SeqDataset(Dataset):
    def __init__(self, sequences, mode='train', fold=None):
        self.items = []
        for feat_arr, ids, srcs, targets in sequences:
            mask = np.zeros(len(ids), dtype=bool)
            for i in range(len(ids)):
                if srcs[i] == 'train':
                    f = id_to_fold.get(ids[i], -1)
                    if mode == 'train' and f != fold:
                        mask[i] = True
                    elif mode == 'val' and f == fold:
                        mask[i] = True
                elif srcs[i] == 'test' and mode == 'test':
                    mask[i] = True
            if mask.any():
                self.items.append((feat_arr, targets, mask, ids))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        feat, tgt, mask, ids = self.items[idx]
        return (torch.FloatTensor(feat), torch.FloatTensor(tgt),
                torch.BoolTensor(mask), torch.LongTensor(ids))


def collate_fn(batch):
    max_len = max(b[0].shape[0] for b in batch)
    B = len(batch)
    X = torch.zeros(B, max_len, N_FEAT)
    Y = torch.zeros(B, max_len)
    M = torch.zeros(B, max_len, dtype=torch.bool)
    lengths = []
    all_ids = []
    for i, (x, y, m, ids) in enumerate(batch):
        L = x.shape[0]
        X[i, :L] = x
        Y[i, :L] = y
        M[i, :L] = m
        lengths.append(L)
        all_ids.append(ids)
    return X, Y, M, lengths, all_ids


class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden=96, layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True,
                            dropout=dropout, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 48),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(48, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out).squeeze(-1)


class CNN1D(nn.Module):
    def __init__(self, input_dim, hidden=96):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(input_dim, hidden, 3, padding=1),
            nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Conv1d(hidden, hidden, 7, padding=3),
            nn.BatchNorm1d(hidden), nn.GELU(),
        )
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.GELU(), nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.layers(x)
        x = x.permute(0, 2, 1)
        return self.head(x).squeeze(-1)


def train_model(model_cls, kwargs, n_epochs=25, lr=1e-3, bs=128):
    oof_preds = {}
    test_preds = {}
    for fold in range(5):
        print(f'  fold {fold}...')
        model = model_cls(**kwargs).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
        criterion = nn.BCEWithLogitsLoss(reduction='none')

        train_ds = SeqDataset(sequences, 'train', fold)
        val_ds = SeqDataset(sequences, 'val', fold)
        test_ds = SeqDataset(sequences, 'test', fold)

        train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate_fn, num_workers=0)
        val_dl = DataLoader(val_ds, batch_size=bs*2, shuffle=False, collate_fn=collate_fn, num_workers=0)
        test_dl = DataLoader(test_ds, batch_size=bs*2, shuffle=False, collate_fn=collate_fn, num_workers=0)

        best_auc = 0; best_state = None; patience = 4; no_imp = 0
        for epoch in range(n_epochs):
            model.train()
            for X, Y, M, lengths, _ in train_dl:
                X, Y, M = X.to(device), Y.to(device), M.to(device)
                logits = model(X)
                loss = (criterion(logits, Y) * M.float()).sum() / M.float().sum().clamp(min=1)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()

            model.eval()
            vp, vl = [], []
            with torch.no_grad():
                for X, Y, M, lengths, _ in val_dl:
                    X, Y, M = X.to(device), Y.to(device), M.to(device)
                    probs = torch.sigmoid(model(X))
                    for b in range(X.shape[0]):
                        idx = M[b].nonzero(as_tuple=True)[0]
                        vp.extend(probs[b][idx].cpu().numpy())
                        vl.extend(Y[b][idx].cpu().numpy())
            if vl:
                auc = roc_auc_score(vl, vp)
                if auc > best_auc:
                    best_auc = auc; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; no_imp = 0
                else:
                    no_imp += 1
                if no_imp >= patience:
                    break

        print(f'    best val AUC = {best_auc:.6f}')
        model.load_state_dict(best_state); model.eval()

        with torch.no_grad():
            for X, Y, M, lengths, ids_batch in val_dl:
                X = X.to(device)
                probs = torch.sigmoid(model(X))
                for b in range(X.shape[0]):
                    idx = M[b].nonzero(as_tuple=True)[0]
                    for i in idx:
                        oof_preds[ids_batch[b][i.item()].item()] = probs[b][i].item()

            for X, Y, M, lengths, ids_batch in test_dl:
                X = X.to(device)
                probs = torch.sigmoid(model(X))
                for b in range(X.shape[0]):
                    idx = M[b].nonzero(as_tuple=True)[0]
                    for i in idx:
                        rid = ids_batch[b][i.item()].item()
                        test_preds.setdefault(rid, []).append(probs[b][i].item())

        del model, opt; gc.collect(); torch.cuda.empty_cache()
    return oof_preds, test_preds


print('\n=== BiLSTM (GPU) ===')
lstm_oof, lstm_test = train_model(BiLSTM, dict(input_dim=N_FEAT, hidden=96, layers=2, dropout=0.3),
                                   n_epochs=30, lr=1e-3, bs=128)

print('\n=== 1D-CNN (GPU) ===')
cnn_oof, cnn_test = train_model(CNN1D, dict(input_dim=N_FEAT, hidden=96),
                                 n_epochs=30, lr=1e-3, bs=128)

def to_array(pred_dict, ids):
    return np.array([np.mean(pred_dict[i]) if i in pred_dict else 0.5 for i in ids])

oof_lstm = to_array(lstm_oof, train_ids_ordered)
oof_cnn = to_array(cnn_oof, train_ids_ordered)
test_lstm = to_array(lstm_test, test_ids_ordered)
test_cnn = to_array(cnn_test, test_ids_ordered)

print(f'\nLSTM OOF AUC = {roc_auc_score(y_full, oof_lstm):.6f}')
print(f'CNN  OOF AUC = {roc_auc_score(y_full, oof_cnn):.6f}')

# Load GBM OOF
print('\n=== Loading GBM ===')
gbm_path = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'oof_lgb.npy' in files:
        gbm_path = root; break

if gbm_path:
    oof_lgb = np.load(f'{gbm_path}/oof_lgb.npy')
    oof_xgb = np.load(f'{gbm_path}/oof_xgb.npy')
    oof_cat = np.load(f'{gbm_path}/oof_cat.npy')
    oof_dart = np.load(f'{gbm_path}/oof_dart.npy')
    test_lgb = np.load(f'{gbm_path}/test_lgb.npy')
    test_xgb = np.load(f'{gbm_path}/test_xgb.npy')
    test_cat = np.load(f'{gbm_path}/test_cat.npy')
    test_dart = np.load(f'{gbm_path}/test_dart.npy')
    names = ['lgb', 'xgb', 'cat', 'dart', 'lstm', 'cnn']
    oofs = [oof_lgb, oof_xgb, oof_cat, oof_dart, oof_lstm, oof_cnn]
    tests = [test_lgb, test_xgb, test_cat, test_dart, test_lstm, test_cnn]
else:
    print('  GBM not found, neural only')
    names = ['lstm', 'cnn']
    oofs = [oof_lstm, oof_cnn]
    tests = [test_lstm, test_cnn]

# Blend
print('\n=== Blend ===')
oof_mat = np.stack(oofs, axis=1).astype(np.float64)
test_mat = np.stack(tests, axis=1).astype(np.float64)

for n, o in zip(names, oofs):
    print(f'  {n}: {roc_auc_score(y_full, o):.6f}')

print('\nCorrelations:')
C = np.corrcoef(oof_mat.T)
print('       ' + ' '.join(f'{n:>7s}' for n in names))
for i, n in enumerate(names):
    print(f'{n:>5s}  ' + ' '.join(f'{C[i,j]:7.4f}' for j in range(len(names))))

def neg_auc(w, X, y):
    w = np.abs(w); w /= max(w.sum(), 1e-9)
    return -roc_auc_score(y, X @ w)

w0 = np.ones(len(names)) / len(names)
res = minimize(neg_auc, x0=w0, args=(oof_mat, y_full), method='Nelder-Mead',
               options={'xatol': 1e-8, 'fatol': 1e-9, 'maxiter': 30000})
w = np.abs(res.x); w /= w.sum()
print(f'\nweights: {dict(zip(names, w.round(4)))}')
w_oof = oof_mat @ w
w_test = test_mat @ w
print(f'weighted OOF AUC = {roc_auc_score(y_full, w_oof):.6f}')

avg_oof = oof_mat.mean(axis=1)
avg_test = test_mat.mean(axis=1)
print(f'simple avg OOF AUC = {roc_auc_score(y_full, avg_oof):.6f}')

best_auc = max(roc_auc_score(y_full, w_oof), roc_auc_score(y_full, avg_oof))
best_test = w_test if roc_auc_score(y_full, w_oof) >= roc_auc_score(y_full, avg_oof) else avg_test
print(f'\n>>> BEST OOF AUC = {best_auc:.6f}')

pd.DataFrame({'id': test_ids_ordered, 'PitNextLap': best_test}).to_csv(f'{OUT}/submission.csv', index=False)
print(f'runtime: {(time.time()-t0)/60:.1f} min')
