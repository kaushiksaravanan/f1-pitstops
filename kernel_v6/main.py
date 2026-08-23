"""
F1 Pit Stops v6 — LSTM + 1D-CNN sequence models blended with GBM OOF.

Key insight: GBMs treat each row independently. Pit stop decisions are
SEQUENTIAL — a driver's laps form a time series where degradation dynamics
(rising lap times, tyre wear curve) determine pit timing.

Strategy:
  1. Build per-(Driver, Race, Year) sequences
  2. Train LSTM that sees full lap sequence → predicts PitNextLap at each step
  3. Train 1D-CNN for local patterns (3-5 lap windows before pit)
  4. Load GBM OOF from v3 kernel output (attached as kernel source)
  5. Optimized blend of all models
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

# ====== FEATURES FOR SEQUENCE MODEL ======
# We use a subset of features that make sense as a time series per (Driver, Race, Year)
seq_feats = ['LapNumber', 'Position', 'TyreLife', 'Stint', 'PitStop', 'LapTime (s)']

te_copy = te.copy(); te_copy['PitNextLap'] = np.nan
df = pd.concat([tr.assign(_src='train'), te_copy.assign(_src='test')], ignore_index=True)
df['LapTime'] = df['LapTime (s)']

compound_order = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2, 'INTERMEDIATE': 3, 'WET': 4}
df['Compound_ord'] = df['Compound'].map(compound_order).fillna(-1).astype(int)

# Stint-level features
gkey = ['Driver', 'Race', 'Year', 'Stint']
g = df.groupby(gkey).agg(stint_max_tyre=('TyreLife', 'max')).reset_index()
df = df.merge(g, on=gkey, how='left')
df['Norm_TyreLife'] = df['TyreLife'] / df['stint_max_tyre'].clip(lower=1)

# Race-level
gkey3 = ['Race', 'Year']
g3 = df.groupby(gkey3).agg(race_max_lap=('LapNumber', 'max'), race_mean_lap=('LapTime', 'mean')).reset_index()
df = df.merge(g3, on=gkey3, how='left')
df['lap_progress'] = df['LapNumber'] / df['race_max_lap'].clip(lower=1)
df['LapTime_vs_mean'] = df['LapTime'] - df['race_mean_lap']

# Sort by sequence order
df = df.sort_values(['Driver', 'Race', 'Year', 'LapNumber']).reset_index(drop=True)

# Sequence features (per-row, will be arranged into sequences)
SEQ_COLS = ['LapNumber', 'Position', 'TyreLife', 'Stint', 'PitStop',
            'LapTime', 'Compound_ord', 'Norm_TyreLife', 'lap_progress', 'LapTime_vs_mean']
N_FEAT = len(SEQ_COLS)

# Group into sequences
print('=== building sequences ===')
groups = df.groupby(['Driver', 'Race', 'Year'])
sequences = []
seq_ids = []  # (id, _src) for each element in sequence
for (drv, race, yr), grp in groups:
    grp = grp.sort_values('LapNumber')
    feat_arr = grp[SEQ_COLS].values.astype(np.float32)
    ids_arr = grp[['id', '_src']].values
    sequences.append(feat_arr)
    seq_ids.append(ids_arr)

print(f'  total sequences: {len(sequences)}')
print(f'  seq lengths: min={min(len(s) for s in sequences)} max={max(len(s) for s in sequences)} mean={np.mean([len(s) for s in sequences]):.1f}')

# Rebuild flat arrays aligned with original id order for OOF scoring
df = df.sort_values('id').reset_index(drop=True)
train_mask = df['_src'] == 'train'
test_mask = df['_src'] == 'test'
y_full = df.loc[train_mask, 'PitNextLap'].astype(int).values
test_ids = df.loc[test_mask, 'id'].values
n_train = train_mask.sum()
n_test = test_mask.sum()
print(f'  n_train={n_train} n_test={n_test}')

# Folds (same seed=42 as GBM kernels for OOF alignment)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
id_to_fold = {}
for f, (_, vidx) in enumerate(skf.split(np.zeros(n_train), y_full)):
    for i in vidx:
        train_id = df.loc[train_mask].iloc[i]['id']
        id_to_fold[train_id] = f

# ====== PYTORCH MODELS ======
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

device = torch.device('cpu')
print(f'device={device}')


class SeqDataset(Dataset):
    """Each item is a full (Driver, Race, Year) sequence."""
    def __init__(self, sequences, seq_ids, mode='train', fold=None, id_to_fold=None):
        self.data = []
        for seq, ids in zip(sequences, seq_ids):
            # Build targets and masks
            targets = []
            masks = []  # which positions to evaluate
            for i in range(len(seq)):
                row_id = int(ids[i][0])
                src = ids[i][1]
                if src == 'train':
                    if mode == 'train' and id_to_fold.get(row_id, -1) != fold:
                        masks.append(True)
                        targets.append(df.loc[df['id'] == row_id, 'PitNextLap'].values[0])
                    elif mode == 'val' and id_to_fold.get(row_id, -1) == fold:
                        masks.append(True)
                        targets.append(df.loc[df['id'] == row_id, 'PitNextLap'].values[0])
                    else:
                        masks.append(False)
                        targets.append(0)
                elif src == 'test' and mode == 'test':
                    masks.append(True)
                    targets.append(0)
                else:
                    masks.append(False)
                    targets.append(0)
            if any(masks):
                self.data.append((seq, np.array(targets, dtype=np.float32),
                                  np.array(masks, dtype=bool), ids))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, targets, masks, ids = self.data[idx]
        return torch.FloatTensor(seq), torch.FloatTensor(targets), torch.BoolTensor(masks), ids


def collate_fn(batch):
    # Pad sequences to max length in batch
    max_len = max(b[0].shape[0] for b in batch)
    X = torch.zeros(len(batch), max_len, N_FEAT)
    Y = torch.zeros(len(batch), max_len)
    M = torch.zeros(len(batch), max_len, dtype=torch.bool)
    all_ids = []
    for i, (x, y, m, ids) in enumerate(batch):
        L = x.shape[0]
        X[i, :L] = x
        Y[i, :L] = y
        M[i, :L] = m
        all_ids.append(ids)
    return X, Y, M, all_ids


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, n_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True,
                            dropout=dropout, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out).squeeze(-1)


class CNN1DModel(nn.Module):
    def __init__(self, input_dim, hidden=64):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(hidden, hidden, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (B, T, F) -> transpose to (B, F, T)
        x = x.permute(0, 2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        # (B, H, T) -> (B, T, H)
        x = x.permute(0, 2, 1)
        return self.head(x).squeeze(-1)


def train_seq_model(model_class, model_kwargs, sequences, seq_ids, n_epochs=15, lr=1e-3, bs=32):
    """Train sequence model with 5-fold OOF."""
    oof_preds = {}  # id -> prediction
    test_preds = {}  # id -> list of predictions (averaged)

    for fold in range(5):
        print(f'  fold {fold}...')
        model = model_class(**model_kwargs).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        criterion = nn.BCEWithLogitsLoss(reduction='none')

        train_ds = SeqDataset(sequences, seq_ids, 'train', fold, id_to_fold)
        val_ds = SeqDataset(sequences, seq_ids, 'val', fold, id_to_fold)
        test_ds = SeqDataset(sequences, seq_ids, 'test', fold, id_to_fold)

        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=bs*2, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_ds, batch_size=bs*2, shuffle=False, collate_fn=collate_fn)

        best_auc = 0
        best_state = None
        patience = 3
        no_improve = 0

        for epoch in range(n_epochs):
            model.train()
            for X, Y, M, _ in train_loader:
                X, Y, M = X.to(device), Y.to(device), M.to(device)
                logits = model(X)
                loss = (criterion(logits, Y) * M.float()).sum() / M.float().sum().clamp(min=1)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            # Validate
            model.eval()
            val_preds_epoch = []
            val_labels_epoch = []
            with torch.no_grad():
                for X, Y, M, _ in val_loader:
                    X, Y, M = X.to(device), Y.to(device), M.to(device)
                    logits = model(X)
                    probs = torch.sigmoid(logits)
                    for b in range(X.shape[0]):
                        mask = M[b]
                        val_preds_epoch.extend(probs[b][mask].cpu().numpy())
                        val_labels_epoch.extend(Y[b][mask].cpu().numpy())

            if len(val_labels_epoch) > 0:
                auc = roc_auc_score(val_labels_epoch, val_preds_epoch)
                if auc > best_auc:
                    best_auc = auc
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= patience:
                    break

        print(f'    best val AUC = {best_auc:.6f} @ epoch {epoch - no_improve}')
        model.load_state_dict(best_state)
        model.eval()

        # OOF predictions
        with torch.no_grad():
            for X, Y, M, ids_batch in val_loader:
                X = X.to(device)
                logits = model(X)
                probs = torch.sigmoid(logits)
                for b in range(X.shape[0]):
                    mask = M[b]
                    ids_b = ids_batch[b]
                    indices = mask.nonzero(as_tuple=True)[0]
                    for idx in indices:
                        row_id = int(ids_b[idx.item()][0])
                        oof_preds[row_id] = probs[b][idx].item()

        # Test predictions
        with torch.no_grad():
            for X, Y, M, ids_batch in test_loader:
                X = X.to(device)
                logits = model(X)
                probs = torch.sigmoid(logits)
                for b in range(X.shape[0]):
                    mask = M[b]
                    ids_b = ids_batch[b]
                    indices = mask.nonzero(as_tuple=True)[0]
                    for idx in indices:
                        row_id = int(ids_b[idx.item()][0])
                        if row_id not in test_preds:
                            test_preds[row_id] = []
                        test_preds[row_id].append(probs[b][idx].item())

        del model, optimizer, train_ds, val_ds, test_ds
        gc.collect()

    return oof_preds, test_preds


# ====== TRAIN LSTM ======
print('\n=== Training BiLSTM ===')
lstm_oof, lstm_test = train_seq_model(
    LSTMModel, dict(input_dim=N_FEAT, hidden_dim=64, n_layers=2, dropout=0.3),
    sequences, seq_ids, n_epochs=20, lr=1e-3, bs=64)

# ====== TRAIN 1D-CNN ======
print('\n=== Training 1D-CNN ===')
cnn_oof, cnn_test = train_seq_model(
    CNN1DModel, dict(input_dim=N_FEAT, hidden=64),
    sequences, seq_ids, n_epochs=20, lr=1e-3, bs=64)

# ====== ALIGN TO SUBMISSION ORDER ======
train_ids_ordered = df.loc[train_mask, 'id'].values
test_ids_ordered = df.loc[test_mask, 'id'].values

def preds_to_array(pred_dict, ids):
    arr = np.zeros(len(ids))
    for i, row_id in enumerate(ids):
        if row_id in pred_dict:
            v = pred_dict[row_id]
            arr[i] = np.mean(v) if isinstance(v, list) else v
        else:
            arr[i] = 0.5  # fallback
    return arr

oof_lstm_arr = preds_to_array(lstm_oof, train_ids_ordered)
oof_cnn_arr = preds_to_array(cnn_oof, train_ids_ordered)
test_lstm_arr = preds_to_array(lstm_test, test_ids_ordered)
test_cnn_arr = preds_to_array(cnn_test, test_ids_ordered)

print(f'\nLSTM OOF AUC = {roc_auc_score(y_full, oof_lstm_arr):.6f}')
print(f'CNN  OOF AUC = {roc_auc_score(y_full, oof_cnn_arr):.6f}')

# ====== LOAD GBM OOF from v3 kernel ======
print('\n=== Loading GBM OOF from attached kernel source ===')
gbm_path = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'oof_lgb.npy' in files:
        gbm_path = root
        break
if gbm_path:
    print(f'  GBM outputs at {gbm_path}')
    oof_lgb = np.load(f'{gbm_path}/oof_lgb.npy')
    oof_xgb = np.load(f'{gbm_path}/oof_xgb.npy')
    oof_cat = np.load(f'{gbm_path}/oof_cat.npy')
    oof_dart = np.load(f'{gbm_path}/oof_dart.npy')
    test_lgb = np.load(f'{gbm_path}/test_lgb.npy')
    test_xgb = np.load(f'{gbm_path}/test_xgb.npy')
    test_cat = np.load(f'{gbm_path}/test_cat.npy')
    test_dart = np.load(f'{gbm_path}/test_dart.npy')
    has_gbm = True
    for n, o in [('lgb', oof_lgb), ('xgb', oof_xgb), ('cat', oof_cat), ('dart', oof_dart)]:
        print(f'    {n} OOF AUC = {roc_auc_score(y_full, o):.6f}')
else:
    print('  GBM outputs not found — using neural models only')
    has_gbm = False

# ====== BLEND ======
print('\n=== Final blend ===')
if has_gbm:
    names = ['lgb', 'xgb', 'cat', 'dart', 'lstm', 'cnn']
    oofs = [oof_lgb, oof_xgb, oof_cat, oof_dart, oof_lstm_arr, oof_cnn_arr]
    tests = [test_lgb, test_xgb, test_cat, test_dart, test_lstm_arr, test_cnn_arr]
else:
    names = ['lstm', 'cnn']
    oofs = [oof_lstm_arr, oof_cnn_arr]
    tests = [test_lstm_arr, test_cnn_arr]

oof_mat = np.stack(oofs, axis=1).astype(np.float64)
test_mat = np.stack(tests, axis=1).astype(np.float64)

for n, o in zip(names, oofs):
    print(f'  {n}: {roc_auc_score(y_full, o):.6f}')

# Correlations
print('\nCorrelation matrix:')
C = np.corrcoef(oof_mat.T)
print('       ' + ' '.join(f'{n:>7s}' for n in names))
for i, n in enumerate(names):
    print(f'{n:>5s}  ' + ' '.join(f'{C[i,j]:7.4f}' for j in range(len(names))))

# Nelder-Mead
def neg_auc(w, X, y):
    w = np.abs(w); w /= max(w.sum(), 1e-9)
    return -roc_auc_score(y, X @ w)

w0 = np.ones(len(names)) / len(names)
res = minimize(neg_auc, x0=w0, args=(oof_mat, y_full), method='Nelder-Mead',
               options={'xatol': 1e-8, 'fatol': 1e-9, 'maxiter': 30000})
w = np.abs(res.x); w /= w.sum()
print(f'\nweighted blend weights: {dict(zip(names, w.round(4)))}')
w_oof = oof_mat @ w
w_test = test_mat @ w
print(f'weighted blend OOF AUC = {roc_auc_score(y_full, w_oof):.6f}')

# Simple avg
avg_oof = oof_mat.mean(axis=1)
avg_test = test_mat.mean(axis=1)
print(f'simple avg     OOF AUC = {roc_auc_score(y_full, avg_oof):.6f}')

# Pick best
candidates = {
    'weighted_mean': (roc_auc_score(y_full, w_oof), w_test),
    'simple_avg': (roc_auc_score(y_full, avg_oof), avg_test),
}
best = max(candidates, key=lambda k: candidates[k][0])
best_auc, best_test = candidates[best]
print(f'\n>>> BEST = {best}  OOF AUC = {best_auc:.6f}')

pd.DataFrame({'id': test_ids_ordered, 'PitNextLap': best_test}).to_csv(f'{OUT}/submission.csv', index=False)
print(f'\nTotal runtime: {(time.time()-t0)/60:.1f} min')
print('Done.')
