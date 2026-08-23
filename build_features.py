"""Build shared feature set for all adversarial agents.

Outputs:
- features_train.parquet, features_test.parquet
- folds.parquet (5-fold StratifiedKFold indices)
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

DATA = '.'

print('[fe] loading...')
tr = pd.read_csv(f'{DATA}/train.csv')
te = pd.read_csv(f'{DATA}/test.csv')

n_tr = len(tr)
print(f'[fe] train={n_tr:,}  test={len(te):,}')

# Combine for unified feature engineering.
te['PitNextLap'] = np.nan
df = pd.concat([tr.assign(_src='train'), te.assign(_src='test')], ignore_index=True)
df['LapTime'] = df['LapTime (s)']  # rename for convenience

# ---- Stint-level reconstruction (the "Normalized_TyreLife" the org dropped) ----
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

# ---- (Driver, Race, Year) aggregates ----
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

# ---- (Race, Year) aggregates ----
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

# ---- Compound x Race aggregates ----
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

# ---- Lap-time normalised within race ----
df['LapTime_vs_race_min'] = df['LapTime'] - df['race_min_lap']
df['LapTime_vs_race_mean'] = df['LapTime'] - df['race_mean_lap']
df['LapTime_vs_dr_min'] = df['LapTime'] - df['dr_min_lap']
df['LapTime_vs_dr_mean'] = df['LapTime'] - df['dr_mean_lap']

# ---- Position / change features ----
df['Position_log'] = np.log1p(df['Position'])
df['LapsToEnd'] = df['race_max_lap'] - df['LapNumber']
df['StintsToGo'] = df['dr_max_stint'] - df['Stint']

# ---- Compound encoding ----
compound_order = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2, 'INTERMEDIATE': 3, 'WET': 4}
df['Compound_ord'] = df['Compound'].map(compound_order).fillna(-1).astype(int)

# ---- Race target encoding (using train only with smoothing) ----
race_te = tr.groupby('Race')['PitNextLap'].agg(['mean', 'count']).reset_index()
global_mean = tr['PitNextLap'].mean()
SMOOTH = 50
race_te['Race_TE'] = (race_te['mean'] * race_te['count'] + global_mean * SMOOTH) / (race_te['count'] + SMOOTH)
df = df.merge(race_te[['Race', 'Race_TE']], on='Race', how='left')

# Driver target encoding
drv_te = tr.groupby('Driver')['PitNextLap'].agg(['mean', 'count']).reset_index()
drv_te['Driver_TE'] = (drv_te['mean'] * drv_te['count'] + global_mean * SMOOTH) / (drv_te['count'] + SMOOTH)
df = df.merge(drv_te[['Driver', 'Driver_TE']], on='Driver', how='left')
df['Driver_TE'] = df['Driver_TE'].fillna(global_mean)

# Compound TE
cmp_te = tr.groupby('Compound')['PitNextLap'].agg(['mean', 'count']).reset_index()
cmp_te['Compound_TE'] = (cmp_te['mean'] * cmp_te['count'] + global_mean * SMOOTH) / (cmp_te['count'] + SMOOTH)
df = df.merge(cmp_te[['Compound', 'Compound_TE']], on='Compound', how='left')

# ---- Categorical codes ----
for c in ['Driver', 'Race', 'Compound']:
    df[f'{c}_code'] = df[c].astype('category').cat.codes

# ---- Stint position codes ----
# rank of TyreLife within (D,R,Y,S) - estimate where in the stint we are
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

# Restore order by id
df = df.sort_values('id').reset_index(drop=True)

# Split back
train_df = df[df['_src'] == 'train'].reset_index(drop=True)
test_df = df[df['_src'] == 'test'].reset_index(drop=True)

# Drop helpers
train_df = train_df.drop(columns=['_src'])
test_df = test_df.drop(columns=['_src', 'PitNextLap'])

print('[fe] train cols:', train_df.shape[1], 'rows:', len(train_df))
print('[fe] test cols:', test_df.shape[1], 'rows:', len(test_df))

# 5-fold StratifiedKFold (group-aware NOT used - since we use union for stint stats anyway)
y = train_df['PitNextLap'].astype(int).values
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = np.full(len(train_df), -1, dtype=np.int8)
for f, (_, vidx) in enumerate(skf.split(np.zeros(len(y)), y)):
    folds[vidx] = f
np.save(f'{DATA}/folds.npy', folds)
print('[fe] fold sizes:', np.bincount(folds))

train_df.to_parquet(f'{DATA}/features_train.parquet', index=False)
test_df.to_parquet(f'{DATA}/features_test.parquet', index=False)
print('[fe] wrote features_train/features_test parquet')
print('[fe] columns:')
for c in train_df.columns:
    print(' ', c, train_df[c].dtype)
