import os, numpy as np, pandas as pd

WD = "/teamspace/studios/this_studio"

# Load all submission.csv files
subs = {}
for root, dirs, files in os.walk(f"{WD}/pub"):
    for fn in files:
        if fn == "submission.csv":
            path = os.path.join(root, fn)
            df = pd.read_csv(path)
            if "PitNextLap" in df.columns and len(df) == 188165:
                label = root.split("/pub/")[-1].replace("/","_")
                subs[label] = df.sort_values("id")["PitNextLap"].values
                print(f"  {label}: mean={subs[label].mean():.4f}")

# Load our submissions
for fn, label in [("submission_v7.csv","ours_v7"), ("submission_v8_ours.csv","ours_v8")]:
    path = f"{WD}/{fn}"
    if os.path.isfile(path):
        df = pd.read_csv(path).sort_values("id")
        subs[label] = df["PitNextLap"].values
        print(f"  {label}: mean={subs[label].mean():.4f}")

print(f"\nTotal: {len(subs)} submissions")
test_ids = pd.read_csv(f"{WD}/test.csv")["id"].values

def normalized_rank(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, len(values))
    return ranks

def logit_transform(p):
    p = np.clip(p, 1e-6, 1-1e-6)
    return np.log(p/(1.0-p))

def sigmoid_transform(x):
    return 1.0/(1.0+np.exp(-x))

def logit_rank_blend(preds_list, weights):
    weights = np.array(weights, dtype=float)
    weights /= weights.sum()
    logit_ranks = [logit_transform(normalized_rank(p)) for p in preds_list]
    blended = sum(w*lr for w, lr in zip(weights, logit_ranks))
    blended_rank = sigmoid_transform(blended)
    anchor = preds_list[int(np.argmax(weights))]
    order = np.argsort(blended_rank, kind="mergesort")
    out = np.empty_like(anchor, dtype=float)
    out[order] = np.sort(anchor)
    return np.clip(out, 1e-7, 1-1e-7)

def rank_avg(preds_list):
    ranks = [pd.Series(p).rank(pct=True).values for p in preds_list]
    return np.mean(ranks, axis=0)

names = list(subs.keys())
preds = [subs[n] for n in names]

# Correlations
print("\n=== Rank Correlations ===")
rank_arrays = [normalized_rank(p) for p in preds]
C = np.corrcoef(rank_arrays)
for i, n in enumerate(names):
    row = " ".join(f"{C[i,j]:.3f}" for j in range(len(names)))
    print(f"  {n[:15]:>15s}: {row}")

# Generate candidates
candidates = {}

# 1. Simple rank avg all
candidates["rank_avg_all"] = rank_avg(preds)

# 2. Pub-only rank avg
pub_preds = [subs[n] for n in names if "ours" not in n]
pub_names = [n for n in names if "ours" not in n]
if pub_preds:
    candidates["pub_rank_avg"] = rank_avg(pub_preds)

# 3. Logit-rank all equal
candidates["logit_all_equal"] = logit_rank_blend(preds, [1]*len(preds))

# 4. Logit-rank pub only
if pub_preds:
    candidates["logit_pub_only"] = logit_rank_blend(pub_preds, [1]*len(pub_preds))

# 5. Weighted: pub 2x, ours 1x
w = [2.0 if "ours" not in n else 1.0 for n in names]
candidates["logit_pub2x"] = logit_rank_blend(preds, w)

# 6. Weighted: pub 3x, ours 0.5x
w = [3.0 if "ours" not in n else 0.5 for n in names]
candidates["logit_pub3x"] = logit_rank_blend(preds, w)

# 7. Anchor on best pub + corrections
# The blender_0.9545 sub is likely the strongest
for anchor_name in pub_names:
    if "blender" in anchor_name or "9545" in anchor_name:
        anchor = subs[anchor_name]
        others = [subs[n] for n in names if n != anchor_name]
        consensus_rank = rank_avg(others)
        for alpha in [0.02, 0.05, 0.10]:
            key = f"anchor_{anchor_name[:10]}_{int(alpha*100)}pct"
            blended_rank = (1-alpha)*normalized_rank(anchor) + alpha*consensus_rank
            order = np.argsort(blended_rank, kind="mergesort")
            out = np.empty_like(anchor, dtype=float)
            out[order] = np.sort(anchor)
            candidates[key] = np.clip(out, 1e-7, 1-1e-7)

# 8. Power-mean blends
for p_val in [0.5, 2.0]:
    # Power mean in probability space
    arr = np.stack(preds, axis=0)
    if p_val > 0:
        pm = np.mean(arr**p_val, axis=0)**(1/p_val)
    candidates[f"power_mean_p{p_val}"] = np.clip(pm, 1e-7, 1-1e-7)

print(f"\n=== Generated {len(candidates)} candidates ===")
for name, pred in candidates.items():
    pd.DataFrame({"id": test_ids, "PitNextLap": pred}).to_csv(f"{WD}/blend_{name}.csv", index=False)
    print(f"  {name}: mean={pred.mean():.5f}")

print("\nDONE")
