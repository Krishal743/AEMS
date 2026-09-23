"""Ablation ladder: attribute the R@1 gain to each change, one at a time.

Each row adds exactly one change to the row above. Fixed weights are tuned per
row on the validation split carved from train; the test split is scored once
per row. Every row reports a bootstrap CI and a paired delta against the row
above, so a gain is only claimed when the interval excludes zero.

    A0   text only (mean-pooled caption)
    A1   + visual and CLAP audio, raw weighted sum   (pre-session arithmetic)
    A2   + per-query z-scoring before fusion
    A3   + WavLM adapter audio instead of CLAP
    A4   + passage branch (late interaction)
    A5   + adaptive gating, cross-fitted              (deployed)
    A5'  + adaptive gating trained WITHOUT cross-fitting

Rows A1-A4 differ only in how the same branches are combined; A5/A5' swap the
fixed weights for per-query predicted ones.
"""

import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F

from src.config import (AEMS_MANIFEST_PATH, AEMS_VID_EMBEDDINGS_PATH, AEMS_AUDIO_EMBEDDINGS_PATH,
                        AEMS_CLAP_AUDIO_EMBEDDINGS_PATH, AEMS_GATING_WEIGHTS_PATH,
                        AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE, AEMS_TEXT_CHUNKS_PATH_TEMPLATE,
                        DEVICE, set_seeds)
from src.evaluation.evaluate_retrieval import (ground_truth_ranks, hits_at_k, metrics_from_ranks,
                                               bootstrap_ci, paired_bootstrap)
from src.routing.query_router import ChunkIndex, load_gate, zscore
from src.training.query_data import (load_records, questions, validation_split, encode_clip_text,
                                     flatten_questions, stack_embeddings, query_rows, recall_metrics)

parser = argparse.ArgumentParser(description="AEMS ablation ladder")
parser.add_argument("--manifest", default=AEMS_MANIFEST_PATH)
parser.add_argument("--gate-weights", default=AEMS_GATING_WEIGHTS_PATH)
parser.add_argument("--gate-weights-no-crossfit", default="models/aems_gating_weights_no_crossfit_v1.pth")
parser.add_argument("--bootstrap-iters", type=int, default=1000)
parser.add_argument("--output", default="outputs/aems/ablations.json")
parser.add_argument("--skip-clap", action="store_true",
                    help="skip rows that need the CLAP encoder (A1, A2)")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
set_seeds(args.seed)

# ---------------------------------------------------------------- data
records = {split: load_records(args.manifest, split) for split in ("train", "test")}
vid_db = torch.load(AEMS_VID_EMBEDDINGS_PATH, weights_only=False)
wavlm_db = torch.load(AEMS_AUDIO_EMBEDDINGS_PATH, weights_only=False)
clap_db = torch.load(AEMS_CLAP_AUDIO_EMBEDDINGS_PATH, weights_only=False)
txt_db = {s: torch.load(AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE.format(split=s), weights_only=False)
          for s in ("train", "test")}
chunk_db = {s: torch.load(AEMS_TEXT_CHUNKS_PATH_TEMPLATE.format(split=s), weights_only=False)
            for s in ("train", "test")}


def usable(split):
    return [v for v, r in records[split].items()
            if v in vid_db and v in wavlm_db and v in clap_db and v in txt_db[split]
            and v in chunk_db[split] and questions(r)]


_, val_vids = validation_split(usable("train"))
test_vids = sorted(usable("test"))
print(f"[DATA] val={len(val_vids)} test={len(test_vids)} videos", flush=True)

splits = {"val": (val_vids, "train"), "test": (test_vids, "test")}
all_texts, all_rows, offset = [], {}, 0
for name, (vids, split) in splits.items():
    texts, rows = flatten_questions(records[split], vids)
    all_rows[name] = {v: [i + offset for i in idxs] for v, idxs in rows.items()}
    all_texts += texts
    offset += len(texts)

print("[ENC] Encoding QA questions with CLIP...", flush=True)
q_clip = encode_clip_text(all_texts, DEVICE)

q_clap = None
if not args.skip_clap:
    print("[ENC] Encoding QA questions with CLAP (for the pre-session audio rows)...", flush=True)
    from src.encoders.clap_encode import CLAPEncoder
    clap = CLAPEncoder(device=DEVICE)
    parts = []
    for i in range(0, len(all_texts), 256):
        emb = clap.encode_text(all_texts[i:i + 256])
        parts.append(F.normalize(torch.as_tensor(np.asarray(emb)).float(), dim=1))
    q_clap = torch.cat(parts)
    del clap
    torch.cuda.empty_cache()


def branch_sims(split_name):
    vids, split = splits[split_name]
    idx, gt = query_rows(all_rows[split_name], vids, DEVICE)
    qc = q_clip[idx].to(DEVICE)
    chunk_rows, owner = [], []
    for i, v in enumerate(vids):
        e = F.normalize(torch.as_tensor(chunk_db[split][v]).float().reshape(-1, 512), dim=1)
        chunk_rows.append(e)
        owner += [i] * e.shape[0]
    chunks = ChunkIndex(torch.cat(chunk_rows).to(DEVICE), torch.tensor(owner, device=DEVICE), len(vids))
    sims = {
        "visual": qc @ stack_embeddings(vid_db, vids, DEVICE).T,
        "text": qc @ stack_embeddings(txt_db[split], vids, DEVICE).T,
        "chunk": chunks.max_sim_batch(qc),
        "audio_wavlm": qc @ stack_embeddings(wavlm_db, vids, DEVICE).T,
    }
    if q_clap is not None:
        sims["audio_clap"] = q_clap[idx].to(DEVICE) @ stack_embeddings(clap_db, vids, DEVICE).T
    return {"q": qc, "gt": gt, "sims": sims}


data = {name: branch_sims(name) for name in splits}
print(f"[DATA] val queries={data['val']['q'].shape[0]} test queries={data['test']['q'].shape[0]}",
      flush=True)

# ---------------------------------------------------------------- rows
GRID = [round(x, 2) for x in np.arange(0, 1.01, 0.1)]


def combine(sims, names, weights, standardize):
    parts = [zscore(sims[n]) if standardize else sims[n] for n in names]
    return sum(w * p for w, p in zip(weights, parts))


def tune_on_val(names, standardize, anchor=1):
    """Grid-search fixed weights on validation; the anchor branch is pinned to 1.0."""
    best, best_w = -1.0, None
    free = [i for i in range(len(names)) if i != anchor]
    for combo in np.ndindex(*[len(GRID)] * len(free)):
        w = [1.0] * len(names)
        for slot, g in zip(free, combo):
            w[slot] = GRID[g]
        r1 = recall_metrics(combine(data["val"]["sims"], names, w, standardize),
                            data["val"]["gt"])["R@1"]
        if r1 > best:
            best, best_w = r1, list(w)
    return best, best_w


def gate_scores(path, split_name):
    gate = load_gate(path, DEVICE)
    names = ["visual", "text", "chunk", "audio_wavlm"]
    with torch.no_grad():
        w = gate(data[split_name]["q"])
    sims = data[split_name]["sims"]
    fused = sum(w[:, i:i + 1] * zscore(sims[n]) for i, n in enumerate(names))
    return fused, w


rows = []
rows.append({"id": "A0", "label": "text only (mean-pooled)",
             "names": ["text"], "weights": [1.0], "z": False})
if q_clap is not None:
    rows.append({"id": "A1", "label": "+ visual & CLAP audio, raw sum",
                 "names": ["visual", "text", "audio_clap"], "z": False})
    rows.append({"id": "A2", "label": "+ per-query z-scoring",
                 "names": ["visual", "text", "audio_clap"], "z": True})
rows.append({"id": "A3", "label": "+ WavLM adapter audio (was CLAP)",
             "names": ["visual", "text", "audio_wavlm"], "z": True})
rows.append({"id": "A4", "label": "+ passage branch (late interaction)",
             "names": ["visual", "text", "chunk", "audio_wavlm"], "z": True})

results, ranks, order = {}, {}, []
for row in rows:
    names = row["names"]
    if "weights" in row:
        val_r1, weights = recall_metrics(combine(data["val"]["sims"], names, row["weights"], row["z"]),
                                         data["val"]["gt"])["R@1"], row["weights"]
    else:
        val_r1, weights = tune_on_val(names, row["z"])
    test_scores = combine(data["test"]["sims"], names, weights, row["z"])
    ranks[row["id"]] = ground_truth_ranks(test_scores, data["test"]["gt"]).cpu()
    results[row["id"]] = {"label": row["label"], "branches": names, "weights": weights,
                          "z_scored": row["z"], "val_R@1": val_r1,
                          "test": metrics_from_ranks(ranks[row["id"]])}
    order.append(row["id"])
    print(f"  {row['id']:>3} {row['label']:<38} val={val_r1:.4f} "
          f"test R@1={results[row['id']]['test']['R@1']:.4f} w={weights}", flush=True)

for row_id, path, label in (("A5", args.gate_weights, "+ adaptive gating (cross-fitted)"),
                            ("A5'", args.gate_weights_no_crossfit,
                             "+ adaptive gating WITHOUT cross-fitting")):
    if not os.path.exists(path):
        print(f"  [skip] {row_id}: {path} not found", flush=True)
        continue
    val_scores, _ = gate_scores(path, "val")
    test_scores, w_test = gate_scores(path, "test")
    ranks[row_id] = ground_truth_ranks(test_scores, data["test"]["gt"]).cpu()
    results[row_id] = {
        "label": label, "branches": ["visual", "text", "chunk", "audio_wavlm"],
        "weights": "per-query (gate)", "z_scored": True,
        "val_R@1": recall_metrics(val_scores, data["val"]["gt"])["R@1"],
        "test": metrics_from_ranks(ranks[row_id]),
        "gate_weight_mean": w_test.mean(0).tolist(), "gate_weight_std": w_test.std(0).tolist(),
        "checkpoint": os.path.basename(path)}
    order.append(row_id)
    print(f"  {row_id:>3} {label:<38} val={results[row_id]['val_R@1']:.4f} "
          f"test R@1={results[row_id]['test']['R@1']:.4f} "
          f"w={[round(x, 3) for x in results[row_id]['gate_weight_mean']]}", flush=True)

# ---------------------------------------------------------------- statistics
ladder = [r for r in order if r != "A5'"]
KS = (1, 5, 10)
for row_id in order:
    for k in KS:
        low, high = bootstrap_ci(hits_at_k(ranks[row_id], k), iters=args.bootstrap_iters)
        results[row_id][f"test_R@{k}_CI95"] = [low, high]
    # compare against the row this one builds on
    previous = "A4" if row_id == "A5'" else (ladder[ladder.index(row_id) - 1]
                                             if row_id in ladder and ladder.index(row_id) > 0 else None)
    if previous:
        results[row_id]["delta_vs"] = previous
        for k in KS:
            results[row_id][f"test_R@{k}_delta"] = list(paired_bootstrap(
                hits_at_k(ranks[row_id], k), hits_at_k(ranks[previous], k),
                iters=args.bootstrap_iters))

if "A5" in results and "A5'" in results:
    results["cross_fitting_effect"] = {
        f"A5_minus_A5prime_R@{k}": list(paired_bootstrap(
            hits_at_k(ranks["A5"], k), hits_at_k(ranks["A5'"], k), iters=args.bootstrap_iters))
        for k in KS}

def fmt_delta(row, k):
    d = row.get(f"test_R@{k}_delta")
    if not d:
        return "—"
    sig = "sig " if (d[1] > 0 or d[2] < 0) else "n.s."
    return f"{d[0]:+.4f} {sig}"


print("\n" + "=" * 104)
print(f"{'row':<5}{'configuration':<42}{'test R@1 [95% CI]':<26}"
      f"{'ΔR@1':<14}{'ΔR@5':<14}{'vs':<5}")
print("=" * 104)
for row_id in order:
    r = results[row_id]
    ci = r["test_R@1_CI95"]
    print(f"{row_id:<5}{r['label']:<42}"
          f"{r['test']['R@1']:.4f} [{ci[0]:.4f},{ci[1]:.4f}]   "
          f"{fmt_delta(r, 1):<14}{fmt_delta(r, 5):<14}{r.get('delta_vs', ''):<5}")
if "cross_fitting_effect" in results:
    print("\nCross-fitting the gate's training audio (A5 - A5'):")
    for k in KS:
        d = results["cross_fitting_effect"][f"A5_minus_A5prime_R@{k}"]
        sig = "significant" if (d[1] > 0 or d[2] < 0) else "not significant"
        print(f"  R@{k}: {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}] — {sig}")

os.makedirs(os.path.dirname(args.output), exist_ok=True)
with open(args.output, "w") as f:
    json.dump({"rows": results, "order": order, "n_test_queries": int(data["test"]["gt"].numel()),
               "n_test_videos": len(test_vids), "bootstrap_iters": args.bootstrap_iters}, f, indent=2)
print(f"\n[SAVE] {args.output}")
