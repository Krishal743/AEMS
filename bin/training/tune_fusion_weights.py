"""Tune the fixed fusion weights on validation by coordinate ascent.

The deployed weights have had to be re-tuned every time the branch set or an
encoder changed, and each time it was done in a throwaway script. This is that
procedure, written down: the weights in `src/config.py` should be reproducible
by running this.

Coordinate ascent rather than a grid: six branches on a grid is intractable,
and the objective is cheap but noisy, so repeated sweeps over one coordinate at
a time converge in a few minutes. Selection is validation R@1 of stage 1 alone
(no reranking) — the reranker is tuned separately and would mask a bad fusion.

The search starts from the current config weights, so a sweep can only keep or
improve on what is deployed.
"""

import argparse, json, os
import torch

from src.config import (AEMS_MANIFEST_PATH, AEMS_FUSION_WEIGHTS, DEVICE, set_seeds)
from src.evaluation.evaluate_retrieval import ground_truth_ranks, metrics_from_ranks
from src.rerank import stage1
from src.retrieval.branches import BranchSources, build_sims, encode_queries
from src.routing.query_router import BRANCHES
from src.training.query_data import (load_records, questions, validation_split,
                                     flatten_questions, query_rows)

parser = argparse.ArgumentParser(description="Tune fixed fusion weights on validation")
parser.add_argument("--manifest", default=AEMS_MANIFEST_PATH)
parser.add_argument("--val-frac", type=float, default=0.15)
parser.add_argument("--sweeps", type=int, default=3, help="passes over all branches")
parser.add_argument("--grid", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.1, 1.4, 1.8],
                    help="candidate values tried for each branch in turn")
parser.add_argument("--floor", type=float, default=0.0,
                    help="smallest weight any branch may take. A floor keeps every branch "
                         "contributing: unconstrained ascent will zero a weak-but-correlated "
                         "branch for a fraction of a point that may not survive a bootstrap.")
parser.add_argument("--subset", default=None,
                    help="query-subset JSON from bin/data/build_visually_grounded_subset.py; "
                         "restricts tuning to one tier of it")
parser.add_argument("--tier", default="visually_grounded",
                    help="which tier of --subset to tune on")
parser.add_argument("--output", default="outputs/aems/fusion_weight_tuning.json")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
set_seeds(args.seed)

records = load_records(args.manifest, "train")
sources = BranchSources.load(records, "train")
usable = sources.usable([v for v, r in records.items() if questions(r)])
_, val_vids = validation_split(usable, args.val_frac)
print(f"[DATA] val={len(val_vids)} videos (fit split unused here; test untouched)", flush=True)

if args.subset:
    import json
    position = {v: i for i, v in enumerate(val_vids)}
    chosen = [q for q in json.load(open(args.subset))["queries"]
              if q["video_id"] in position and args.tier in q["tiers"]]
    if not chosen:
        raise SystemExit(f"no {args.tier!r} queries fall in the validation split")
    query_texts = [q["question"] for q in chosen]
    gt = torch.tensor([position[q["video_id"]] for q in chosen], device=DEVICE)
    idx = None
    print(f"[DATA] tuning on {len(query_texts)} {args.tier!r} queries", flush=True)
else:
    texts, rows = flatten_questions(records, val_vids)
    idx, gt = query_rows(rows, val_vids, DEVICE)
    query_texts = [texts[i] for i in idx.tolist()]
print(f"[ENC] Encoding {len(query_texts)} validation queries...", flush=True)
clip_q, dense_q = encode_queries(query_texts, DEVICE)
sims = build_sims(sources, val_vids, clip_q.to(DEVICE), query_texts,
                  dense_q.to(DEVICE), DEVICE)


def score(weights):
    w = torch.tensor([[weights[b] for b in BRANCHES]], dtype=torch.float32, device=DEVICE)
    fused = stage1.fuse(w, sims)
    return metrics_from_ranks(ground_truth_ranks(fused, gt))["R@1"]


weights = {b: max(AEMS_FUSION_WEIGHTS[b], args.floor) for b in BRANCHES}
best = score(weights)
print(f"[BASELINE] current config weights: val R@1={best:.4f}", flush=True)
for b in BRANCHES:
    solo = {k: (1.0 if k == b else 0.0) for k in BRANCHES}
    print(f"  {b:<7} alone: val R@1={score(solo):.4f}", flush=True)

history = []
for sweep in range(args.sweeps):
    improved = False
    for b in BRANCHES:
        start = weights[b]
        for value in [max(v, args.floor) for v in args.grid]:
            if value == start:
                continue
            trial = dict(weights, **{b: value})
            if not any(trial.values()):
                continue
            r1 = score(trial)
            if r1 > best + 1e-6:
                best, weights, improved = r1, trial, True
        if weights[b] != start:
            print(f"  sweep {sweep + 1}: {b} {start:g} -> {weights[b]:g}  val R@1={best:.4f}",
                  flush=True)
    history.append({"sweep": sweep + 1, "val_R@1": best, "weights": dict(weights)})
    if not improved:
        print(f"[STOP] sweep {sweep + 1} changed nothing; converged", flush=True)
        break

print(f"\n[BEST] val R@1={best:.4f} (was {history[0]['val_R@1'] if history else best:.4f})")
print("[WEIGHTS] AEMS_FUSION_WEIGHTS = {")
for b in BRANCHES:
    print(f'    "{b}": {weights[b]:g},')
print("}")

os.makedirs(os.path.dirname(args.output), exist_ok=True)
with open(args.output, "w") as f:
    json.dump({"val_R@1": best, "val_R@1_baseline": baseline, "weights": weights,
               "baseline_weights": AEMS_FUSION_WEIGHTS,
               "history": history, "grid": args.grid, "seed": args.seed}, f, indent=2)
print(f"[SAVE] {args.output}")
