"""Score the system by how much a query's words overlap its own source text.

The headline number is an aggregate over 5,097 LLM-generated questions, and
those questions reuse the wording of the transcript and description they were
generated from: their content words appear in their own description 10.8x and
their own transcript 5.7x more often than in a random video's. If retrieval is
largely matching a question back to the text it came from, the aggregate
overstates what the system can do on a query someone actually types.

Stratifying by overlap separates the two. A system doing semantic retrieval
degrades gently as overlap falls; one exploiting the artifact falls off a
cliff. This is a cheap, general test for any benchmark whose queries were
generated from the documents being retrieved, and it belongs beside every
aggregate score this project reports.
"""

import argparse, json, os, re
import numpy as np
import torch

from src.config import AEMS_MANIFEST_PATH, DEVICE, set_seeds
from src.data.query_subsets import content_words, source_pool
from src.evaluation.evaluate_retrieval import (ground_truth_ranks, metrics_from_ranks,
                                               bootstrap_ci, hits_at_k)
from src.rerank import stage1
from src.retrieval.branches import BranchSources, build_sims, encode_queries
from src.routing.query_router import BRANCHES, fixed_weights
from src.training.query_data import load_records, questions, flatten_questions, query_rows

parser = argparse.ArgumentParser(description="Score stratified by query/source lexical overlap")
parser.add_argument("--manifest", default=AEMS_MANIFEST_PATH)
parser.add_argument("--split", default="test", choices=["train", "test"])
parser.add_argument("--edges", type=float, nargs="+", default=[0.0, 0.2, 0.6],
                    help="slice boundaries; 0.0 is reported as its own exact-zero slice")
parser.add_argument("--rerank-top-k", type=int, default=100,
                    help="shortlist depth whose recall is reported per slice. The "
                         "reranker can only reorder this list, so its recall is a hard "
                         "ceiling on anything stage 2 could recover.")
parser.add_argument("--bootstrap-iters", type=int, default=2000)
parser.add_argument("--output", default="outputs/aems/overlap_stratified.json")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
set_seeds(args.seed)


records = load_records(args.manifest, args.split)
sources = BranchSources.load(records, args.split)
video_ids = sources.usable([v for v, r in records.items() if questions(r)])
texts, rows = flatten_questions(records, video_ids)
idx, gt = query_rows(rows, video_ids, DEVICE)
query_texts = [texts[i] for i in idx.tolist()]
print(f"[DATA] {len(query_texts)} queries over {len(video_ids)} {args.split} videos", flush=True)

clip_q, dense_q = encode_queries(query_texts, DEVICE)
sims = build_sims(sources, video_ids, clip_q.to(DEVICE), query_texts, dense_q.to(DEVICE), DEVICE)
w = fixed_weights().to(DEVICE)
fused = stage1.fuse(w.unsqueeze(0) if w.dim() == 1 else w, sims)
ranks = ground_truth_ranks(fused, gt)
branch_ranks = {b: ground_truth_ranks(sims[i], gt) for i, b in enumerate(BRANCHES)}

source_words = {v: source_pool(records[v]) for v in video_ids}
owners = [video_ids[int(g)] for g in gt.tolist()]
overlap = np.array([
    len(content_words(q) & source_words[o]) / max(1, len(content_words(q)))
    for q, o in zip(query_texts, owners)])

edges = sorted(set(args.edges))
slices = [("all", np.ones_like(overlap, dtype=bool)), ("overlap = 0", overlap == 0)]
lo = 0.0
for hi in [e for e in edges if e > 0] + [1.01]:
    slices.append((f"overlap ({lo:g}, {hi:g}]", (overlap > lo) & (overlap <= hi)))
    lo = hi

print(f"\n{'slice':<22} {'n':>5}  {'R@1':>18}  {'R@10':>7}  {'MRR':>7}  "
      f"{'recall@' + str(args.rerank_top_k):>10}")
print("-" * 84)
out = {}
for name, mask in slices:
    if mask.sum() == 0:
        continue
    sel = ranks[torch.tensor(mask, device=DEVICE)]
    m = metrics_from_ranks(sel)
    low, high = bootstrap_ci(hits_at_k(sel, 1), iters=args.bootstrap_iters, seed=args.seed)
    per_branch = {b: metrics_from_ranks(branch_ranks[b][torch.tensor(mask, device=DEVICE)])["R@1"]
                  for b in BRANCHES}
    shortlist = float((sel <= args.rerank_top_k).float().mean())
    out[name] = {"n": int(mask.sum()), **m, "R@1_CI": [low, high],
                 f"recall@{args.rerank_top_k}": shortlist, "per_branch_R@1": per_branch}
    print(f"{name:<22} {int(mask.sum()):>5}  {m['R@1']:.4f} [{low:.4f},{high:.4f}]  "
          f"{m['R@10']:>7.4f}  {m['MRR']:>7.4f}  {shortlist:>10.4f}")
    print("        " + "  ".join(f"{b}={per_branch[b]:.3f}" for b in BRANCHES))

best = max(out["overlap = 0"]["per_branch_R@1"].items(), key=lambda kv: kv[1])
zero = out["overlap = 0"]
print(f"\nWith no lexical overlap the strongest branch is {best[0]} at {best[1]:.4f} "
      f"(aggregate R@1 is {out['all']['R@1']:.4f}).")
print(f"Only {zero[f'recall@{args.rerank_top_k}']:.1%} of those queries have their answer "
      f"in the top {args.rerank_top_k}, so stage 2 cannot recover them either: a reranker "
      f"reorders the shortlist, it does not repair it.")

os.makedirs(os.path.dirname(args.output), exist_ok=True)
with open(args.output, "w") as f:
    json.dump({"split": args.split, "n_videos": len(video_ids),
               "mean_overlap": float(overlap.mean()), "slices": out}, f, indent=2)
print(f"[SAVE] {args.output}")
