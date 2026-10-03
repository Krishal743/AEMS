"""How much of the visually-grounded result is the thresholds talking?

The subset is defined by two cut-offs (question overlap, answer overlap) plus a
keyword list, all three judgement calls. A finding that only holds at one
setting is a finding about the setting. This sweeps both cut-offs and reports,
for each, the tier size and the branch ordering the conclusion rests on:
BM25 should collapse and the visual branch should hold up, and that ordering
should be stable across the grid if it means anything.

Branch similarities are computed once over all queries and re-masked per cell,
so the sweep costs barely more than one evaluation.
"""

import argparse, json, os
import torch

from src.config import AEMS_MANIFEST_PATH, DEVICE, set_seeds
from src.data.query_subsets import VISUAL_MARKERS, source_pool, overlap, tiers_for
from src.evaluation.evaluate_retrieval import ground_truth_ranks, metrics_from_ranks
from src.rerank import stage1
from src.retrieval.branches import BranchSources, build_sims, encode_queries
from src.routing.query_router import BRANCHES, fixed_weights
from src.training.query_data import load_records, questions

parser = argparse.ArgumentParser(description="Sensitivity of the subset to its thresholds")
parser.add_argument("--manifest", default=AEMS_MANIFEST_PATH)
parser.add_argument("--split", default="test", choices=["train", "test"])
parser.add_argument("--question-overlaps", type=float, nargs="+",
                    default=[0.2, 0.3, 0.4, 0.5])
parser.add_argument("--answer-overlaps", type=float, nargs="+",
                    default=[0.2, 0.3, 0.4, 0.5])
parser.add_argument("--require-marker", action="store_true", default=True)
parser.add_argument("--no-marker", dest="require_marker", action="store_false",
                    help="drop the keyword filter, isolating the two overlap cut-offs")
parser.add_argument("--min-queries", type=int, default=50,
                    help="cells smaller than this are reported but flagged as too small")
parser.add_argument("--output", default="outputs/aems/subset_threshold_sweep.json")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
set_seeds(args.seed)

records = load_records(args.manifest, args.split)
sources = BranchSources.load(records, args.split)
video_ids = sources.usable([v for v, r in records.items() if questions(r)])
position = {v: i for i, v in enumerate(video_ids)}

rows = []
for v in video_ids:
    record = records[v]
    pool = source_pool(record)
    answers = record.get("qa_answers") or []
    for i, question in enumerate(record.get("qa_questions") or []):
        q_ov = overlap(question, pool)
        a_ov = overlap(answers[i] if i < len(answers) else "", pool)
        if q_ov is None or a_ov is None:
            continue
        rows.append({"video_id": v, "question": question, "q": q_ov, "a": a_ov,
                     "marker": bool(VISUAL_MARKERS.search(question))})
print(f"[DATA] {len(rows)} queries over {len(video_ids)} {args.split} videos", flush=True)

query_texts = [r["question"] for r in rows]
gt = torch.tensor([position[r["video_id"]] for r in rows], device=DEVICE)
clip_q, dense_q = encode_queries(query_texts, DEVICE)
sims = build_sims(sources, video_ids, clip_q.to(DEVICE), query_texts, dense_q.to(DEVICE), DEVICE)
w = fixed_weights().to(DEVICE)
ranks = ground_truth_ranks(stage1.fuse(w.unsqueeze(0) if w.dim() == 1 else w, sims), gt)
branch_ranks = {b: ground_truth_ranks(sims[i], gt) for i, b in enumerate(BRANCHES)}

marker_note = "with visual marker" if args.require_marker else "overlaps only"
print(f"\nTier = question overlap <= Q and answer overlap <= A ({marker_note})\n")
print(f"{'Q':>4} {'A':>5} {'n':>6}  {'R@1':>7}  {'visual':>7} {'bm25':>7} {'dense':>7}  "
      f"{'visual/bm25':>12}")
print("-" * 62)
out = {}
for q_t in args.question_overlaps:
    for a_t in args.answer_overlaps:
        mask = torch.tensor(
            [("visually_grounded" in tiers_for(r["q"], r["a"], r["marker"], q_t, a_t))
             if args.require_marker else (r["q"] <= q_t and r["a"] <= a_t)
             for r in rows], device=DEVICE)
        n = int(mask.sum())
        if n == 0:
            continue
        m = metrics_from_ranks(ranks[mask])
        per = {b: metrics_from_ranks(branch_ranks[b][mask])["R@1"] for b in BRANCHES}
        ratio = per["visual"] / per["bm25"] if per["bm25"] > 0 else float("inf")
        out[f"q{q_t}_a{a_t}"] = {"n": n, "R@1": m["R@1"], "per_branch_R@1": per,
                                 "visual_over_bm25": ratio}
        flag = "  (small)" if n < args.min_queries else ""
        print(f"{q_t:>4} {a_t:>5} {n:>6}  {m['R@1']:>7.4f}  {per['visual']:>7.3f} "
              f"{per['bm25']:>7.3f} {per['dense']:>7.3f}  {ratio:>12.2f}{flag}")

big = {k: v for k, v in out.items() if v["n"] >= args.min_queries}
if big:
    ratios = [v["visual_over_bm25"] for v in big.values()]
    beats = sum(1 for v in big.values() if v["per_branch_R@1"]["visual"]
                > v["per_branch_R@1"]["bm25"])
    print(f"\nAcross {len(big)} cells with >= {args.min_queries} queries: visual beats bm25 "
          f"in {beats}/{len(big)}; ratio ranges {min(ratios):.2f}-{max(ratios):.2f}.")
    print("On all queries the same ratio is "
          f"{metrics_from_ranks(branch_ranks['visual'])['R@1'] / metrics_from_ranks(branch_ranks['bm25'])['R@1']:.2f}.")

os.makedirs(os.path.dirname(args.output), exist_ok=True)
with open(args.output, "w") as f:
    json.dump({"split": args.split, "require_marker": args.require_marker,
               "n_queries": len(rows), "cells": out}, f, indent=2)
print(f"[SAVE] {args.output}")
