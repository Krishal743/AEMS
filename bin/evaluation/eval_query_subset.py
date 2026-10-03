"""Score the system on query subsets that remove the benchmark's shortcuts.

`bin/data/build_visually_grounded_subset.py` carves the test queries into tiers
by whether the answer is present in the video's text and whether the question
reuses its wording. This scores each tier against the **full** gallery, so the
task gets no easier — only the queries change.

The `visually_grounded` tier is the closest thing this dataset offers to an
honest measure of multimodal retrieval: the answer is not in the transcript and
the question shares little wording with it, so a text branch has nothing to
match against.
"""

import argparse, json, os
import torch

from src.config import AEMS_MANIFEST_PATH, DEVICE, set_seeds
from src.evaluation.evaluate_retrieval import (ground_truth_ranks, metrics_from_ranks,
                                               bootstrap_ci, hits_at_k)
from src.rerank import stage1
from src.retrieval.branches import BranchSources, build_sims, encode_queries
from src.routing.query_router import BRANCHES, fixed_weights
from src.training.query_data import load_records, questions

parser = argparse.ArgumentParser(description="Score the system on query subsets")
parser.add_argument("--manifest", default=AEMS_MANIFEST_PATH)
parser.add_argument("--subset",
                    default="data/processed/aems/metadata/aems_query_subsets_test.json")
parser.add_argument("--split", default="test", choices=["train", "test"])
parser.add_argument("--tiers", nargs="+",
                    default=["all", "text_blind", "answer_grounded", "visually_grounded"])
parser.add_argument("--weights-json", default=None,
                    help="tuning output from bin/training/tune_fusion_weights.py; its "
                         "weights are scored alongside the deployed ones")
parser.add_argument("--rerank", choices=["none", "gate", "cross", "chain"], default="none",
                    help="stage 2 to apply. Reported per tier, since recall@100 differs "
                         "sharply between tiers and bounds what stage 2 can recover.")
parser.add_argument("--rerank-top-k", type=int, default=100)
parser.add_argument("--rerank-passages", type=int, default=8)
parser.add_argument("--rerank-alpha", type=float, default=0.5)
parser.add_argument("--rerank-beta", type=float, default=0.5)
parser.add_argument("--per-candidate-gate", default=None)
parser.add_argument("--cross-encoder", default=None)
parser.add_argument("--bootstrap-iters", type=int, default=2000)
parser.add_argument("--output", default="outputs/aems/query_subset_eval.json")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
set_seeds(args.seed)

subset = json.load(open(args.subset))
records = load_records(args.manifest, args.split)
sources = BranchSources.load(records, args.split)
video_ids = sources.usable([v for v, r in records.items() if questions(r)])
position = {v: i for i, v in enumerate(video_ids)}

rows = [q for q in subset["queries"] if q["video_id"] in position]
print(f"[DATA] {len(rows)} queries over a gallery of {len(video_ids)} {args.split} videos",
      flush=True)

query_texts = [q["question"] for q in rows]
gt = torch.tensor([position[q["video_id"]] for q in rows], device=DEVICE)
clip_q, dense_q = encode_queries(query_texts, DEVICE)
sims = build_sims(sources, video_ids, clip_q.to(DEVICE), query_texts,
                  dense_q.to(DEVICE), DEVICE)
def rank_with(weights):
    w = fixed_weights(weights).to(DEVICE)
    return ground_truth_ranks(stage1.fuse(w.unsqueeze(0) if w.dim() == 1 else w, sims), gt)


fused_w = fixed_weights(None).to(DEVICE)
fused = stage1.fuse(fused_w.unsqueeze(0) if fused_w.dim() == 1 else fused_w, sims)
ranks = rank_with(None)
branch_ranks = {b: ground_truth_ranks(sims[i], gt) for i, b in enumerate(BRANCHES)}
alt_ranks, alt_weights = None, None
if args.weights_json:
    alt_weights = json.load(open(args.weights_json))["weights"]
    alt_ranks = rank_with(alt_weights)
    print(f"[WEIGHTS] also scoring {args.weights_json}: "
          + "  ".join(f"{b}={alt_weights[b]:g}" for b in BRANCHES), flush=True)

rerank_ranks = None
if args.rerank != "none":
    from src.config import (AEMS_PER_CANDIDATE_GATE_PATH, AEMS_TEXT_CHUNKS_PATH_TEMPLATE)
    from src.rerank.pipeline import rerank_batch
    from src.routing.query_router import CROSS_ENCODER_MODEL
    chunk_db = torch.load(AEMS_TEXT_CHUNKS_PATH_TEMPLATE.format(split=args.split),
                          weights_only=False)
    print(f"[RERANK] {args.rerank}: K={args.rerank_top_k}, {args.rerank_passages} passages",
          flush=True)
    reranked, _ = rerank_batch(
        args.rerank, fused, sims, clip_q, query_texts, video_ids,
        records=records, chunk_db=chunk_db,
        gate_path=args.per_candidate_gate or AEMS_PER_CANDIDATE_GATE_PATH,
        cross_encoder_name=args.cross_encoder or CROSS_ENCODER_MODEL,
        top_k=args.rerank_top_k, n_passages=args.rerank_passages,
        alpha=args.rerank_alpha, beta=args.rerank_beta, device=DEVICE)
    rerank_ranks = ground_truth_ranks(reranked, gt)

    # Re-weighting and reranking are independent interventions of similar size on
    # the visual tier; whether they compose is not answerable from either alone.
    combined_ranks = None
    if alt_weights is not None:
        alt_w = fixed_weights(alt_weights).to(DEVICE)
        alt_fused = stage1.fuse(alt_w.unsqueeze(0) if alt_w.dim() == 1 else alt_w, sims)
        combined, _ = rerank_batch(
            args.rerank, alt_fused, sims, clip_q, query_texts, video_ids,
            records=records, chunk_db=chunk_db,
            gate_path=args.per_candidate_gate or AEMS_PER_CANDIDATE_GATE_PATH,
            cross_encoder_name=args.cross_encoder or CROSS_ENCODER_MODEL,
            top_k=args.rerank_top_k, n_passages=args.rerank_passages,
            alpha=args.rerank_alpha, beta=args.rerank_beta, device=DEVICE)
        combined_ranks = ground_truth_ranks(combined, gt)
else:
    combined_ranks = None

print(f"\n{'tier':<20} {'n':>5}  {'R@1':>18}  {'R@10':>7}  {'MRR':>7}  "
      f"{'recall@' + str(args.rerank_top_k):>10}")
print("-" * 82)
out = {}
for tier in args.tiers:
    mask = torch.tensor([tier == "all" or tier in q["tiers"] for q in rows], device=DEVICE)
    if not bool(mask.any()):
        continue
    sel = ranks[mask]
    m = metrics_from_ranks(sel)
    lo, hi = bootstrap_ci(hits_at_k(sel, 1), iters=args.bootstrap_iters, seed=args.seed)
    shortlist = float((sel <= args.rerank_top_k).float().mean())
    per_branch = {b: metrics_from_ranks(branch_ranks[b][mask])["R@1"] for b in BRANCHES}
    out[tier] = {"n": int(mask.sum()), **m, "R@1_CI": [lo, hi],
                 f"recall@{args.rerank_top_k}": shortlist, "per_branch_R@1": per_branch}
    print(f"{tier:<20} {int(mask.sum()):>5}  {m['R@1']:.4f} [{lo:.4f},{hi:.4f}]  "
          f"{m['R@10']:>7.4f}  {m['MRR']:>7.4f}  {shortlist:>10.4f}")
    print("        " + "  ".join(f"{b}={per_branch[b]:.3f}" for b in BRANCHES))
    if rerank_ranks is not None:
        from src.evaluation.evaluate_retrieval import paired_bootstrap
        rm = metrics_from_ranks(rerank_ranks[mask])
        diff, dlo, dhi = paired_bootstrap(hits_at_k(rerank_ranks[mask], 1),
                                          hits_at_k(sel, 1), seed=args.seed)
        out[tier][f"R@1_rerank_{args.rerank}"] = rm["R@1"]
        out[tier]["paired_delta_rerank"] = [float(diff), float(dlo), float(dhi)]
        flag = "sig" if dlo > 0 or dhi < 0 else "ns"
        print(f"{'  + ' + args.rerank:<20} {' ':>5}  {rm['R@1']:.4f}{' ':>14}  "
              f"{rm['R@10']:>7.4f}  {rm['MRR']:>7.4f}   Δ={diff:+.4f} "
              f"[{dlo:+.4f},{dhi:+.4f}] {flag}")
    if combined_ranks is not None:
        from src.evaluation.evaluate_retrieval import paired_bootstrap
        cm = metrics_from_ranks(combined_ranks[mask])
        diff, dlo, dhi = paired_bootstrap(hits_at_k(combined_ranks[mask], 1),
                                          hits_at_k(sel, 1), seed=args.seed)
        out[tier]["R@1_tuned_plus_rerank"] = cm["R@1"]
        flag = "sig" if dlo > 0 or dhi < 0 else "ns"
        print(f"{'  tuned + ' + args.rerank:<20} {' ':>5}  {cm['R@1']:.4f}{' ':>14}  "
              f"{cm['R@10']:>7.4f}  {cm['MRR']:>7.4f}   Δ={diff:+.4f} "
              f"[{dlo:+.4f},{dhi:+.4f}] {flag}")
    if alt_ranks is not None:
        from src.evaluation.evaluate_retrieval import paired_bootstrap
        am = metrics_from_ranks(alt_ranks[mask])
        diff, dlo, dhi = paired_bootstrap(hits_at_k(sel, 1),
                                          hits_at_k(alt_ranks[mask], 1), seed=args.seed)
        out[tier]["R@1_tuned_weights"] = am["R@1"]
        out[tier]["paired_delta_deployed_minus_tuned"] = [float(diff), float(dlo), float(dhi)]
        flag = "sig" if dlo > 0 or dhi < 0 else "ns"
        print(f"{'  tuned weights':<20} {' ':>5}  {am['R@1']:.4f}{' ':>14}  "
              f"{am['R@10']:>7.4f}  {am['MRR']:>7.4f}   Δ={-diff:+.4f} "
              f"[{-dhi:+.4f},{-dlo:+.4f}] {flag}")

if "visually_grounded" in out:
    vg = out["visually_grounded"]
    best = max(vg["per_branch_R@1"].items(), key=lambda kv: kv[1])
    print(f"\nOn the visually-grounded tier the strongest single branch is {best[0]} "
          f"at {best[1]:.4f}; fused R@1 is {vg['R@1']:.4f} against {out['all']['R@1']:.4f} "
          f"on all queries.")

os.makedirs(os.path.dirname(args.output), exist_ok=True)
with open(args.output, "w") as f:
    json.dump({"split": args.split, "subset": args.subset,
               "gallery_size": len(video_ids), "tiers": out}, f, indent=2)
print(f"[SAVE] {args.output}")
