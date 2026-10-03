"""Benchmark non-text queries: image, video, audio and mixed.

Every headline number the project reports comes from text queries — all 5,097
test queries are QA questions — while the image, video, audio and mixed query
paths in `src/routing/query_router.py` went unmeasured. This measures them
against one shared gallery.

**Leakage is the whole difficulty.** Querying with material already in the index
scores a trivial 1.0 and measures nothing, so each modality is held out in time
(`src/evaluation/holdout.py`). Disjoint is not the same as independent: frames
sampled uniformly are near-duplicates of their neighbours, so an even/odd split
is formally clean and still inflates image R@1 by 7.5 points. The temporal split
(first half indexes, second half queries) is the default; `--diagnose-frames`
quantifies the difference.

These rows measure **query-by-example** — a clip cut from the target video,
matched back to its source — not semantic retrieval, and they are not
comparable to the headline text numbers. The `text` row here is the matched
baseline, run on the same reduced gallery with one question per video.

`--weights modality` applies the per-modality weights from
`bin/training/tune_modality_weights.py`; `--weights deployed` uses the
text-tuned weights for everything, which is what the router did before.
"""

import argparse, json, os
import torch

from src.config import (AEMS_MANIFEST_PATH, AEMS_FUSION_WEIGHTS, AEMS_AUDIO_CLIP_SEC,
                        DEVICE, set_seeds)
from src.evaluation.evaluate_retrieval import (ground_truth_ranks, metrics_from_ranks,
                                               bootstrap_ci, hits_at_k, paired_bootstrap)
from src.evaluation.holdout import SPLITS, TEMPORAL, frame_holdout
from src.evaluation.multimodal_queries import (MODALITIES, Gallery, audio_queries,
                                               build_query, fuse, text_queries)
from src.retrieval.branches import BranchSources
from src.routing.query_router import BRANCHES, chunk_index_from, modality_weights
from src.training.query_data import load_records, questions

parser = argparse.ArgumentParser(description="Benchmark image/video/audio/mixed queries")
parser.add_argument("--manifest", default=AEMS_MANIFEST_PATH)
parser.add_argument("--split", default="test", choices=["train", "test"])
parser.add_argument("--modalities", nargs="+", default=list(MODALITIES))
parser.add_argument("--weights", default="modality", choices=["modality", "deployed"])
parser.add_argument("--compare-weights", action="store_true",
                    help="score each modality under both weightings, with a paired CI")
parser.add_argument("--frame-split", default=TEMPORAL, choices=list(SPLITS))
parser.add_argument("--image-frame", type=int, default=4)
parser.add_argument("--per-branch", action="store_true",
                    help="also score each reachable branch alone, which separates "
                         "appearance matching from cross-modal semantic matching")
parser.add_argument("--diagnose-frames", action="store_true")
parser.add_argument("--bootstrap", action="store_true")
parser.add_argument("--bootstrap-iters", type=int, default=2000)
parser.add_argument("--output", default="outputs/aems/multimodal_query_benchmark.json")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
set_seeds(args.seed)

records = load_records(args.manifest, args.split)
sources = BranchSources.load(records, args.split)
video_ids = sources.usable([v for v, r in records.items() if questions(r)])
gallery = Gallery(records, sources, video_ids, DEVICE, frame_split=args.frame_split)
n = gallery.n
print(f"[DATA] gallery = {n} {args.split} videos; visual branch rebuilt from "
      f"{len(next(iter(gallery.index_frames.values())))} frames per video "
      f"({args.frame_split} split)", flush=True)

if args.diagnose_frames:
    print("[DIAG] query frame vs its own gallery frames, against the best other video:",
          flush=True)
    for split in SPLITS:
        held = {v: frame_holdout(f, split) for v, f in gallery.frames.items()}
        idx = gallery._on(chunk_index_from({v: i for v, (i, _) in held.items()}, video_ids))
        q = torch.stack([held[v][1][args.image_frame] for v in video_ids]).to(DEVICE)
        sims = idx.max_sim_batch(torch.nn.functional.normalize(q, dim=1))
        d = torch.arange(n, device=DEVICE)
        own = sims[d, d]
        other = sims.clone()
        other[d, d] = -1e4
        best = other.max(dim=1).values
        print(f"  {split:<11} own={own.mean():.4f}  best other={best.mean():.4f}  "
              f"margin={(own - best).mean():+.4f}", flush=True)

text = text_queries(records, video_ids, DEVICE) if (
    {"text", "image+text"} & set(args.modalities)) else None
audio = audio_queries(records, video_ids, DEVICE) if (
    {"audio", "audio_clip"} & set(args.modalities)) else None


def weights_for(modality, scheme):
    return AEMS_FUSION_WEIGHTS if scheme == "deployed" else modality_weights(modality)


print(f"\n{'query modality':<14} {'n':>5}  {'R@1':>7} {'R@5':>7} {'R@10':>7} {'MRR':>7}  branches")
print("-" * 90)
out = {}
for modality in args.modalities:
    sims, gt = build_query(modality, gallery, records, video_ids, DEVICE,
                           image_frame=args.image_frame, audio=audio, text=text)
    reachable = [b for b, s in zip(BRANCHES, sims) if s is not None]
    ranks = ground_truth_ranks(fuse(sims, weights_for(modality, args.weights), DEVICE), gt)
    m = metrics_from_ranks(ranks)
    entry = {"n_queries": int(ranks.numel()), "branches": reachable,
             "weights": args.weights, **m}
    if args.bootstrap:
        lo, hi = bootstrap_ci(hits_at_k(ranks, 1), iters=args.bootstrap_iters, seed=args.seed)
        entry["R@1_CI"] = [lo, hi]
    ci = f"  [{entry['R@1_CI'][0]:.4f}, {entry['R@1_CI'][1]:.4f}]" if args.bootstrap else ""
    print(f"{modality:<14} {entry['n_queries']:>5}  {m['R@1']:>7.4f} {m['R@5']:>7.4f} "
          f"{m['R@10']:>7.4f} {m['MRR']:>7.4f}  {'+'.join(reachable)}{ci}")

    if args.compare_weights:
        other = "deployed" if args.weights == "modality" else "modality"
        r2 = ground_truth_ranks(fuse(sims, weights_for(modality, other), DEVICE), gt)
        m2 = metrics_from_ranks(r2)
        diff, lo, hi = paired_bootstrap(hits_at_k(ranks, 1), hits_at_k(r2, 1), seed=args.seed)
        entry[f"R@1_{other}_weights"] = m2["R@1"]
        entry["paired_delta"] = [float(diff), float(lo), float(hi)]
        flag = "sig" if lo > 0 or hi < 0 else "ns"
        print(f"{'  vs ' + other:<14} {' ':>5}  {m2['R@1']:>7.4f}  "
              f"Δ={diff:+.4f} [{lo:+.4f}, {hi:+.4f}] {flag}")

    if args.per_branch and len(reachable) > 1:
        entry["per_branch"] = {}
        for b in reachable:
            bm = metrics_from_ranks(ground_truth_ranks(sims[BRANCHES.index(b)], gt))
            entry["per_branch"][b] = bm
            print(f"{'  └ ' + b:<14} {entry['n_queries']:>5}  {bm['R@1']:>7.4f} "
                  f"{bm['R@5']:>7.4f} {bm['R@10']:>7.4f} {bm['MRR']:>7.4f}")
    out[modality] = entry

os.makedirs(os.path.dirname(args.output), exist_ok=True)
with open(args.output, "w") as f:
    json.dump({"split": args.split, "gallery_size": n,
               "protocol": {"frame_split": args.frame_split, "image_frame": args.image_frame,
                            "audio_clip_sec": AEMS_AUDIO_CLIP_SEC, "weights": args.weights},
               "results": out}, f, indent=2)
print(f"\n[SAVE] {args.output}")
