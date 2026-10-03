"""Tune fusion weights separately for each query modality.

The deployed weights were tuned on text queries, where the audio branch is
nearly worthless and is weighted accordingly. Those weights are then applied to
every query, and the multimodal benchmark showed what that costs: an audio query
read as a CLIP query scores 0.2201 while its audio branch alone scores 0.9131,
because renormalizing text-tuned weights over the reachable branches hands the
audio branch 14% of the mass and three near-chance branches the other 86%.

The router already knows which branches a query can *reach*. It has no notion of
which branches are *reliable* for that kind of query, which is what this fixes:
one weight vector per modality, each tuned by the same coordinate ascent used
for the text weights.

Selection is on validation videos carved from the **train** split; the test
split is untouched here and is where `eval_multimodal_queries.py` applies the
result.
"""

import argparse, json, os
import torch

from src.config import AEMS_MANIFEST_PATH, AEMS_FUSION_WEIGHTS, DEVICE, set_seeds
from src.evaluation.evaluate_retrieval import ground_truth_ranks, metrics_from_ranks
from src.evaluation.holdout import SPLITS, TEMPORAL
from src.evaluation.multimodal_queries import (MODALITIES, Gallery, audio_queries,
                                               build_query, fuse, text_queries)
from src.retrieval.branches import BranchSources
from src.routing.query_router import BRANCHES
from src.training.query_data import load_records, questions, validation_split

parser = argparse.ArgumentParser(description="Tune fusion weights per query modality")
parser.add_argument("--manifest", default=AEMS_MANIFEST_PATH)
parser.add_argument("--modalities", nargs="+", default=list(MODALITIES))
parser.add_argument("--val-frac", type=float, default=0.15)
parser.add_argument("--frame-split", default=TEMPORAL, choices=list(SPLITS))
parser.add_argument("--image-frame", type=int, default=4)
parser.add_argument("--sweeps", type=int, default=3)
parser.add_argument("--grid", type=float, nargs="+",
                    default=[0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0])
parser.add_argument("--floor", type=float, default=0.0,
                    help="smallest weight a reachable branch may take. Unlike the text "
                         "weights, zero is allowed here: a branch that is near-chance "
                         "for a modality is noise, not a weak signal.")
parser.add_argument("--output", default="outputs/aems/modality_weight_tuning.json")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
set_seeds(args.seed)

records = load_records(args.manifest, "train")
sources = BranchSources.load(records, "train")
usable = sources.usable([v for v, r in records.items() if questions(r)])
_, val_vids = validation_split(usable, args.val_frac)
print(f"[DATA] tuning on {len(val_vids)} validation videos from train "
      f"(test untouched)", flush=True)

gallery = Gallery(records, sources, val_vids, DEVICE, frame_split=args.frame_split)
text = text_queries(records, val_vids, DEVICE) if (
    {"text", "image+text"} & set(args.modalities)) else None
audio = audio_queries(records, val_vids, DEVICE) if (
    {"audio", "audio_clip"} & set(args.modalities)) else None

tuned, summary = {}, {}
for modality in args.modalities:
    sims, gt = build_query(modality, gallery, records, val_vids, DEVICE,
                           image_frame=args.image_frame, audio=audio, text=text)
    reachable = [b for b, s in zip(BRANCHES, sims) if s is not None]

    def score(weights):
        return metrics_from_ranks(
            ground_truth_ranks(fuse(sims, weights, DEVICE), gt))["R@1"]

    weights = {b: max(AEMS_FUSION_WEIGHTS[b], args.floor) for b in BRANCHES}
    baseline = score(weights)
    best = baseline
    for sweep in range(args.sweeps):
        improved = False
        for b in reachable:          # unreachable branches cannot matter
            start = weights[b]
            for value in [max(v, args.floor) for v in args.grid]:
                if value == start:
                    continue
                trial = dict(weights, **{b: value})
                if not any(trial[x] for x in reachable):
                    continue
                r1 = score(trial)
                if r1 > best + 1e-6:
                    best, weights, improved = r1, trial, True
        if not improved:
            break

    tuned[modality] = {b: weights[b] for b in BRANCHES}
    summary[modality] = {"val_R@1_text_weights": baseline, "val_R@1_tuned": best,
                         "gain": best - baseline, "reachable": reachable}
    print(f"  {modality:<12} val R@1 {baseline:.4f} -> {best:.4f} "
          f"({best - baseline:+.4f})   "
          + "  ".join(f"{b}={weights[b]:g}" for b in reachable), flush=True)

print("\n[WEIGHTS] AEMS_MODALITY_WEIGHTS = {")
for modality in args.modalities:
    inner = ", ".join(f'"{b}": {tuned[modality][b]:g}' for b in BRANCHES)
    print(f'    "{modality}": {{{inner}}},')
print("}")

os.makedirs(os.path.dirname(args.output), exist_ok=True)
with open(args.output, "w") as f:
    json.dump({"weights": tuned, "summary": summary, "val_videos": len(val_vids),
               "frame_split": args.frame_split, "floor": args.floor,
               "seed": args.seed}, f, indent=2)
print(f"[SAVE] {args.output}")
