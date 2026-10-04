"""Pick the visual fusion weight for temporal grounding, on validation.

With an untuned weight of 0.5 the fused method scored *below* dense alone
(0.1428 against 0.1548 at IoU 0.5) — the same failure as reading an audio query
as a CLIP query: a near-chance signal given a weight it has not earned. Rather
than conclude from that that visual evidence cannot help, the weight is chosen
on validation moments carved from the train split and then applied unchanged to
test.

A tuned weight of 0 is a legitimate outcome and should be reported as one: it
says the visual branch has nothing to add to this task.
"""

import argparse, json, os
import numpy as np
import torch

from src.config import AEMS_MANIFEST_PATH, AEMS_FRAME_EMBEDDINGS_PATH, DEVICE, set_seeds
from src.data.timecodes import MAX_SEGMENTS
from src.evaluation.moments import encode_side, load_frames, score_moments

parser = argparse.ArgumentParser(description="Tune the visual weight for grounding")
parser.add_argument("--manifest", default=AEMS_MANIFEST_PATH)
parser.add_argument("--moments", default="data/processed/aems/metadata/aems_moments_val.json")
parser.add_argument("--dense-frames", default="embeddings/aems_dense_frames_val.pt")
parser.add_argument("--frame-embeddings", default=AEMS_FRAME_EMBEDDINGS_PATH)
parser.add_argument("--n-frames", type=int, default=16)
parser.add_argument("--max-segments", type=int, default=MAX_SEGMENTS)
parser.add_argument("--weights", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0])
parser.add_argument("--select-on", default="IoU@0.5", choices=["IoU@0.5", "mIoU"])
parser.add_argument("--output", default="outputs/aems/moment_fusion_tuning.json")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
set_seeds(args.seed)

bench = json.load(open(args.moments))
records = {r["video_id"]: r for r in json.load(open(args.manifest))
           if r.get("split") == "train"}
moments = [m for m in bench["moments"] if m["video_id"] in records]
print(f"[DATA] {len(moments)} validation moments over "
      f"{len({m['video_id'] for m in moments})} videos (test untouched)", flush=True)

frame_db, frame_time_db, interval = load_frames(args.dense_frames, args.frame_embeddings,
                                                args.n_frames)
moments = [m for m in moments if m["video_id"] in frame_db]
print(f"[FRAMES] {'dense every %gs' % interval if interval else 'uniform'} over "
      f"{len(frame_db)} videos; {len(moments)} moments have frames", flush=True)

q_clip, q_dense, seg_emb, seg_index, video_segs = encode_side(moments, records, DEVICE)

rows = {}
for w in args.weights:
    results, frame_hit, _ = score_moments(
        moments, records, seg_emb, seg_index, video_segs, q_clip, q_dense,
        frame_db, frame_time_db, DEVICE, max_segments=args.max_segments,
        visual_weight=w, n_frames=args.n_frames,
        methods=("dense", "visual", "dense+visual"))
    fused = results["dense+visual"]
    rows[w] = {"mIoU": float(fused.mean()), "IoU@0.5": float((fused >= 0.5).mean())}
    print(f"  visual weight {w:>4g}:  mIoU={rows[w]['mIoU']:.4f}  "
          f"IoU>=0.5={rows[w]['IoU@0.5']:.4f}", flush=True)
    if w == args.weights[0]:
        base = {"dense_mIoU": float(results["dense"].mean()),
                "dense_IoU@0.5": float((results["dense"] >= 0.5).mean()),
                "visual_mIoU": float(results["visual"].mean()),
                "visual_IoU@0.5": float((results["visual"] >= 0.5).mean()),
                "frac_with_frame": float(frame_hit.mean())}
        print(f"  (dense alone IoU>=0.5={base['dense_IoU@0.5']:.4f}; "
              f"visual alone {base['visual_IoU@0.5']:.4f}; "
              f"{base['frac_with_frame']:.1%} of spans contain a frame)", flush=True)

best = max(rows, key=lambda w: rows[w][args.select_on])
print(f"\n[BEST] visual weight {best:g} on {args.select_on}="
      f"{rows[best][args.select_on]:.4f} (dense alone "
      f"{base['dense_IoU@0.5'] if args.select_on == 'IoU@0.5' else base['dense_mIoU']:.4f})")
if best == 0.0:
    print("[NOTE] zero is the tuned value: on validation the visual branch adds "
          "nothing to transcript matching for this task.")

os.makedirs(os.path.dirname(args.output), exist_ok=True)
with open(args.output, "w") as f:
    json.dump({"best_visual_weight": float(best), "select_on": args.select_on,
               "sweep": {str(k): v for k, v in rows.items()}, "baselines": base,
               "n_moments": len(moments)}, f, indent=2)
print(f"[SAVE] {args.output}")
