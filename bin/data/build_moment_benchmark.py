"""Build a temporal grounding benchmark from the timecoded transcripts.

AEMS ships no temporal annotations, so the task does not exist until labels do.
Each QA pair gets a span: the run of consecutive transcript segments best
covering the **answer**'s content words. The answer is privileged — used to
build the benchmark, never visible to a method, which sees only the question.
Rationale and limits are in `src/data/timecodes.py`.

Two properties are reported because they decide whether the benchmark is worth
anything:

* **question overlap with the labelled window.** The question partly points at
  its own span; if that were high the task would be trivially solvable by text
  matching and the labels would be worthless.
* **frames inside the span.** At 16 uniformly sampled frames per video there
  are ~20 s between frames while a moment is ~8 s, so a large fraction of spans
  contain no frame at all. That is a hard ceiling on any frame-based localiser
  and has to be known before concluding anything about visual evidence.
"""

import argparse, json, os
import numpy as np

from src.config import AEMS_MANIFEST_PATH, set_seeds
from src.data.query_subsets import content_words, source_pool, overlap, tiers_for
from src.data.timecodes import MAX_SEGMENTS, frames_inside, locate, segments

parser = argparse.ArgumentParser(description="Build the moment-retrieval benchmark")
parser.add_argument("--manifest", default=AEMS_MANIFEST_PATH)
parser.add_argument("--split", default="test", choices=["train", "test"])
parser.add_argument("--min-coverage", type=float, default=0.3,
                    help="smallest share of the answer's words the window must cover "
                         "for the label to be kept")
parser.add_argument("--max-segments", type=int, default=MAX_SEGMENTS)
parser.add_argument("--min-video-segments", type=int, default=5,
                    help="videos with fewer segments cannot support localisation")
parser.add_argument("--max-span-fraction", type=float, default=0.5,
                    help="drop labels covering more of the video than this; such a "
                         "span is a section, not a moment, and is trivially hit")
parser.add_argument("--n-frames", type=int, default=16)
parser.add_argument("--output",
                    default="data/processed/aems/metadata/aems_moments_{split}.json")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
set_seeds(args.seed)

records = [r for r in json.load(open(args.manifest)) if r.get("split") == args.split]
print(f"[DATA] {len(records)} {args.split} videos", flush=True)

moments, skipped = [], {"no_segments": 0, "no_label": 0, "low_coverage": 0, "too_long": 0}
for record in records:
    segs = segments(record)
    if len(segs) < args.min_video_segments:
        skipped["no_segments"] += 1
        continue
    duration = float(record.get("duration_seconds") or 0.0)
    pool = source_pool(record)
    answers = record.get("qa_answers") or []
    for i, question in enumerate(record.get("qa_questions") or []):
        answer = answers[i] if i < len(answers) else ""
        found = locate(segs, answer, max_segments=args.max_segments)
        if found is None:
            skipped["no_label"] += 1
            continue
        score, start, end, lo, hi = found
        if score < args.min_coverage:
            skipped["low_coverage"] += 1
            continue
        if duration > 0 and (end - start) / duration > args.max_span_fraction:
            skipped["too_long"] += 1
            continue
        window_words = set()
        for _, _, text in segs[lo:hi]:
            window_words |= content_words(text)
        q_words = content_words(question)
        q_ov = overlap(question, pool)
        a_ov = overlap(answer, pool)
        moments.append({
            "video_id": record["video_id"], "question": question, "answer": answer,
            "start": round(start, 3), "end": round(end, 3),
            "duration": round(duration, 3),
            "answer_coverage": round(score, 4),
            "question_overlap_window": round(
                len(q_words & window_words) / len(q_words), 4) if q_words else 0.0,
            "question_overlap_source": round(q_ov, 4) if q_ov is not None else None,
            "n_segments": len(segs),
            "frames_inside": frames_inside((start, end), duration, args.n_frames),
            "tiers": tiers_for(q_ov, a_ov, False) if (q_ov is not None and a_ov is not None) else [],
            "category": record.get("content_parent_category"),
        })

print(f"[SKIP] {skipped}")
print(f"[BUILD] {len(moments)} labelled moments over "
      f"{len({m['video_id'] for m in moments})} videos", flush=True)

spans = np.array([m["end"] - m["start"] for m in moments])
fracs = np.array([(m["end"] - m["start"]) / m["duration"] for m in moments if m["duration"] > 0])
q_ov = np.array([m["question_overlap_window"] for m in moments])
cov = np.array([m["answer_coverage"] for m in moments])
inside = np.array([m["frames_inside"] for m in moments])

print(f"\nspan length      median {np.median(spans):6.1f}s   p90 {np.percentile(spans, 90):6.1f}s")
print(f"span / video     median {np.median(fracs):6.3f}    p90 {np.percentile(fracs, 90):6.3f}")
print(f"answer coverage  median {np.median(cov):6.3f}")
print(f"question overlap median {np.median(q_ov):6.3f}  "
      f"(vs answer {np.median(cov):.3f} — labels are not given away by the question)")
print(f"\nframes inside the span, at {args.n_frames} frames/video:")
for k in (0, 1, 2):
    print(f"  exactly {k}: {(inside == k).mean():6.1%}")
print(f"  >= 1      : {(inside >= 1).mean():6.1%}   <- ceiling on any frame-based localiser")

out_path = args.output.format(split=args.split)
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump({"split": args.split,
               "config": {"min_coverage": args.min_coverage,
                          "max_segments": args.max_segments,
                          "max_span_fraction": args.max_span_fraction,
                          "n_frames": args.n_frames},
               "stats": {"n_moments": len(moments),
                         "n_videos": len({m["video_id"] for m in moments}),
                         "median_span_s": float(np.median(spans)),
                         "median_span_fraction": float(np.median(fracs)),
                         "median_question_overlap": float(np.median(q_ov)),
                         "frame_coverage": float((inside >= 1).mean()),
                         "skipped": skipped},
               "moments": moments}, f, indent=2)
print(f"\n[SAVE] {out_path}")
