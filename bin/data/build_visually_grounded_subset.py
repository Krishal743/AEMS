"""Carve out the test queries that actually require watching the video.

The aggregate benchmark is dominated by lexical overlap: R@1 is 0.965 where a
question shares most of its words with its own transcript and 0.035 where it
shares none (`bin/evaluation/eval_overlap_stratified.py`). The questions are
also mostly *about* visual detail while being answerable from text, so the
benchmark rewards the wrong behaviour.

This builds a subset where that shortcut is unavailable, using two independent
signals rather than keywords alone:

* **answer grounding** — the manifest carries a written answer for every
  question. If the answer's content words are absent from the transcript and
  description, the answer is not in the text and the question cannot be
  answered from it. This is the stronger signal and the main filter.
* **question overlap** — how much of the question's own wording appears in its
  source text. Low overlap removes the lexical-matching shortcut at retrieval
  time, independently of whether the answer is in the text.
* **visual markers** — a keyword pattern (colour, count, on-screen text,
  clothing, position). This is a crude heuristic and is used only to label a
  high-precision tier, never on its own.

Tiers emitted:

| tier | rule | purpose |
|---|---|---|
| `text_blind` | question overlap <= `--max-question-overlap` | no lexical shortcut |
| `answer_grounded` | answer overlap <= `--max-answer-overlap` | answer absent from text |
| `visually_grounded` | both, plus a visual marker | the strict slice |

A random sample is written out for manual verification, because the thresholds
and the marker list are judgement calls and the subset is only trustworthy to
the extent someone has read some of it.
"""

import argparse, json, os, re
import numpy as np

from src.config import AEMS_MANIFEST_PATH, set_seeds
from src.data.query_subsets import (VISUAL_MARKERS, VISUALLY_GROUNDED, TEXT_BLIND,
                                    ANSWER_GROUNDED, content_words, overlap,
                                    source_pool, tiers_for)

parser = argparse.ArgumentParser(description="Build the visually-grounded query subset")
parser.add_argument("--manifest", default=AEMS_MANIFEST_PATH)
parser.add_argument("--split", default="test", choices=["train", "test"])
parser.add_argument("--max-question-overlap", type=float, default=0.3)
parser.add_argument("--max-answer-overlap", type=float, default=0.3)
parser.add_argument("--verify-sample", type=int, default=300,
                    help="queries written out for manual checking")
parser.add_argument("--output",
                    default="data/processed/aems/metadata/aems_query_subsets_{split}.json")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
set_seeds(args.seed)


records = json.load(open(args.manifest))
records = [r for r in records if r.get("split") == args.split]
print(f"[DATA] {len(records)} {args.split} videos", flush=True)

queries = []
for record in records:
    pool = source_pool(record)
    answers = record.get("qa_answers") or []
    for i, question in enumerate(record.get("qa_questions") or []):
        answer = answers[i] if i < len(answers) else ""
        q_ov, a_ov = overlap(question, pool), overlap(answer, pool)
        if q_ov is None or a_ov is None:
            continue
        queries.append({
            "video_id": record["video_id"], "question": question, "answer": answer,
            "question_overlap": round(q_ov, 4), "answer_overlap": round(a_ov, 4),
            "visual_marker": bool(VISUAL_MARKERS.search(question)),
            "category": record.get("content_parent_category"),
        })

text_blind = [q for q in queries if q["question_overlap"] <= args.max_question_overlap]
answer_grounded = [q for q in queries if q["answer_overlap"] <= args.max_answer_overlap]
visually_grounded = [q for q in queries
                     if q["question_overlap"] <= args.max_question_overlap
                     and q["answer_overlap"] <= args.max_answer_overlap
                     and q["visual_marker"]]
for q in queries:
    q["tiers"] = tiers_for(q["question_overlap"], q["answer_overlap"], q["visual_marker"],
                           args.max_question_overlap, args.max_answer_overlap)

print(f"[SUBSET] all                {len(queries):5d}")
print(f"[SUBSET] text_blind         {len(text_blind):5d}  "
      f"(question overlap <= {args.max_question_overlap})")
print(f"[SUBSET] answer_grounded    {len(answer_grounded):5d}  "
      f"(answer overlap <= {args.max_answer_overlap})")
print(f"[SUBSET] visually_grounded  {len(visually_grounded):5d}  (both + visual marker)")
videos = len({q["video_id"] for q in visually_grounded})
print(f"[SUBSET] visually_grounded spans {videos} distinct videos")

rng = np.random.default_rng(args.seed)
sample = [visually_grounded[i] for i in
          rng.permutation(len(visually_grounded))[:min(args.verify_sample,
                                                       len(visually_grounded))]]

out_path = args.output.format(split=args.split)
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump({"split": args.split,
               "thresholds": {"max_question_overlap": args.max_question_overlap,
                              "max_answer_overlap": args.max_answer_overlap},
               "counts": {"all": len(queries), "text_blind": len(text_blind),
                          "answer_grounded": len(answer_grounded),
                          "visually_grounded": len(visually_grounded)},
               "queries": queries}, f, indent=2)
print(f"[SAVE] {out_path}")

verify_path = out_path.replace(".json", "_to_verify.jsonl")
with open(verify_path, "w") as f:
    for q in sample:
        f.write(json.dumps({**{k: q[k] for k in ("video_id", "question", "answer")},
                            "requires_video": None}) + "\n")
print(f"[SAVE] {verify_path} — {len(sample)} queries for manual checking "
      f"(set requires_video true/false)")
