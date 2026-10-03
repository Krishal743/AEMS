"""Benchmark non-text queries: image, video, audio and mixed.

Every number the project reports comes from text queries — all 5,097 test
queries are QA questions — while the image, video, audio and mixed query paths
in `src/routing/query_router.py` have never been measured. This script measures
them, against the same gallery, so "any-to-any retrieval" can be stated with
evidence or corrected.

**Leakage is the whole difficulty.** Querying with material that is already in
the index scores a trivial 1.0 and measures nothing. Each modality is therefore
held out in time:

* **image / video** — each video has 16 indexed frames, and the gallery is
  rebuilt from half of them so no frame is ever both query and index. *Which*
  half matters enormously. Under `--frame-split interleave` (even indexes the
  gallery, odd queries it) every query frame sits temporally adjacent to an
  indexed one and is very nearly a duplicate of it, which measures
  near-duplicate matching rather than retrieval. Under `--frame-split temporal`
  (default) the first half of the video indexes and the second half queries, so
  query and gallery frames are separated by half the runtime. The two settings
  differ by a large margin; `--diagnose-frames` quantifies why.
* **audio** — the indexed embedding averages three fixed 10 s segments (start,
  middle, end). Queries use a 10 s segment at the 25% mark, which is disjoint
  from all three provided the video runs at least 60 s, so shorter videos are
  excluded from the audio rows (not from the gallery).

The gallery is identical for every modality, including the reduced visual index,
so rows are comparable with each other. They are *not* directly comparable to
the headline text numbers, which use the full 16-frame index and all questions;
the `text` row here is the matched baseline for that reason.

Branch reachability is not uniform, and that is a finding rather than a defect:
a CLIP-space query (image/video) cannot reach BM25 or the dense retriever, which
need query *text*, so it is scored on three branches out of six.
"""

import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F

from src.config import (AEMS_MANIFEST_PATH, AEMS_AUDIO_ADAPTER_PATH, AEMS_AUDIO_DIR,
                        AEMS_WAVLM_SR, AEMS_AUDIO_CLIP_SEC, DEVICE, set_seeds)
from src.data.text_chunks import lexical_fields
from src.evaluation.evaluate_retrieval import (ground_truth_ranks, metrics_from_ranks,
                                               bootstrap_ci, hits_at_k)
from src.evaluation.holdout import SPLITS, frame_holdout, audio_segment_is_disjoint
from src.models.audio_adapter import load_audio_adapter
from src.retrieval.bm25 import BM25PassageIndex
from src.retrieval.branches import BranchSources, encode_queries
from src.rerank import stage1
from src.routing.query_router import BRANCHES, chunk_index_from, fixed_weights, zscore
from src.training.query_data import load_records, questions, stack_embeddings

parser = argparse.ArgumentParser(description="Benchmark image/video/audio/mixed queries")
parser.add_argument("--manifest", default=AEMS_MANIFEST_PATH)
parser.add_argument("--split", default="test", choices=["train", "test"])
parser.add_argument("--modalities", nargs="+",
                    default=["text", "image", "video", "audio", "audio_clip", "image+text"],
                    help="rows to measure")
parser.add_argument("--image-frame", type=int, default=4,
                    help="which held-out frame (0-7) is the single-image query")
parser.add_argument("--frame-split", default="temporal", choices=list(SPLITS),
                    help="temporal: index the first half of the video, query the second. "
                         "interleave: index even frames, query odd ones — adjacent in time "
                         "and so near-duplicates, which inflates the result.")
parser.add_argument("--per-branch", action="store_true",
                    help="also score each reachable branch alone. For an image or video "
                         "query this separates appearance matching (the visual branch, "
                         "frames against frames) from genuinely cross-modal semantic "
                         "matching (the caption and passage branches, image against text).")
parser.add_argument("--diagnose-frames", action="store_true",
                    help="report how close query frames sit to their own gallery frames "
                         "relative to the best other video, under both splits")
parser.add_argument("--min-duration", type=float, default=60.0,
                    help="audio queries need a segment disjoint from all three "
                         "indexed segments; below this they would overlap")
parser.add_argument("--audio-position", type=float, default=0.25,
                    help="where in the video the held-out audio segment is taken from")
parser.add_argument("--bootstrap", action="store_true")
parser.add_argument("--bootstrap-iters", type=int, default=2000)
parser.add_argument("--output", default="outputs/aems/multimodal_query_benchmark.json")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
set_seeds(args.seed)

records = load_records(args.manifest, args.split)
sources = BranchSources.load(records, args.split)
video_ids = sources.usable([v for v, r in records.items() if questions(r)])
n = len(video_ids)
print(f"[DATA] gallery = {n} {args.split} videos", flush=True)

# ---------------------------------------------------------------- gallery side
frames = {v: torch.as_tensor(sources.frames[v]).float().reshape(-1, 512) for v in video_ids}
per_video = {f.shape[0] for f in frames.values()}
if per_video != {16}:
    raise SystemExit(f"expected 16 frames per video, found {sorted(per_video)}; "
                     "the even/odd hold-out assumes a uniform frame count")
_held = {v: frame_holdout(f, args.frame_split) for v, f in frames.items()}
index_frames = {v: i for v, (i, _) in _held.items()}
query_frames = {v: q for v, (_, q) in _held.items()}

def _on(index):
    return index._replace(rows=index.rows.to(DEVICE), owner=index.owner.to(DEVICE))

visual_index = _on(chunk_index_from(index_frames, video_ids))
chunk_index = _on(chunk_index_from(sources.chunks, video_ids))
dense_dim = torch.as_tensor(sources.dense[video_ids[0]]).shape[-1]
dense_index = _on(chunk_index_from(sources.dense, video_ids, dim=dense_dim))
caption_matrix = stack_embeddings(sources.caption, video_ids, DEVICE)
audio_matrix = stack_embeddings(sources.audio, video_ids, DEVICE)
bm25_index = BM25PassageIndex([lexical_fields(records[v]) for v in video_ids])
print(f"[DATA] visual branch rebuilt from 8 frames per video ({args.frame_split} split)",
      flush=True)


def diagnose_frames():
    """How much of an image query's score is near-duplicate matching?

    For each split, compare a query frame's best similarity to its *own* video's
    gallery frames against its best similarity to any other video. A large gap
    means the task is dominated by near-duplicate matching rather than by the
    encoder recognising content.
    """
    for split in SPLITS:
        held = {v: frame_holdout(f, split) for v, f in frames.items()}
        idx = {v: i for v, (i, _) in held.items()}
        qry = {v: q for v, (_, q) in held.items()}
        gallery = _on(chunk_index_from(idx, video_ids))
        q = F.normalize(torch.stack([qry[v][args.image_frame] for v in video_ids]), dim=1)
        sims = gallery.max_sim_batch(q.to(DEVICE))
        own = sims[torch.arange(n, device=DEVICE), torch.arange(n, device=DEVICE)]
        other = sims.clone()
        other[torch.arange(n, device=DEVICE), torch.arange(n, device=DEVICE)] = -1e4
        best_other = other.max(dim=1).values
        print(f"  {split:<11} own-video best-frame sim={own.mean():.4f}  "
              f"best other video={best_other.mean():.4f}  margin={(own - best_other).mean():+.4f}",
              flush=True)


if args.diagnose_frames:
    print("[DIAG] query-frame similarity to its own gallery frames vs. the best other video:",
          flush=True)
    diagnose_frames()


def branch_sims(clip_q=None, dense_q=None, texts=None, audio_q=None):
    """Per-branch z-scored similarities, None where the query cannot reach a branch."""
    sims = {b: None for b in BRANCHES}
    if clip_q is not None:
        clip_q = F.normalize(clip_q.float().to(DEVICE), dim=1)
        sims["visual"] = visual_index.max_sim_batch(clip_q)
        sims["text"] = clip_q @ caption_matrix.T
        sims["chunk"] = chunk_index.max_sim_batch(clip_q)
    if audio_q is not None:
        sims["audio"] = F.normalize(audio_q.float().to(DEVICE), dim=1) @ audio_matrix.T
    if texts is not None:
        sims["bm25"] = bm25_index.score_batch(texts).to(DEVICE)
    if dense_q is not None:
        sims["dense"] = dense_index.max_sim_batch(dense_q.float().to(DEVICE))
    return [None if sims[b] is None else zscore(sims[b]) for b in BRANCHES]


def evaluate_one(sims, gt, branch):
    """One branch alone, as a ranking over the gallery."""
    i = BRANCHES.index(branch)
    return ground_truth_ranks(sims[i], gt)


def evaluate(sims, gt):
    """Fuse reachable branches with the deployed weights, renormalized over them."""
    w = fixed_weights().to(DEVICE)
    mask = torch.tensor([s is not None for s in sims], dtype=torch.float32, device=DEVICE)
    w = w * mask
    if w.sum() <= 0:                      # every reachable branch has weight 0
        w = mask
    w = (w / w.sum()).unsqueeze(0)
    zeros = torch.zeros(sims[[i for i, s in enumerate(sims) if s is not None][0]].shape,
                        device=DEVICE)
    fused = stage1.fuse(w, [zeros if s is None else s for s in sims])
    return ground_truth_ranks(fused, gt), [b for b, s in zip(BRANCHES, sims) if s is not None]


# ---------------------------------------------------------------- query side
gt_all = torch.arange(n, device=DEVICE)
rows, built = {}, {}

if {"text", "image+text"} & set(args.modalities):
    # One question per video, so every modality is scored on the same footing.
    texts = [questions(records[v])[0] for v in video_ids]
    clip_text, dense_text = encode_queries(texts, DEVICE)
    built["text"] = (clip_text, dense_text, texts)

if "text" in args.modalities:
    clip_text, dense_text, texts = built["text"]
    rows["text"] = (branch_sims(clip_q=clip_text, dense_q=dense_text, texts=texts), gt_all)

if "image" in args.modalities:
    q = torch.stack([query_frames[v][args.image_frame] for v in video_ids])
    rows["image"] = (branch_sims(clip_q=q), gt_all)

if "video" in args.modalities:
    q = torch.stack([F.normalize(query_frames[v].mean(0), dim=0) for v in video_ids])
    rows["video"] = (branch_sims(clip_q=q), gt_all)

if "image+text" in args.modalities:
    clip_text, dense_text, texts = built["text"]
    img = torch.stack([query_frames[v][args.image_frame] for v in video_ids]).to(DEVICE)
    mixed = F.normalize(F.normalize(img, dim=1) + clip_text.to(DEVICE), dim=1)
    rows["image+text"] = (branch_sims(clip_q=mixed, dense_q=dense_text, texts=texts), gt_all)

if {"audio", "audio_clip"} & set(args.modalities):
    import librosa
    from src.encoders.wavlm_encode import WavLMEncoder

    eligible = [(i, v) for i, v in enumerate(video_ids)
                if float(records[v].get("duration_seconds", 0.0)) >= args.min_duration
                and audio_segment_is_disjoint(float(records[v]["duration_seconds"]),
                                              args.audio_position, AEMS_AUDIO_CLIP_SEC)]
    print(f"[AUDIO] {len(eligible)}/{n} videos run >= {args.min_duration:g}s, so a "
          f"{AEMS_AUDIO_CLIP_SEC}s segment at the {args.audio_position:.0%} mark is "
          f"disjoint from all three indexed segments", flush=True)

    encoder = WavLMEncoder(DEVICE)
    adapter = load_audio_adapter(AEMS_AUDIO_ADAPTER_PATH, DEVICE)
    seg_len = AEMS_WAVLM_SR * AEMS_AUDIO_CLIP_SEC
    feats, keep = [], []
    for i, v in eligible:
        path = records[v].get("audio_path") or os.path.join(AEMS_AUDIO_DIR, f"{v}.wav")
        try:
            wav, _ = librosa.load(path, sr=AEMS_WAVLM_SR, mono=True)
        except Exception as exc:                      # a missing or unreadable wav
            print(f"  [SKIP] {v}: {exc}", flush=True)
            continue
        start = int(max(0, len(wav) * args.audio_position - seg_len / 2))
        seg = wav[start:start + seg_len]
        if len(seg) < seg_len:
            seg = np.pad(seg, (0, seg_len - len(seg)))
        feats.append(seg.astype("float32"))
        keep.append(i)
        if len(feats) % 200 == 0:
            print(f"  loaded {len(feats)}/{len(eligible)}", flush=True)

    embs = []
    for s in range(0, len(feats), 32):
        with torch.no_grad():
            w1024 = encoder.encode_segments(feats[s:s + 32]).to(DEVICE)
            embs.append(adapter(w1024).float().cpu())
    audio_q = torch.cat(embs)
    gt_audio = torch.tensor(keep, device=DEVICE)
    print(f"[AUDIO] encoded {audio_q.shape[0]} held-out segments", flush=True)
    del encoder
    torch.cuda.empty_cache()

    if "audio" in args.modalities:
        rows["audio"] = (branch_sims(audio_q=audio_q), gt_audio)
    if "audio_clip" in args.modalities:
        # The adapter emits a CLIP-text-space vector, so an audio query can also be
        # read as a CLIP query and reach the caption, passage and visual branches.
        # Whether that helps is exactly what the deployed router does not test.
        rows["audio_clip"] = (branch_sims(clip_q=audio_q, audio_q=audio_q), gt_audio)

# ---------------------------------------------------------------- report
print(f"\n{'query modality':<14} {'n':>5}  {'R@1':>7} {'R@5':>7} {'R@10':>7} {'MRR':>7}  branches")
print("-" * 86)
out = {}
for name in args.modalities:
    if name not in rows:
        continue
    sims, gt = rows[name]
    ranks, reachable = evaluate(sims, gt)
    m = metrics_from_ranks(ranks)
    entry = {"n_queries": int(ranks.numel()), "branches": reachable, **m}
    if args.bootstrap:
        lo, hi = bootstrap_ci(hits_at_k(ranks, 1), iters=args.bootstrap_iters, seed=args.seed)
        entry["R@1_CI"] = [lo, hi]
    out[name] = entry
    ci = f"  [{entry['R@1_CI'][0]:.4f}, {entry['R@1_CI'][1]:.4f}]" if args.bootstrap else ""
    print(f"{name:<14} {entry['n_queries']:>5}  {m['R@1']:>7.4f} {m['R@5']:>7.4f} "
          f"{m['R@10']:>7.4f} {m['MRR']:>7.4f}  {'+'.join(reachable)}{ci}")
    if args.per_branch and len(reachable) > 1:
        entry["per_branch"] = {}
        for b in reachable:
            bm = metrics_from_ranks(evaluate_one(sims, gt, b))
            entry["per_branch"][b] = bm
            print(f"{'  └ ' + b:<14} {entry['n_queries']:>5}  {bm['R@1']:>7.4f} "
                  f"{bm['R@5']:>7.4f} {bm['R@10']:>7.4f} {bm['MRR']:>7.4f}")

os.makedirs(os.path.dirname(args.output), exist_ok=True)
with open(args.output, "w") as f:
    json.dump({"split": args.split, "gallery_size": n, "protocol": {
        "frame_split": args.frame_split,
        "image_frame": args.image_frame, "audio_position": args.audio_position,
        "audio_min_duration": args.min_duration}, "results": out}, f, indent=2)
print(f"\n[SAVE] {args.output}")
