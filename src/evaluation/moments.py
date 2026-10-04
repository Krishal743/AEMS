"""Scoring candidate windows for temporal grounding.

Shared by `bin/evaluation/eval_moment_retrieval.py` and
`bin/training/tune_moment_fusion.py`: the tuner picks a fusion weight on
validation and the evaluator applies it to test, so they must score windows
identically or the tuned value means nothing.

Candidates are every run of 1..`max_segments` consecutive transcript segments —
the same space the labels were drawn from — so the correct span is always
reachable and the task is pure ranking rather than regression.
"""

import numpy as np
import torch
import torch.nn.functional as F

from src.data.timecodes import iou, segments
from src.retrieval.bm25 import BM25PassageIndex

METHODS = ("whole_video", "center", "bm25", "dense", "visual", "dense+visual")


def load_frames(dense_path=None, uniform_path=None, n_frames=16):
    """(embeddings, timestamps or None). Timestamps are None for uniform sampling,
    where frame times are derived from each video's duration instead."""
    if dense_path:
        bundle = torch.load(dense_path, weights_only=False)
        return bundle["embeddings"], bundle["timestamps"], bundle.get("interval")
    return torch.load(uniform_path, weights_only=False), None, None


def windows_for(segs, max_segments):
    """Spans, joined texts and segment ranges for every candidate window."""
    spans, texts, ranges = [], [], []
    for width in range(1, max_segments + 1):
        for i in range(len(segs) - width + 1):
            spans.append((segs[i][0], segs[i + width - 1][1]))
            texts.append(" ".join(t for _, _, t in segs[i:i + width]))
            ranges.append((i, i + width))
    return spans, texts, ranges


def _z(x):
    return (x - x.mean()) / (x.std() + 1e-6)


def score_moments(moments, records, seg_emb, seg_index, video_segs, q_clip, q_dense,
                  frame_db, frame_time_db, device, max_segments=5, visual_weight=0.5,
                  n_frames=16, methods=METHODS, query_index=None):
    """IoU per moment per method, plus whether a frame fell inside each span.

    `query_index` maps a moment's position in `moments` to its row in the query
    embeddings; defaults to the identity.
    """
    by_video = {}
    for pos, m in enumerate(moments):
        by_video.setdefault(m["video_id"], []).append((pos, m))

    results = {name: [] for name in methods}
    order, frame_hit = [], []
    for video, items in by_video.items():
        segs = video_segs[video]
        spans, texts, ranges = windows_for(segs, max_segments)
        if not spans:
            continue
        duration = float(records[video].get("duration_seconds") or 0.0)
        lo, hi = seg_index[video]
        seg_v = seg_emb[lo:hi]
        win_dense = F.normalize(torch.stack([seg_v[a:b].mean(0) for a, b in ranges]), dim=1)
        bm25 = BM25PassageIndex([[t] for t in texts])

        frames = frame_times = None
        if video in frame_db:
            frames = F.normalize(
                torch.as_tensor(frame_db[video]).float().reshape(-1, 512).to(device), dim=1)
            if frame_time_db is not None:
                frame_times = np.asarray(frame_time_db[video], dtype=float)
            elif duration > 0:
                frame_times = np.array([(k + 0.5) * duration / n_frames
                                        for k in range(frames.shape[0])])
            if frame_times is None or len(frame_times) != frames.shape[0]:
                frames = frame_times = None

        # Which candidate windows each frame falls into, computed once per video.
        frame_in_window = None
        if frames is not None:
            frame_in_window = [np.where((frame_times >= s) & (frame_times <= e))[0]
                               for s, e in spans]
        median_len = float(np.median([e - s for s, e in spans]))

        for pos, m in items:
            qi = query_index[pos] if query_index is not None else pos
            gt = (m["start"], m["end"])
            order.append(pos)
            frame_hit.append(bool(frame_times is not None
                                  and ((frame_times >= gt[0]) & (frame_times <= gt[1])).any()))
            preds, dn, vs = {}, None, None
            if "whole_video" in results:
                preds["whole_video"] = (0.0, duration)
            if "center" in results:
                preds["center"] = (max(0.0, duration / 2 - median_len / 2),
                                   min(duration, duration / 2 + median_len / 2))
            if "bm25" in results:
                bm = bm25.score_batch([m["question"]]).squeeze(0)
                preds["bm25"] = spans[int(bm.argmax())]
            if {"dense", "dense+visual"} & set(results):
                dn = win_dense @ q_dense[qi]
                if "dense" in results:
                    preds["dense"] = spans[int(dn.argmax())]
            if {"visual", "dense+visual"} & set(results) and frames is not None:
                per_frame = frames @ q_clip[qi].to(device)
                vs = torch.full((len(spans),), float("-inf"), device=device)
                for k, hit in enumerate(frame_in_window):
                    if len(hit):
                        vs[k] = per_frame[torch.as_tensor(hit, device=device)].max()
                if torch.isinf(vs).all():
                    vs = torch.zeros_like(vs)
                if "visual" in results:
                    preds["visual"] = spans[int(vs.argmax())]
            if "dense+visual" in results and dn is not None:
                combined = _z(dn)
                if vs is not None:
                    finite = torch.isfinite(vs)
                    vz = torch.zeros_like(vs)
                    if finite.any():
                        vz[finite] = _z(vs[finite])
                    combined = combined + visual_weight * vz
                preds["dense+visual"] = spans[int(combined.argmax())]
            for name, span in preds.items():
                results[name].append(iou(span, gt))
    return ({k: np.array(v) for k, v in results.items()},
            np.array(frame_hit), np.array(order))


def encode_side(moments, records, device, dense_encoder=None):
    """Question embeddings (CLIP, dense) and per-video transcript segment embeddings."""
    from src.encoders.text_retrieval import load_dense_encoder
    from src.training.query_data import encode_clip_text

    questions = [m["question"] for m in moments]
    q_clip = encode_clip_text(questions, device)
    encoder = dense_encoder or load_dense_encoder(device)
    q_dense = encoder.encode_queries(questions, batch_size=256).to(device)

    video_segs, seg_index, seg_texts = {}, {}, []
    for video in dict.fromkeys(m["video_id"] for m in moments):
        segs = segments(records[video])
        video_segs[video] = segs
        seg_index[video] = (len(seg_texts), len(seg_texts) + len(segs))
        seg_texts += [t for _, _, t in segs]
    seg_emb = encoder.encode_passages(seg_texts, batch_size=256).to(device)
    if dense_encoder is None:
        del encoder
        if device == "cuda":
            torch.cuda.empty_cache()
    return q_clip, q_dense, seg_emb, seg_index, video_segs
