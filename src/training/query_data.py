"""Shared data preparation for AEMS trainers: splits, QA queries, CLIP encoding."""

import json
import random

import torch
import torch.nn.functional as F
import clip


def load_records(manifest_path, split):
    """{video_id: record} for one manifest split."""
    return {r["video_id"]: r for r in json.load(open(manifest_path)) if r["split"] == split}


def questions(record):
    return [q for q in record["qa_questions"] if isinstance(q, str) and q.strip()]


def validation_split(videos, val_frac=0.15, seed=0):
    """Split a video list into (fit, validation), deterministically.

    Every trainer uses this with the same seed so the audio adapter, the gating
    network and any weight tuning share one validation set, and no model is
    selected on data another model was fitted on.
    """
    shuffled = sorted(videos)
    random.Random(seed).shuffle(shuffled)
    n_val = int(val_frac * len(shuffled))
    return sorted(shuffled[n_val:]), sorted(shuffled[:n_val])


def encode_clip_text(texts, device, batch_size=512):
    """(n, 512) L2-normalized CLIP text embeddings on CPU."""
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            tokens = clip.tokenize(texts[i:i + batch_size], truncate=True).to(device)
            out.append(F.normalize(model.encode_text(tokens).float(), dim=1).cpu())
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(out)


def flatten_questions(records, videos):
    """Return (texts, rows) where rows[video_id] indexes that video's queries."""
    texts, rows = [], {}
    for v in videos:
        for q in questions(records[v]):
            rows.setdefault(v, []).append(len(texts))
            texts.append(q)
    return texts, rows


def stack_embeddings(db, videos, device=None):
    """(len(videos), d) L2-normalized rows in `videos` order."""
    m = F.normalize(torch.stack([torch.as_tensor(db[v]).float().flatten() for v in videos]), dim=1)
    return m.to(device) if device else m


def query_rows(rows, videos, device=None):
    """Flat query indices and their ground-truth positions within `videos`."""
    idx, gt = [], []
    for j, v in enumerate(videos):
        idx += rows[v]
        gt += [j] * len(rows[v])
    gt = torch.tensor(gt)
    return torch.tensor(idx), (gt.to(device) if device else gt)


def recall_metrics(sim, gt):
    rank = (sim > sim.gather(1, gt[:, None])).sum(1).float()
    return {"R@1": (rank < 1).float().mean().item(),
            "R@5": (rank < 5).float().mean().item(),
            "R@10": (rank < 10).float().mean().item(),
            "MRR": (1 / (rank + 1)).mean().item()}
