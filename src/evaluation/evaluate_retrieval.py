"""Retrieval metrics and bootstrap confidence intervals.

Everything derives from one quantity: the rank of each query's ground-truth
video. Computing it needs a single comparison pass, not a sort, and once it is
cached every R@k, MRR, category slice and bootstrap resample is a cheap
reduction over that vector. The previous implementation sorted the full
similarity matrix once per query *per k per bootstrap iteration*, which made
`--bootstrap` too slow to finish.

Ranks are 0-based and optimistic under ties: a rank counts how many candidates
score strictly higher than the ground truth.
"""

import numpy as np
import torch

DEFAULT_KS = (1, 5, 10)
BOOTSTRAP_ITERS = 1000


def ground_truth_ranks(similarity_matrix, gt_indices):
    """(n_queries,) rank of each query's ground-truth column. No sort needed."""
    sim = torch.as_tensor(similarity_matrix)
    gt = torch.as_tensor(gt_indices, dtype=torch.long, device=sim.device).view(-1, 1)
    return (sim > sim.gather(1, gt)).sum(1)


def hits_at_k(ranks, k):
    """(n_queries,) float 1/0 vector: was the ground truth in the top k?"""
    return (torch.as_tensor(ranks) < k).float()


def metrics_from_ranks(ranks, ks=DEFAULT_KS):
    ranks = torch.as_tensor(ranks).float()
    out = {f"R@{k}": hits_at_k(ranks, k).mean().item() for k in ks}
    out["MRR"] = (1.0 / (ranks + 1)).mean().item()
    out["MdR"] = int(ranks.median().item()) + 1
    return out


def evaluate_retrieval(similarity_matrix, text_video_ids, video_ids, ks=DEFAULT_KS):
    """Recall@k for each k. Kept signature-compatible with earlier callers."""
    if similarity_matrix.size(0) != len(text_video_ids):
        raise ValueError(
            f"Similarity rows ({similarity_matrix.size(0)}) != text_video_ids ({len(text_video_ids)})"
        )
    if similarity_matrix.size(1) != len(video_ids):
        raise ValueError(
            f"Similarity columns ({similarity_matrix.size(1)}) != video_ids ({len(video_ids)})"
        )
    position = {v: i for i, v in enumerate(video_ids)}
    # A ground truth outside the gallery can never be retrieved. Rank it beyond
    # any k a caller could ask for, so it is a miss even when k > n_videos.
    gt = [position.get(v, -1) for v in text_video_ids]
    missing = torch.tensor([i < 0 for i in gt])
    ranks = ground_truth_ranks(similarity_matrix, [max(i, 0) for i in gt])
    unreachable = torch.iinfo(ranks.dtype).max
    ranks = torch.where(missing.to(ranks.device), torch.full_like(ranks, unreachable), ranks)
    return {f"R@{k}": hits_at_k(ranks, k).mean().item() for k in ks}


def bootstrap_ci(hits, iters=BOOTSTRAP_ITERS, seed=42, alpha=0.05):
    """95% CI for a mean over queries, by resampling the per-query hit vector.

    Resampling hits is equivalent to resampling queries and recomputing the
    metric, because the metric is a mean over queries and each query's
    contribution does not depend on the others.
    """
    hits = torch.as_tensor(hits, dtype=torch.float32).cpu()
    n = hits.numel()
    if n == 0:
        return 0.0, 0.0
    generator = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, n, (iters, n), generator=generator)
    means = hits[idx].mean(dim=1).numpy()
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def paired_bootstrap(hits_a, hits_b, iters=BOOTSTRAP_ITERS, seed=42, alpha=0.05):
    """Mean difference a - b with a 95% CI, resampling the paired differences.

    Both systems are scored on the same queries, so pairing removes the
    variance from query difficulty and leaves the variance that matters: how
    often a beats b. A CI excluding zero is the evidence that a change is real.
    """
    a = torch.as_tensor(hits_a, dtype=torch.float32).cpu()
    b = torch.as_tensor(hits_b, dtype=torch.float32).cpu()
    if a.shape != b.shape:
        raise ValueError(f"paired bootstrap needs matching queries: {a.shape} vs {b.shape}")
    diff = a - b
    low, high = bootstrap_ci(diff, iters=iters, seed=seed, alpha=alpha)
    return float(diff.mean().item()), low, high
