"""Stage-1 candidate generation, shared by every reranker.

Stage 1 is the existing fused bi-encoder search over the whole gallery. A
reranker only ever sees its top K, so stage-1 recall@K is the hard ceiling on
what any stage 2 can achieve — measure it before tuning anything else
(`candidate_recall`).
"""

import torch

from src.routing.query_router import zscore


def fuse(weights, sims):
    """weights: (n_queries, n_branches) or (n_branches,); sims: list of (n_queries, n_videos)."""
    w = torch.as_tensor(weights)
    if w.dim() == 1:
        w = w.view(1, -1)
    return sum(w[:, i:i + 1].to(sims[i].device) * sims[i] for i in range(len(sims)))


def zscore_branches(sims):
    return [zscore(s) for s in sims]


def top_k_candidates(scores, k):
    """(n_queries, k) candidate indices, best first."""
    k = min(k, scores.shape[1])
    return scores.topk(k, dim=1).indices


def candidate_recall(candidates, gt):
    """Fraction of queries whose ground truth is among the candidates."""
    gt = torch.as_tensor(gt, device=candidates.device).view(-1, 1)
    return (candidates == gt).any(dim=1).float().mean().item()


def gather_branch_scores(sims, candidates):
    """Per-branch scores restricted to each query's candidates.

    Returns (n_queries, k, n_branches); branch order follows `sims`.
    """
    return torch.stack([s.gather(1, candidates) for s in sims], dim=-1)


def rerank_scores_to_ranking(stage1_scores, candidates, rerank_scores):
    """Fold stage-2 scores back into a full-gallery score matrix.

    Candidates are lifted above every non-candidate by construction: a
    reranked candidate's score is `max_non_candidate + 1 + rerank_score`, so the
    reranker reorders its own shortlist without letting an unseen video
    overtake it. Non-candidates keep their stage-1 order below.
    """
    out = stage1_scores.clone()
    floor = out.max(dim=1, keepdim=True).values + 1.0
    out.scatter_(1, candidates, floor + rerank_scores)
    return out


def positive_position(candidates, gt):
    """Index of the ground truth within each query's candidate list, or -1."""
    gt = torch.as_tensor(gt, device=candidates.device).view(-1, 1)
    match = candidates == gt
    found = match.any(dim=1)
    position = match.float().argmax(dim=1)
    return torch.where(found, position, torch.full_like(position, -1))
