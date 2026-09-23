"""Option 1: per-candidate gating.

The deployed gate predicts one weight vector per query, applied to every
candidate alike. This variant predicts weights per *query-candidate pair*, so it
can decide that for this query, on this candidate, the passage evidence is what
matters while on another candidate the visual evidence is. It reads only the
branch scores the first stage already computed, so it is cheap: no extra
encoding, one small MLP over K candidates.
"""

import torch

from src.models.gating_network import GatingNetworkPerCandidate


def build(n_branches, hidden_dim=64, dropout=0.2, input_dim=512):
    return GatingNetworkPerCandidate(input_dim=input_dim, sim_dim=n_branches,
                                     hidden_dim=hidden_dim, num_modalities=n_branches,
                                     dropout=dropout)


def score(model, query_emb, candidate_branch_scores):
    """(n_queries, k) reranking scores.

    candidate_branch_scores: (n_queries, k, n_branches) from
    `src.rerank.stage1.gather_branch_scores`.
    """
    weights = model(query_emb, candidate_branch_scores)
    return (weights * candidate_branch_scores).sum(dim=-1)


def load(path, device, n_branches=4, hidden_dim=64):
    state = torch.load(path, map_location=device, weights_only=False)
    fusion = state.get("fusion_fc.weight")
    if fusion is None:
        raise RuntimeError(f"{path} is not a GatingNetworkPerCandidate checkpoint")
    if fusion.shape[0] != n_branches:
        raise RuntimeError(f"{path} predicts {fusion.shape[0]} weights but the router has "
                           f"{n_branches} branches; retrain with "
                           f"bin/training/train_per_candidate_gate.py")
    model = build(n_branches, hidden_dim=state["query_fc.weight"].shape[0]).to(device)
    model.load_state_dict(state)
    model.eval()
    return model
