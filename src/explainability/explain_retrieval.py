"""Explain a fused ranking: which branch earned each result, and what decided rank 1.

Branch similarities are z-scores, so a contribution can be negative (the branch
argues *against* that video). Shares are therefore reported over absolute
contributions, and the sign is kept in the raw score.
"""

import torch

from src.routing.query_router import BRANCHES

LABELS = {"visual": "visual", "text": "caption", "chunk": "passage", "audio": "audio"}


def explain_modality_contributions(weights, sims, video_ids, top_k=5):
    """sims: per-branch similarity vectors, in BRANCHES order."""
    contributions = [float(w) * s for w, s in zip(weights, sims)]
    fused = sum(contributions)

    explanations = []
    for rank_idx in torch.argsort(fused, descending=True)[:top_k].tolist():
        values = {b: contributions[i][rank_idx].item() for i, b in enumerate(BRANCHES)}
        magnitude = sum(abs(v) for v in values.values()) or 1.0
        entry = {"video_id": video_ids[rank_idx], "fused_score": fused[rank_idx].item()}
        for branch, value in values.items():
            entry[f"{branch}_score"] = value
            entry[f"{branch}_pct"] = 100.0 * abs(value) / magnitude
        explanations.append(entry)
    return explanations


def explain_gating_decision(weights):
    w = [float(x) for x in torch.as_tensor(weights).squeeze().tolist()]
    dominant = max(range(len(BRANCHES)), key=lambda i: w[i])
    decision = {f"w_{b}": w[i] for i, b in enumerate(BRANCHES)}
    decision.update({
        "weights": w,
        "dominant_modality": BRANCHES[dominant],
        "dominant_weight": w[dominant],
        "confidence_spread": max(w) - min(w),
    })
    return decision


def explain_ranking_difference(rank1, rank2):
    diffs = {b: rank1[f"{b}_score"] - rank2[f"{b}_score"] for b in BRANCHES}
    deciding = max(diffs, key=lambda k: abs(diffs[k]))
    return {
        "fused_delta": rank1["fused_score"] - rank2["fused_score"],
        "per_modality_delta": diffs,
        "deciding_modality": deciding,
    }


def format_explanation(contributions, gating, top_k=5):
    weights = ", ".join(f"{LABELS[b]}={gating[f'w_{b}']:.4f}" for b in BRANCHES)
    lines = [f"Gating weights: [{weights}]",
             f"Dominant modality: {LABELS[gating['dominant_modality']]} "
             f"({gating['dominant_weight']:.2f}, spread={gating['confidence_spread']:.2f})",
             "", f"Top-{top_k} results:"]
    for i, c in enumerate(contributions):
        lines.append(f"  {i + 1}. {c['video_id']}  score={c['fused_score']:.4f}")
        for b in BRANCHES:
            lines.append(f"       {LABELS[b]:<8}={c[f'{b}_score']:+.4f} ({c[f'{b}_pct']:.1f}%)")
    if len(contributions) >= 2:
        rd = explain_ranking_difference(contributions[0], contributions[1])
        delta = rd["per_modality_delta"][rd["deciding_modality"]]
        lines.append(f"  Rank1 vs Rank2: deciding modality = "
                     f"{LABELS[rd['deciding_modality']]} (Δ={delta:+.4f})")
    return "\n".join(lines)
