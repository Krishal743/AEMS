import torch


def explain_modality_contributions(weights, sim_v, sim_t, sim_a, video_ids, top_k=5):
    w_v, w_t, w_a = weights
    contrib_v = w_v * sim_v
    contrib_t = w_t * sim_t
    contrib_a = w_a * sim_a
    fused = contrib_v + contrib_t + contrib_a

    ranked = torch.argsort(fused, descending=True)
    explanations = []
    for rank_idx in ranked[:top_k].tolist():
        vid = video_ids[rank_idx]
        cv = contrib_v[rank_idx].item()
        ct = contrib_t[rank_idx].item()
        ca = contrib_a[rank_idx].item()
        total = fused[rank_idx].item()
        explanations.append({
            "video_id": vid,
            "fused_score": total,
            "visual_score": cv,
            "visual_pct": 100.0 * cv / total if total > 0 else 0.0,
            "caption_score": ct,
            "caption_pct": 100.0 * ct / total if total > 0 else 0.0,
            "audio_score": ca,
            "audio_pct": 100.0 * ca / total if total > 0 else 0.0,
        })
    return explanations


def explain_gating_decision(weights):
    w = weights.squeeze().tolist()
    labels = ["visual", "caption", "audio"]
    dominant_idx = max(range(3), key=lambda i: w[i])
    spread = max(w) - min(w)
    return {
        "w_v": w[0],
        "w_t": w[1],
        "w_a": w[2],
        "dominant_modality": labels[dominant_idx],
        "dominant_weight": w[dominant_idx],
        "confidence_spread": spread,
    }


def explain_ranking_difference(rank1, rank2):
    diffs = {}
    for key in ["visual_score", "caption_score", "audio_score"]:
        d = rank1[key] - rank2[key]
        diffs[key.replace("_score", "")] = d
    deciding = max(diffs, key=lambda k: abs(diffs[k]))
    return {
        "fused_delta": rank1["fused_score"] - rank2["fused_score"],
        "per_modality_delta": diffs,
        "deciding_modality": deciding,
    }


def format_explanation(contributions, gating, top_k=5):
    lines = []
    lines.append(f"Gating weights: [visual={gating['w_v']:.4f}, caption={gating['w_t']:.4f}, audio={gating['w_a']:.4f}]")
    lines.append(f"Dominant modality: {gating['dominant_modality']} ({gating['dominant_weight']:.2f}, spread={gating['confidence_spread']:.2f})")
    lines.append("")
    lines.append(f"Top-{top_k} results:")
    for i, c in enumerate(contributions):
        lines.append(f"  {i+1}. {c['video_id']}  score={c['fused_score']:.4f}")
        lines.append(f"       visual={c['visual_score']:.4f} ({c['visual_pct']:.1f}%)")
        lines.append(f"       caption={c['caption_score']:.4f} ({c['caption_pct']:.1f}%)")
        lines.append(f"       audio={c['audio_score']:.4f} ({c['audio_pct']:.1f}%)")
    if len(contributions) >= 2:
        rd = explain_ranking_difference(contributions[0], contributions[1])
        lines.append(f"  Rank1 vs Rank2: deciding modality = {rd['deciding_modality']} (Δ={rd['per_modality_delta'][rd['deciding_modality']]:.4f})")
    return "\n".join(lines)
