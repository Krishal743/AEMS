# Explainability

## Purpose
Generate human-readable explanations for every retrieval result, fulfilling the "Explainable" requirement in AEMS.

## Components

### `src/explainability/explain_retrieval.py`

Three functions:

### explain_gating_decision(weights)
- Input: [w_v, w_t, w_a] tensor from GatingNetwork
- Output: dominant modality, confidence spread, per-modality weights
- Purpose: Show how the query was routed

### explain_modality_contributions(weights, sim_v, sim_t, sim_a, video_ids, top_k=5)
- Input: gating weights + per-modality similarity vectors
- Output: per-video breakdown of visual/caption/audio contribution to fused score
- Purpose: Explain why each video was retrieved

### explain_ranking_difference(rank1, rank2)
- Input: Two consecutive ranked results
- Output: which modality caused the score difference
- Purpose: Explain why rank1 beat rank2

### format_explanation(contributions, gating, top_k=5)
- Combines all three into a printable string

## Example Output

```
Gating weights: [visual=0.4591, caption=0.0711, audio=0.4698]
Dominant modality: audio (0.47, spread=0.40)

Top-5 results:
  1. video9975  score=0.2200
       visual=0.1333 (60.6%)
       caption=0.0444 (20.2%)
       audio=0.0423 (19.2%)
  Rank1 vs Rank2: deciding modality = visual (Δ=0.0060)
```

## Integration
- Used by: `bin/queries/query_text.py`, `bin/demo/demo.py`, `bin/queries/query_mixed.py`
- Output format: plain text with visual/caption/audio breakdown per video
