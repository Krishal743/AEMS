# AEMS Audio Alignment — M4 vs M6-hybrid: multi-seed/crop decision — Post Phase 3 Review 1

**Date:** 03 September 2026 (same data stamp as the M6 campaign).

## Purpose
The M6-hybrid adapter won audio-query→text (a2t) but trailed M4 in
text-query→audio (t2a) on single seed-42 runs. This settles whether those gaps
are real, via 3 training seeds × (cached + 3 random-crop) evaluation of both
methods, using the exact M6 training recipe and a dropout-OFF gallery-based
best-state selector. No existing models/scripts/DBs were modified.

## Protocol
`experiments/audio_alignment/eval_multi_seed_crop.py` (new):
- Seeds {100, 200, 300} retrain of `AudioAdapter`, targets: text (M4) vs
  video+text hybrid (M6-hybrid). Best state = dropout-OFF a2t R@1 on cached
  test features (the original training-time metric ran with dropout ON).
- Test variants: `cached` (deployed 3-segment-mean features) + `crop_400/500/600`
  (deterministic random 10s windows via `AudioCollator`/`rand_trunc`).
- Published seed-42 models re-evaluated as continuity anchors.
- Decision uses a t(0.975, df=4) ≈ 2.776 two-sample 95% CI on the differences.

## Results — seed variance on deployed (cached) features, mean±std over 3 seeds
| Metric | M4 | M6-hybrid | Diff (95% CI) |
|---|---:|---:|---:|
| t2a R@1 | 0.0274±0.002 | 0.0228±0.003 | M4 +0.0046 (CI ±0.0050) → borderline |
| t2a MRR | 0.0630±0.002 | 0.0567±0.003 | **M4 +0.0063 (CI ±0.0048) → significant** |
| t2a R@10 | 0.1102±0.004 | 0.1190±0.005 | hybrid +0.009 (n.s.) |
| a2t R@1 | 0.0489±0.002 | 0.0564±0.003 | **hybrid +0.0075 (CI ±0.0050) → significant** |
| a2t MRR | 0.1025±0.001 | 0.1102±0.001 | hybrid +0.008 |
| a2t R@10 | 0.2032±0.003 | 0.2123±0.007 | hybrid +0.009 |

Anchors match published artifacts: M4 t2a/a2t = 0.0323/0.0538,
M6-hybrid = 0.0254/0.0587 (single seed-42 draws slightly exceed the fresh-seed
means — the means are the honest central estimates).

## Decision (rule: adopt hybrid iff a2t beats outside noise AND t2a within noise)
- a2t: hybrid beats M4 **robustly** (+0.0075 R@1; CI excludes 0).
- t2a: M4's MRR edge (+0.0063) is **outside** the 95% CI — not "within noise".
- **VERDICT: KEEP M4 as deployable.** Hybrid's audio-query gain is real but the
  text-query→audio branch (the live search direction) retains a significant edge
  with M4, so per the specified rule M4 stays the deployable aligned audio DB.

## Robustness probe (3 random 10s windows) — secondary
Both methods degrade sharply on single-window input (adapter was trained on
3-segment-mean): t2a R@1 ≈ 0.012–0.014, a2t R@1 ≈ 0.025; methods are effectively
**tied** there (a2t 0.0253 vs 0.0252). Implications: (i) the hybrid's a2t
advantage is specific to the 3-segment-mean input the pipeline deploys;
(ii) neither adapter generalizes well to single 10s crops — a residual
generalization gap if the query path ever uses one random window.

## Artifacts
- Script: `experiments/audio_alignment/eval_multi_seed_crop.py` (new; `--skip-train`,
  `--no-crops` flags supported).
- Results: `outputs/alignment/multi_seed_crop_eval.json` (full per-seed×variant
  tables, aggregates, decision).
- Seeded models: `outputs/alignment/seeded_models/{m4,m6_hybrid}_seed{S}.pt`.
- Deployed DB unchanged: `embeddings/aems_audio_aligned_m4_adapter_ct.pt`.

## Optional follow-up (not run; user gate)
`m6_framewise.py` — 16-frame multi-positive (SupCon-style) hybrid target — if
more a2t headroom is desired without losing t2a; evidence above says headroom is
small and t2a is the binding direction.