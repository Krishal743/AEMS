# Architecture Audit

**Date**: July 2026 (Updated after Phase 0)

---

## Completed Modules

| Module | File | Status | Notes |
|--------|------|--------|-------|
| Data pipeline | `src/data/datasets.py`, `src/data/metadata.py` | ✅ | MSRVTT loading, split filtering, common video ID extraction |
| CLIP encoder | `src/encoders/clip_encode.py` | ✅ | Video frame encoding + text encoding |
| CLAP encoder | `src/encoders/clap_encode.py` | ✅ | Audio encoding via laion_clap |
| Temporal transformer | `src/models/temporal_transformer.py` | ✅ | 6.6M params, 2-layer, CLS token |
| Gating network | `src/models/gating_network.py` | ✅ | 66K params, MLP(512→128→3), softmax |
| Retrieval evaluation | `src/evaluation/evaluate_retrieval.py` | ✅ | R@1/R@5/R@10, dimension validation added |
| Config | `src/config.py` | ✅ | Shared constants, seeds, GPU utils |
| Embedding extraction | `scripts/embeddings/` | ✅ | Video, audio, caption embedding scripts |
| Transformer training | `bin/training/train_temporal_transformer.py` | ✅ | InfoNCE loss, 12 epochs |
| Gating retraining | `scripts/training/retrain_gating.py` | ✅ | Ranking loss, loads train+test caption splits |
| Eval orchestration | `bin/evaluation/run_final_eval.sh` | ✅ | 7-step eval suite with PYTHONPATH |
| Eval multimodal | `scripts/evaluation/eval_multimodal.py` | ✅ | Per-modality + gated fusion eval |
| CLIP baseline | `scripts/baselines/run_clip_baseline.py` | ✅ | Visual-only baseline |
| Query router | `src/routing/query_router.py` | ✅ | Multi-query encoding (text/image/audio/video) |
| Explainability | `src/explainability/explain_retrieval.py` | ✅ | Modality contributions, gating decision, ranking explanation |
| Text query | `bin/queries/query_text.py` | ✅ | Text-to-video with explanation |
| Image query | `bin/queries/query_image.py` | ✅ | Image-to-video with explanation |
| Audio query | `bin/queries/query_audio.py` | ✅ | Audio-to-video with explanation |
| Video query | `bin/queries/query_video.py` | ✅ | Video-to-video with explanation |
| Mixed query | `bin/queries/query_mixed.py` | ✅ | Mixed text+image query with explanation |
| Behavioural test | `bin/evaluation/behavioural_test.py` | ✅ | 12 queries across 4 sets, routing table output |
| Final eval | `bin/evaluation/final_eval.py` | ✅ | Unified 5-system comparison |
| Ablation study | `bin/evaluation/ablation_study.py` | ✅ | 11 ablations |
| Demo pipeline | `bin/demo/demo.py` | ✅ | Interactive text query demo with explanations |
| Architecture audit | `docs/ARCHITECTURE_AUDIT.md` | ✅ | This document |

## Partially Completed / Negative Results

| Module | Status | Gap |
|--------|--------|-----|
| Gating network (meanpool) | ⚠️ Trained, stable negative result | w_v=0.40, w_t=0.21, w_a=0.40; test R@1=0.1287 — ranking loss cannot learn strong preferences |
| Gating network (transformer) | ⚠️ Trained, collapsed negative result | w_v=0.015, w_t=0.445, w_a=0.540; test R@1=0.0451 — gate ignores video entirely |
| Temporal transformer | ⚠️ Trained 12 epochs | Loss still decreasing; R@1 improves R@10 (+0.0175) but regresses R@1 (-0.0225) vs meanpool |

## Negative Results Summary (Phase 0 Verification)

Phase 0 re-trained both gating variants with the corrected caption split. Results are indistinguishable from the buggy runs:

| System | w_v | w_t | w_a | Test R@1 | Test R@5 | Test R@10 |
|--------|-----|-----|-----|---------|---------|----------|
| Meanpool (original buggy) | 0.4010 | 0.2014 | 0.3976 | 0.1268 | 0.2760 | 0.3568 |
| Meanpool (re-trained) | 0.3985 | 0.2058 | 0.3957 | 0.1287 | 0.2738 | 0.3557 |
| Transformer (original buggy) | 0.0154 | 0.4496 | 0.5350 | 0.0461 | 0.1240 | 0.1742 |
| Transformer (re-trained) | 0.0150 | 0.4450 | 0.5400 | 0.0450 | 0.1216 | 0.1711 |

**Conclusion:** Bug 2 was not the root cause of the gating collapse. The gating network's ranking loss produces weak gradients once softmax weights are near-equal. This is a fundamental limitation of the architecture, not a data bug. Documented as a valid negative result.

## Known Bugs

| Bug | Status | File | Fix |
|-----|--------|------|-----|
| Bug 1: zip-truncation | ✅ Fixed | `src/evaluation/evaluate_retrieval.py` | Added dimension validation |
| Bug 2: train caption split | ✅ Fixed (not root cause) | `scripts/training/retrain_gating.py` | Now loads both splits; re-training confirmed no change |
| Bug 3: gate weight paths | ✅ Fixed | `bin/evaluation/run_final_eval.sh` | Paths now correct |

## Embedding Inventory

| File | Keys | Shape per Key | Notes |
|------|------|---------------|-------|
| `video_embeddings.pt` | 10,000 | (512,) | CLIP meanpool |
| `video_embeddings_transformer.pt` | 9,087 | (512,) | Temporal transformer output |
| `audio_embeddings.pt` | 8,809 | (512,) | CLAP |
| `caption_embeddings_train.pt` | 7,010 | (20, 512) | CLIP text, train split |
| `caption_embeddings_test.pt` | 2,990 | (20, 512) | CLIP text, test split |
| `caption_embeddings.pt` | 2,990 | (20, 512) | Alias for test (identical) |

## Integration Issues

1. Audio coverage: 8,809/10,000 videos — 12% missing
2. Transformer coverage: 9,087/10,000 videos — 9% missing
3. Gating network ranking loss cannot learn strong modality preferences — architecture-level limitation
4. Old buggy weights preserved as `models/*_buggy.pth` for reference
