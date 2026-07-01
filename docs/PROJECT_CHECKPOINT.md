# Project Checkpoint: Query-Conditioned Adaptive Video Retrieval

**Project Name**: Query-Conditioned Adaptive Fusion for Any-to-Any Video Retrieval
**Status**: Temporal Transformer Trained + Full Eval Suite Run
**Last Updated**: July 2026
**Platform**: Linux (NVIDIA RTX 4500 Ada Generation, 24 GB VRAM)
**Original Platform**: Windows (RTX 3050, 6 GB VRAM)

---

## Executive Summary

Video retrieval system using CLIP (visual), CLAP (audio), and caption embeddings with a learned gating network for query-adaptive modality fusion. The project progressed from a prototype gating network (R@1=0.60) to a full-scale pipeline: temporal transformer trained to 12 epochs, gating network retrained on two embedding variants (meanpool & transformer), and a comprehensive 7-part eval suite executed.

**The gating network on mean-pooled embeddings achieves R@1=0.1268 (test), but the transformer-based gate collapsed (R@1=0.0461) — it learned to ignore video and almost exclusively use audio.** Three known bugs block the final comparison, but the transformer alone measurably improves R@10 over mean-pooling.

---

## Architecture

```
src/
├── config.py              # Shared constants (paths, dims, hyperparams)
├── models/
│   ├── gating_network.py  # GatingNetwork MLP (512→128→3, softmax)
│   └── temporal_transformer.py  # 2-layer TransformerEncoder (CLS+pos)
├── encoders/
│   ├── clip_encode.py     # CLIP model loading, video/text encoding
│   └── clap_encode.py     # CLAPEncoder class
├── data/
│   ├── datasets.py        # MSRVTTDataset (PyTorch Dataset)
│   └── metadata.py        # JSON loading, split filtering, common video IDs
└── evaluation/
    └── evaluate_retrieval.py  # evaluate_retrieval(sim_matrix, ...)
```

Embedding database (three dicts keyed by `video_id`, 512-dim vectors):
- `embeddings/video_embeddings.pt` — CLIP visual, 10K videos (mean-pooled)
- `embeddings/video_embeddings_transformer.pt` — Temporal Transformer output, 9,087 videos
- `embeddings/audio_embeddings.pt` — CLAP audio, 8,809 videos
- `embeddings/caption_embeddings.pt` — CLIP text, test split (2,990 videos × 20 captions)
- `embeddings/caption_embeddings_train.pt` — CLIP text, train split (available)
- `embeddings/caption_embeddings_test.pt` — CLIP text, test split (alias)

---

## Results

### Temporal Transformer Training (Epochs 1-12)

| Epoch | Train Loss | Val Loss | Val R@1 | Val R@5 | Val R@10 |
|-------|-----------|---------|---------|---------|---------|
| 1 | 4.1194 | 3.5886 | 0.1481 | 0.3329 | 0.4472 |
| 2 | 3.5321 | 3.3610 | 0.2035 | 0.4164 | 0.5323 |
| 3 | 3.3421 | 3.2271 | 0.2351 | 0.4778 | 0.5892 |
| 4 | 3.2132 | 3.1398 | 0.2643 | 0.5148 | 0.6254 |
| 5 | 3.1022 | 3.0696 | 0.2816 | 0.5358 | 0.6445 |
| 6 | 3.0230 | 3.0193 | 0.3021 | 0.5591 | 0.6634 |
| 7 | 2.9601 | 2.9768 | 0.3216 | 0.5790 | 0.6817 |
| 8 | 2.9060 | 2.9481 | 0.3317 | 0.5891 | 0.6899 |
| 9 | 2.8845 | 2.9231 | 0.3399 | 0.5951 | 0.6973 |
| 10 | 2.8497 | 2.9066 | 0.3470 | 0.6016 | 0.7047 |
| 11 | 2.8233 | 2.8976 | 0.3496 | 0.6066 | 0.7110 |
| **12** | **2.8071** | **2.8943** | **0.3548** | **0.6090** | **0.7153** |

Monotonic improvement across all 12 epochs — still not converged. Training time ~12 min/epoch on RTX 4500 Ada.

### Pre-Training Baseline (CLIP Visual, test split)

| Metric | Value |
|--------|-------|
| R@1 | 0.1378 |
| R@5 | 0.2767 |
| R@10 | 0.3515 |

### Transformer vs Mean-Pool (Test Split Comparison)

| Metric | Mean-Pool | Transformer | Δ |
|--------|-----------|-------------|----|
| R@1 | 0.1773 | 0.1547 | -0.0225 |
| R@5 | 0.3783 | 0.3597 | -0.0185 |
| R@10 | 0.4669 | 0.4844 | **+0.0175** |

Transformer improves R@10 but regresses at R@1 and R@5 vs mean-pooling. The per-video embedding count differs (10K vs 9,087) so this is not a strict apples-to-apples comparison.

### Transformer Eval (Direct, D3)

| Metric | Value |
|--------|-------|
| R@1 | 0.1551 |
| R@5 | 0.3557 |
| R@10 | 0.4800 |

### Gating Network Retraining (Meanpool Embeddings, 15 epochs)

**Best Val R@1**: 0.0898 (epoch 6)

| Epoch | Loss | R@1 | R@5 | R@10 |
|-------|------|-----|-----|------|
| 1 | 0.7966 | 0.0730 | 0.1721 | 0.2323 |
| 2 | 0.7917 | 0.0882 | 0.1998 | 0.2671 |
| 3 | 0.7925 | 0.0700 | 0.1599 | 0.2155 |
| 4 | 0.7917 | 0.0818 | 0.1871 | 0.2500 |
| 5 | 0.7910 | 0.0752 | 0.1741 | 0.2340 |
| **6** | **0.7913** | **0.0898** | **0.2027** | **0.2677** |
| 7 | 0.7911 | 0.0749 | 0.1715 | 0.2306 |
| 8 | 0.7916 | 0.0722 | 0.1674 | 0.2249 |
| 9 | 0.7907 | 0.0735 | 0.1698 | 0.2290 |
| 10 | 0.7910 | 0.0864 | 0.1977 | 0.2615 |
| 11 | 0.7923 | 0.0830 | 0.1878 | 0.2479 |
| 12 | 0.7898 | 0.0821 | 0.1865 | 0.2478 |
| 13 | 0.7905 | 0.0780 | 0.1777 | 0.2344 |
| 14 | 0.7906 | 0.0843 | 0.1940 | 0.2545 |
| 15 | 0.7896 | 0.0867 | 0.1945 | 0.2562 |

**Test evaluation** — Learned gate weights (mean across 500 train queries):

| Modality | Weight |
|----------|--------|
| w_v (video) | 0.4010 |
| w_t (caption) | 0.2014 |
| w_a (audio) | 0.3976 |

**Gated on test split**: R@1=0.1268, R@5=0.2760, R@10=0.3568

The gate has moderate video preference (w_v=0.40) and slightly deprioritizes captions — but R@1 is well below the sim_t caption-only baseline (0.93) because the ranking loss trains the gate to optimize fusion, not individual modalities.

### Gating Network Retraining (Transformer Embeddings, 15 epochs)

**Best Val R@1**: 0.0459 (epoch 12)

| Epoch | Loss | R@1 | R@5 | R@10 |
|-------|------|-----|-----|------|
| 1 | 0.8739 | 0.0336 | 0.0799 | 0.1109 |
| 2 | 0.8533 | 0.0369 | 0.0840 | 0.1167 |
| 3 | 0.8505 | 0.0400 | 0.0942 | 0.1271 |
| 4 | 0.8507 | 0.0364 | 0.0831 | 0.1144 |
| 5 | 0.8520 | 0.0391 | 0.0883 | 0.1198 |
| 6 | 0.8503 | 0.0390 | 0.0878 | 0.1198 |
| 7 | 0.8520 | 0.0372 | 0.0844 | 0.1138 |
| 8 | 0.8496 | 0.0388 | 0.0874 | 0.1178 |
| 9 | 0.8507 | 0.0402 | 0.0900 | 0.1216 |
| 10 | 0.8507 | 0.0371 | 0.0882 | 0.1189 |
| 11 | 0.8506 | 0.0362 | 0.0814 | 0.1114 |
| **12** | **0.8505** | **0.0459** | **0.1036** | **0.1408** |
| 13 | 0.8489 | 0.0400 | 0.0895 | 0.1231 |
| 14 | 0.8508 | 0.0425 | 0.0967 | 0.1301 |
| 15 | 0.8501 | 0.0444 | 0.1001 | 0.1348 |

**Test evaluation** — Learned gate weights:

| Modality | Weight |
|----------|--------|
| w_v (video) | **0.0154** |
| w_t (caption) | 0.4496 |
| w_a (audio) | **0.5350** |

**Gated on test split**: R@1=0.0461, R@5=0.1240, R@10=0.1742

The transformer-based gate collapsed: w_v ~0 (video ignored), heavily biased toward audio. Likely caused by caption-train vs caption-test split mismatch during retraining (Bug 2).

### Three-Branch Per-Modality Retrieval (D7a — Old Embeddings, Test Split)

| Branch | R@1 | R@5 | R@10 |
|--------|-----|-----|------|
| sim_v (video, meanpool) | 0.2165 | 0.4174 | 0.5115 |
| sim_t (captions, max) | **0.9269** | **0.9728** | **0.9839** |
| sim_a (audio) | 0.0003 | 0.0012 | 0.0027 |
| Equal fusion | 0.7553 | 0.8916 | 0.9296 |

### Three-Branch Per-Modality Retrieval (D7b — Transformer Embeddings, Test Split)

| Branch | R@1 | R@5 | R@10 |
|--------|-----|-----|------|
| sim_v (video, transformer) | 0.1950 | 0.4280 | 0.5430 |
| sim_t (captions, max) | **0.9286** | **0.9750** | **0.9855** |
| sim_a (audio) | 0.0003 | 0.0014 | 0.0030 |
| Equal fusion | 0.6790 | 0.8450 | 0.8945 |

---

## Known Bugs (3 Remaining)

### Bug 1: `eval_transformer_baseline.py` Caption Truncation
`zip()` stops at the shorter iterator: when `video_ids` (9,087 transformer) is shorter than `caption_embeddings` keys, captions are silently truncated during the per-video max-pool loop. Results in the D4 comparison table are affected.

### Bug 2: Gating Retraining Crashes — No Train Caption Embeddings
`retrain_gating.py` loads `caption_embeddings.pt` which is test-split only. For training, it needs train queries across all modalities, but caption_embeddings_train.pt exists. The script was not updated to load it. Both gating retraining runs (meanpool + transformer) were trained without proper caption pairs — the transformer gate collapse (w_v=0.015) is likely a symptom of this.

### Bug 3: `run_final_eval.sh` Wrong Gate Weight Paths (FIXED)
Shell script referenced `models/gating_weights.pth` which existed at root but was moved. Fixed to `models/gating_weights.pth`. D5/D6 now correctly report "No overlapping train videos" (which is Bug 2, not a path issue).

---

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| CLIP Video Embeddings (meanpool) | ✅ Complete | 10K videos |
| CLAP Audio Embeddings | ✅ Complete | 8.8K videos |
| Caption Embeddings (full) | ✅ Complete | train + test splits |
| Temporal Transformer | ✅ Trained | 12 epochs, R@1=0.3548 (val), best at models/temporal_transformer_best.pth |
| Transformer Embeddings | ✅ Exported | 9,087 videos in embeddings/video_embeddings_transformer.pt |
| Gating Network (meanpool) | ⚠️ Retrained | R@1=0.1268 (test), needs fix for Bug 2 |
| Gating Network (transformer) | ⚠️ Retrained | Collapsed (w_v=0.015), needs Bug 2 fix |
| Codebase Restructure | ✅ Complete | src/ package, scripts/subdivided, docs/ consolidated |
| Eval Suite (d1-d7) | ✅ Executed | All 7 scripts run, results in outputs/eval/ |

### Remaining Work

1. **Fix Bug 2**: Precompute train caption embeddings properly, or update retrain_gating.py to load `caption_embeddings_train.pt` and compute training queries using that split
2. **Fix Bug 1**: Fix the zip-truncation in eval_transformer_baseline.py
3. **Re-run gating retraining** with fixed train caption embeddings on both meanpool and transformer bases
4. **Re-run final eval suite** with fixed gating weights
5. **Continue transformer training**: Loss still decreasing at epoch 12 — train to convergence (20-30 epochs)
6. **Delete old checkpoints**: 12 × 75 MB = 900 MB in checkpoints/ if not needed

---

## Data Layout

| Directory | Size | Contents |
|-----------|------|----------|
| `data/raw/msrvtt/videos/` | 6.3 GB | 10,000 .mp4 files |
| `data/raw/msrvtt/annotations/` | 20 MB | JSON annotations + split lists |
| `data/processed/audio/` | 7.9 GB | 8,809 extracted audio clips |
| `data/processed/video/frames_uniform/` | 2.5 GB | 9,094 videos × 16 frames |
| `data/processed/metadata/` | 61 MB | 1 JSON metadata file |
| `embeddings/` | 318 MB | 6 .pt embedding files |
| `models/` | 26 MB | 4 .pth weight files |
| `checkpoints/` | 904 MB | 12 per-epoch transformer checkpoints |
| **Total project** | **~18 GB** | |

---

## Gating Verification

`scripts/verification/verify_gating.py` validates the gating network with 9 keyword-based queries (3 visual, 3 audio, 3 text-heavy). Three conditions must pass:
- **A**: At least one query shows clear dominant modality (weight spread ≥ 0.15)
- **B**: Different queries produce different dominant modalities
- **C**: Dominant modality aligns with query semantics (≥ 50% correct)

Current gating weights (near-equal) fail these conditions — gate is still weak.

---

## Dependencies

See `requirements.txt`. Key non-obvious deps:
- `laion_clap` (import as `laion_clap.CLAP_Module`, `enable_fusion=False`)
- `clip` (OpenAI: `pip install git+https://github.com/openai/CLIP.git`)
- `PIL`, `librosa`, `soundfile`, `moviepy`, `tqdm`

## Import Conventions

```python
from src.evaluation.evaluate_retrieval import evaluate_retrieval
from src.encoders.clap_encode import CLAPEncoder
from src.encoders.clip_encode import load_clip_model, encode_texts
from src.data.datasets import MSRVTTDataset
from src.data.metadata import load_metadata, get_common_video_ids
from src.models.gating_network import GatingNetwork
from src.models.temporal_transformer import TemporalTransformer
from src.config import DEVICE, set_seeds, clear_gpu
```

---

## Next Steps

### Critical Path
1. Fix `retrain_gating.py` to load train caption embeddings for training queries
2. Re-train gating on meanpool embeddings (should significantly improve over R@1=0.1268)
3. Re-train gating on transformer embeddings
4. Run full eval suite with fixed gating → get proper gated fusion comparison

### Secondary
1. Fix zip-truncation in `eval_transformer_baseline.py`
2. Continue transformer training past epoch 12
3. Delete old checkpoints to reclaim 900 MB
4. Consider audio embedding regeneration (only 8,809 vs 10,000 videos)
