# Project Checkpoint: Query-Conditioned Adaptive Video Retrieval

**Project Name**: Query-Conditioned Adaptive Fusion for Any-to-Any Video Retrieval
**Status**: Phase 0–7 Complete — Ablation Study + Behavioural Verification Done
**Last Updated**: July 2026
**Platform**: Linux (NVIDIA RTX 4500 Ada Generation, 24 GB VRAM)
**Original Platform**: Windows (RTX 3050, 6 GB VRAM)

---

## Executive Summary

Video retrieval system using CLIP (visual), CLAP (audio), and caption embeddings with a learned gating network for query-adaptive modality fusion. The project progressed through Phases 0–7: data pipeline → embedding precomputation → temporal transformer (R@1=0.3548 val) → gating network retraining (meanpool R@1=0.1268, transformer collapsed at R@1=0.0461) → 3-branch per-modality eval → ablation study → behavioural verification.

**Key finding**: The gating network collapse (w_v=0.015 on transformer embeddings) was **not** caused by the known caption-split bug (Bug 2). Re-training the gating with properly split caption pairs produced identical collapsed weights, confirming the collapse is an architectural limitation. Behavioural verification confirmed the gate does not adapt per-query semantics — only 1/12 test queries matched the expected dominant modality. Audio retrieval remains near-zero (R@1=0.0003).

A novel **adaptive gating without audio** variant achieves the best retrieval results: R@1=0.2697, R@5=0.4822, R@10=0.5745.

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
├── evaluation/
│   └── evaluate_retrieval.py  # evaluate_retrieval(sim_matrix, ...)
├── routing/
│   └── query_router.py    # Query type detection, modal similarity, gating
└── explainability/
    └── explain_retrieval.py  # Per-result modality contribution breakdown
```

```
scripts/
├── queries/               # Interactive query scripts (text/image/audio/video/mixed)
├── demo/                  # Interactive demo
├── data/                  # Data preparation
├── embeddings/            # Embedding precomputation
├── baselines/             # Single-modality/fusion baselines
├── training/              # Model training (gating, transformer)
├── evaluation/            # Evaluation & orchestration
└── verification/          # Diagnostics & verification
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

The transformer-based gate collapsed: w_v ~0 (video ignored), heavily biased toward audio. Re-training with properly split caption pairs confirmed this is an architectural limitation, not a data bug.

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

### Ablation Study

11-way ablation across modality combinations on the D7a test split. Key results:

| System | R@1 | R@5 | R@10 |
|--------|-----|-----|------|
| Visual only | 0.2165 | 0.4173 | 0.5116 |
| Caption only | 0.1159 | 0.2678 | 0.3475 |
| Audio only | 0.0003 | 0.0012 | 0.0026 |
| Equal fusion (all 3) | 0.1711 | 0.3465 | 0.4322 |
| Adaptive gating | 0.1177 | 0.2544 | 0.3322 |
| Adaptive no audio | **0.2697** | **0.4822** | **0.5745** |

Adaptive gating without audio is the best learned-fusion system — it beats visual-only and caption-only at all recall levels, confirming that the gating network does add value when not forced to allocate weight to the near-zero audio modality.

### Behavioural Verification

Evaluated whether the gating network adapts per-query semantics:

- 12 queries across 4 semantic sets (3 visual, 3 audio, 3 text-heavy, 3 mixed)
- Only **1/12** queries correctly matched the expected dominant modality
- **Conclusion**: Gating network does **not** adapt per query semantics — weights are effectively query-agnostic

---

## Scripts Reference

| Category | Script | Purpose |
|---|---|---|
| **Queries** | `scripts/queries/query_text.py` | Text query with explainability |
| | `scripts/queries/query_image.py` | Image query with explainability |
| | `scripts/queries/query_audio.py` | Audio query with explainability |
| | `scripts/queries/query_video.py` | Video query with explainability |
| | `scripts/queries/query_mixed.py` | Mixed text+image query |
| **Evaluation** | `scripts/evaluation/final_eval.py` | Unified 5-system evaluation |
| | `scripts/evaluation/ablation_study.py` | 11-way ablation study |
| | `scripts/evaluation/behavioural_test.py` | Gating behaviour verification |
| **Demo** | `scripts/demo/demo.py` | Interactive text query demo with explanations |
| **Data prep** | `scripts/data/download_msrvtt.py` | Download MSR-VTT from HuggingFace |
| | `scripts/data/parse_msrvtt_captions.py` | Parse annotations, build metadata JSON |
| | `scripts/data/extract_frames_msrvtt.py` | Frames via ffmpeg (fps=1, max=15) |
| | `scripts/data/extract_uniform_frames.py` | Exactly 16 uniform frames for transformer |
| | `scripts/data/extract_audio_msrvtt.py` | Audio via moviepy + librosa |
| **Embeddings** | `scripts/embeddings/precompute_video_embeddings.py` | CLIP encode → `video_embeddings.pt` |
| | `scripts/embeddings/precompute_audio_embeddings.py` | CLAP encode → `audio_embeddings.pt` |
| | `scripts/embeddings/precompute_caption_embeddings.py` | CLIP encode → `caption_embeddings.pt` |
| **Models** | `scripts/training/run_query_routing.py` | **Main gating network** (train + eval, ranking loss) |
| | `scripts/training/train_temporal_transformer.py` | 2-layer transformer over 16 frames, InfoNCE loss |
| **Baselines** | `scripts/baselines/run_clip_baseline.py` | CLIP visual-only baseline |
| | `scripts/baselines/run_clap_baseline.py` | CLAP audio-only baseline |
| | `scripts/baselines/run_fusion_baseline.py` | Equal-weight fusion baseline |
| | `scripts/baselines/run_three_branch.py` | All 3 branches + equal fusion |
| **Verification** | `scripts/verification/verify_gating.py` | Check gating weights vs keyword expectations |
| | `scripts/verification/diagnose_loss.py` | Debug loss/weight behavior during gating training |

---

## Known Bugs

### Bug 1: `eval_transformer_baseline.py` Caption Truncation
`zip()` stops at the shorter iterator: when `video_ids` (9,087 transformer) is shorter than `caption_embeddings` keys, captions are silently truncated during the per-video max-pool loop. Results in the D4 comparison table are affected.

### Bug 2: Gating Retraining Crashes — No Train Caption Embeddings
`retrain_gating.py` loads `caption_embeddings.pt` which is test-split only. For training, it needs train queries across all modalities, but caption_embeddings_train.pt exists. The script was not updated to load it. Both gating retraining runs (meanpool + transformer) were trained without proper caption pairs — the transformer gate collapse (w_v=0.015) is likely a symptom of this.

**Status: Resolved — not root cause.** Re-training the gating network with properly split caption pairs (train vs test) produced identical collapsed weights (w_v ≈ 0.015, w_a ≈ 0.535). The gating collapse on transformer embeddings is confirmed to be an architectural limitation, not a data bug.

### Bug 3: `run_final_eval.sh` Wrong Gate Weight Paths ✅ Fixed
Shell script referenced `models/gating_weights.pth` which existed at root but was moved. Fixed to `models/gating_weights.pth`. D5/D6 now correctly report "No overlapping train videos".

### Bug 4: Audio Retrieval Near-Zero R@1
Audio retrieval R@1 ≈ 0.0003 — 12% of test entries had no matching audio embeddings. Partially mitigated by filtering test entries to only include videos present in `audio_embeddings.pt`. Root cause may be a CLAP encoding issue.

### Bug 5: LEXICON_LOGGING Undefined ✅ Fixed
A variable `LEXICON_LOGGING` was used in the curriculum phase check but never defined. Fixed during restructuring — defaults to `False`.

---

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| CLIP Video Embeddings (meanpool) | ✅ Complete | 10K videos |
| CLAP Audio Embeddings | ✅ Complete | 8.8K videos |
| Caption Embeddings (full) | ✅ Complete | train + test splits |
| Temporal Transformer | ✅ Trained | 12 epochs, R@1=0.3548 (val), best at models/temporal_transformer_best.pth |
| Transformer Embeddings | ✅ Exported | 9,087 videos in embeddings/video_embeddings_transformer.pt |
| Gating Network (meanpool) | ⚠️ Retrained | R@1=0.1268 (test), limited by architecture |
| Gating Network (transformer) | ⚠️ Retrained | Collapsed (w_v=0.015), confirmed architecture issue |
| Ablation Study | ✅ Complete | 11-way, adaptive no audio best (R@1=0.2697) |
| Behavioural Verification | ✅ Complete | Gate does not adapt per-query semantics |
| Query Scripts | ✅ Complete | text/image/audio/video/mixed with explainability |
| Interactive Demo | ✅ Complete | Text query demo with per-result breakdown |
| Codebase Restructure | ✅ Complete | src/ package, scripts/subdivided, docs/ consolidated |
| Eval Suite (d1-d7) | ✅ Executed | All 7 scripts run, results in outputs/eval/ |

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
from src.explainability.explain_retrieval import explain_modality_contributions, explain_gating_decision, format_explanation
from src.routing.query_router import compute_modal_similarities
```

---

## Next Steps

### Critical Path
1. Redesign gating architecture — current MLP produces query-agnostic weights. Consider cross-attention between query and modality embeddings.
2. Fix audio embedding quality — CLAP-based audio retrieval is essentially non-functional (R@1=0.0003).
3. Continue transformer training past epoch 12 — loss is still decreasing.

### Secondary
1. Fix zip-truncation in `eval_transformer_baseline.py`
2. Delete old checkpoints to reclaim 900 MB
3. Build video query demo with frame extraction pipeline
