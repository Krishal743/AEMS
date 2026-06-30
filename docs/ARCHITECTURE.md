# Architecture

## System Overview

**Query-Conditioned Adaptive Fusion for Any-to-Any Video Retrieval** on MSR-VTT.

The system uses three precomputed embedding modalities — CLIP visual, CLAP audio, and CLIP caption — with a learned gating network that predicts per-query modality fusion weights.

```
                        ┌──────────────────┐
                        │  TEXT QUERY       │
                        └────────┬─────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │  CLIP Encode      │
                        │  (ViT-B/32)      │
                        └────────┬─────────┘
                                 │ 512-dim
                    ┌────────────┼────────────┐
                    │            │            │
                    ▼            ▼            ▼
            ┌──────────┐ ┌──────────┐ ┌──────────┐
            │ Gating   │ │ sim_v    │ │ sim_t    │
            │ Network  │ │(video)   │ │(caption) │
            │ MLP→3    │ │ CLIP     │ │ CLIP     │
            └────┬─────┘ └────┬─────┘ └────┬─────┘
                 │            │            │
                 └────────────┼────────────┘
                              ▼
                    ┌──────────────────┐
                    │ Weighted Fusion  │
                    │ w_v·sim_v +      │
                    │ w_t·sim_t +      │
                    │ w_a·sim_a        │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Top-K Retrieval  │
                    └──────────────────┘
```

## Components

### Encoders (`src/encoders/`)
- **CLIP** (`clip_encode.py`): OpenAI CLIP ViT-B/32 for video frames and text. `load_clip_model()`, `encode_videos()`, `encode_texts()`.
- **CLAP** (`clap_encode.py`): LAION CLAP for audio. `CLAPEncoder` class with `encode_audio()` and `encode_text()`.

### Models (`src/models/`)
- **GatingNetwork** (`gating_network.py`): MLP(512→128→ReLU→3→Softmax). ~66K parameters.
- **TemporalTransformer** (`temporal_transformer.py`): 2-layer TransformerEncoder over 16 frame embeddings. CLS token + learned pos embed.

### Evaluation (`src/evaluation/`)
- **evaluate_retrieval()**: Standard Recall@K (R@1, R@5, R@10).

## Data Flow

1. Download MSR-VTT → videos + annotations
2. Extract frames (fps=1, ~15/video OR 16 uniform/video)
3. Extract audio (.wav, 48kHz)
4. Precompute embeddings (CLIP for video/text, CLAP for audio)
5. Run baselines → train gating → train transformer → final eval
