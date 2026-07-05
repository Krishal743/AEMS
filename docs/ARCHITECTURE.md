# Architecture

## System Overview

**Query-Conditioned Adaptive Fusion for Any-to-Any Video Retrieval** on MSR-VTT.

The system uses three precomputed embedding modalities — CLIP visual, CLAP audio, and CLIP caption — with a learned gating network that predicts per-query modality fusion weights.

```
                         ┌──────────────────┐
                         │   QUERY INPUT     │
                         │ (text/image/audio │
                         │  /video/mixed)    │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │  Query Router     │
                         │ detect_query_type │
                         │ + multi-modal     │
                         │   encode          │
                         └────────┬─────────┘
                                  │ 512-dim
                     ┌────────────┼──────────────────────┐
                     │            │                      │
                     ▼            ▼                      ▼
             ┌──────────┐ ┌──────────┐          ┌──────────┐
             │ Gating   │ │ sim_v    │          │ sim_t    │
             │ Network  │ │(visual)  │          │(caption) │
             │ MLP→3    │ │ CLIP     │          │ CLIP     │
             └────┬─────┘ └────┬─────┘          └────┬─────┘
                  │     ┌──────┴──────────┐          │
                  │     │ sim_a (audio)   │          │
                  │     │ CLAP            │          │
                  └─────┴───────┬─────────┴──────────┘
                                │
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
                     └────────┬─────────┘
                              │
                              ▼
                     ┌────────────────────────┐
                     │   Explainability       │
                     │ · gating decision      │
                     │ · modality contribs    │
                     │ · ranking difference   │
                     └────────────────────────┘
```

## Components

### Encoders (`src/encoders/`)
- **CLIP** (`clip_encode.py`): OpenAI CLIP ViT-B/32 for video frames and text. `load_clip_model()`, `encode_videos()`, `encode_texts()`.
- **CLAP** (`clap_encode.py`): LAION CLAP for audio. `CLAPEncoder` class with `encode_audio()` and `encode_text()`.

### Models (`src/models/`)
- **GatingNetwork** (`gating_network.py`): MLP(512→128→ReLU→3→Softmax). ~66K parameters.
- **TemporalTransformer** (`temporal_transformer.py`): 2-layer TransformerEncoder over 16 frame embeddings. CLS token + learned pos embed.

### Routing (`src/routing/`)
- **Query Router** (`query_router.py`): Query type detection (text/image/audio/video/mixed), multi-modal encoding via CLIP/CLAP, and per-modality similarity computation.

### Explainability (`src/explainability/`)
- **Explain Retrieval** (`explain_retrieval.py`): `explain_gating_decision()` — dominant modality and confidence spread; `explain_modality_contributions()` — per-video visual/caption/audio breakdown; `explain_ranking_difference()` — deciding modality between ranked results; `format_explanation()` — printable output.

### Evaluation (`src/evaluation/`)
- **evaluate_retrieval()**: Standard Recall@K (R@1, R@5, R@10).

## Data Flow

1. Download MSR-VTT → videos + annotations
2. Extract frames (fps=1, ~15/video OR 16 uniform/video)
3. Extract audio (.wav, 48kHz)
4. Precompute embeddings (CLIP for video/text, CLAP for audio)
5. Run baselines → train gating → train transformer → final eval
6. Query scripts (`scripts/queries/`) for text/image/audio/video/mixed retrieval with explainability
7. Demo script (`scripts/demo/demo.py`) for interactive demonstration
