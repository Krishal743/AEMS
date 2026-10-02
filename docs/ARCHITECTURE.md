# Architecture

## System Overview

**Query-Conditioned Adaptive Fusion for Any-to-Any Video Retrieval** on AEMS.

The system indexes each video six ways and fuses the branches with **fixed
weights** (`AEMS_FUSION_WEIGHTS`), after z-scoring each branch's similarities per
query over the gallery.

| Branch | Encoder | Weight | R@1 alone (test) |
|---|---|---|---|
| dense | E5-base passages (retrieval-trained) | 1.0 | 0.624 |
| bm25 | lexical over passages + title/description/tags | 0.5 | 0.550 |
| text | CLIP caption (description + transcript, mean-pooled) | 0.25 | 0.389 |
| visual | CLIP ViT-B/32 over 16 frames, mean-pooled | 0.25 | 0.194 |
| audio | WavLM-Large → adapter → CLIP text space | 0.25 | 0.053 |
| chunk | CLIP passages, best-match | 0.0 (measured redundant) | 0.365 |

Four branches share CLIP text space (the adapter projects audio into it), while
BM25 has no vector space at all and E5 has its own; z-scoring is what makes them
comparable, not shared geometry.

**On "adaptive":** the gating network predicts per-query weights and is still
available via `--fusion gate`, but it loses to tuned fixed weights on both splits
(validation 0.7075 vs 0.7100; test 0.6865 vs 0.6900), so static weights are
deployed. Query-dependence remains in the per-query z-scoring and in the
cross-encoder, but branch weighting is not currently adaptive.

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
             │ Weights  │ │ sim_v    │          │ sim_t    │
             │ fixed or │ │(visual)  │          │(caption) │
             │ gate MLP │ │ CLIP     │          │ CLIP mean│
             └────┬─────┘ └────┬─────┘          └────┬─────┘
                  │     ┌──────┴──────────┐          │
                  │     │ sim_c (passage) │          │
                  │     │ CLIP max-sim    │          │
                  │     ├─────────────────┤          │
                  │     │ sim_a (audio)   │          │
                  │     │ WavLM+adapter   │          │
                  └─────┴───────┬─────────┴──────────┘
                                │
                                ▼
                     ┌──────────────────┐
                     │ z-score each     │
                     │ branch per query,│
                     │ then w_v·sim_v + │
                     │ w_t·sim_t +      │
                     │ w_c·sim_c +      │
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
- **WavLM** (`wavlm_encode.py`): torchaudio WavLM-Large for audio. `WavLMEncoder` with `encode_wave()`/`encode_file()`; three fixed 10 s segments at 16 kHz, last-layer mean pooling, 1024-d.
- **CLAP** (`clap_encode.py`): LAION CLAP. No longer part of the search path; retained for `experiments/`.

### Models (`src/models/`)
- **GatingNetwork** (`gating_network.py`): MLP(512→128→ReLU→3→Softmax). ~66K parameters.
- **AudioAdapter** (`audio_adapter.py`): MLP with a linear skip path mapping 1024-d WavLM features into 512-d CLIP text space, trained by `bin/training/train_audio_adapter.py`.
- **TemporalTransformer** (`temporal_transformer.py`): 2-layer TransformerEncoder over 16 frame embeddings. CLS token + learned pos embed.

### Routing (`src/routing/`)
- **ChunkIndex** (`query_router.py`): every video's passage embeddings stacked with an owner index; `max_sim` scores a query against each video's best-matching passage. CLIP takes 77 tokens while transcripts run to a median of 677 words, so the mean-pooled caption vector averages ~30 chunks together; keeping both views is worth ~5 R@1 points over either alone.
- **Reranking** (`src/rerank/`): optional stage 2 over stage 1's shortlist.
  `stage1.py` builds candidate lists and folds stage-2 scores back into a full
  ranking; `per_candidate.py` predicts fusion weights per query-candidate pair;
  `cross_encoder.py` scores (query, passage) pairs with a pretrained
  cross-encoder. Off by default (`--rerank none`).
- **Query Router** (`query_router.py`): per-query-type encoding (text/image/audio/video/mixed), per-branch similarities, z-scoring, and weighted fusion. Text queries score all three branches with one CLIP vector; audio-clip queries use WavLM + adapter and reach the audio branch only; image and video queries skip the audio branch, since image vectors are not aligned with the adapter's text space. Branches a query cannot reach get zero weight and the rest are renormalized.

### Explainability (`src/explainability/`)
- **Explain Retrieval** (`explain_retrieval.py`): `explain_gating_decision()` — dominant modality and confidence spread; `explain_modality_contributions()` — per-video visual/caption/audio breakdown; `explain_ranking_difference()` — deciding modality between ranked results; `format_explanation()` — printable output.

### Evaluation (`src/evaluation/`)
- **evaluate_retrieval()**: Standard Recall@K (R@1, R@5, R@10).

## Data Flow

1. Build the AEMS manifest from `aems/dataset/`
2. Extract 16 uniform frames per video
3. Extract audio (.wav)
4. Precompute embeddings (CLIP for video/text/passages, WavLM features for audio)
5. Train the audio adapter (exports the CLIP-space audio branch) → optionally train the transformer and the gating network → canonical eval
6. Query scripts (`bin/queries/`) for text/image/audio/video/mixed retrieval with explainability
7. Demo script (`bin/demo/demo.py`) for interactive demonstration
