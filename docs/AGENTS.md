# AGENTS.md

## Quick start

```bash
source venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python bin/<category>/<name>.py          # production scripts
python experiments/<area>/<name>.py      # research scripts
```

## Project structure

```
team23/
├── src/                       # Importable library package
│   ├── config.py              # Shared constants (paths, dims, hyperparams)
│   ├── models/
│   │   ├── gating_network.py  # GatingNetwork MLP (512→128→3, softmax)
│   │   └── temporal_transformer.py  # 2-layer TransformerEncoder
│   ├── encoders/
│   │   ├── clip_encode.py     # CLIP model loading, video/text encoding
│   │   ├── wavlm_encode.py    # WavLMEncoder class (audio branch)
│   │   └── clap_encode.py     # CLAPEncoder class (experiments/ only)
│   ├── data/
│   │   ├── datasets.py        # MSRVTTDataset (PyTorch Dataset)
│   │   └── metadata.py        # JSON loading, split filtering, common video ID utils
│   ├── evaluation/
│   │   └── evaluate_retrieval.py  # evaluate_retrieval(sim_matrix, ...)
│   ├── explainability/        # Explainability module
│   │   └── explain_retrieval.py
│   └── routing/               # Query routing
│       └── query_router.py
├── bin/                       # Production scripts (run directly)
│   ├── data/                  # Data preparation
│   ├── embeddings/            # Embedding precomputation
│   ├── training/              # Model training (gating, transformer)
│   ├── evaluation/            # Evaluation & orchestration
│   ├── queries/               # Query scripts with explainability
│   ├── demo/                  # Interactive demo
│   └── verification/          # Diagnostics & verification
├── experiments/               # Research code (not part of the pipeline)
│   ├── audio_alignment/       # Audio projection methods (M1-M6, LoRA, BEATs/WavLM/ImageBind)
│   ├── audio_optimization/    # Embedding & similarity improvements
│   ├── diagnostics/           # Anisotropy, hubness, modality-gap analysis
│   └── ablations/             # Ablation studies & weight sweeps
├── pipeline/                  # Pipeline orchestration (run_aems_pipeline.py)
├── tests/                     # Pytest suite
├── data/                      # Raw & processed data (gitignored)
├── embeddings/                # Precomputed .pt files (gitignored)
├── checkpoints/               # Per-epoch checkpoints (gitignored)
├── outputs/                   # Eval results & logs (gitignored)
├── models/                    # Final trained weights (gitignored)
└── docs/                      # Documentation
```

## Pipeline order / Data flow

build_manifest → extract_frames → extract_audio → precompute_video/audio/text embeddings → train_temporal_transformer → export_transformer_embeddings → train_gating_network → eval_aems_retrieval → queries/demo

`pipeline/run_aems_pipeline.py` orchestrates all of the above; run it with `--stage <name>` to execute a single stage. The MSR-VTT-era data-prep and baseline scripts were removed in commit `102e67b`.

The query scripts (`bin/queries/`) and demo (`bin/demo/`) sit at the end of the pipeline. They load precomputed embeddings and trained gating weights, perform retrieval with explainability, and output ranked results with per-modality contribution breakdowns.

## Key scripts

| Category | Script | Purpose |
|---|---|---|
| **Queries** | `bin/queries/query_text.py` | Text query with explainability |
| | `bin/queries/query_image.py` | Image query with explainability |
| | `bin/queries/query_audio.py` | Audio query with explainability |
| | `bin/queries/query_video.py` | Video query with explainability |
| | `bin/queries/query_mixed.py` | Mixed text+image query |
| **Evaluation** | `bin/evaluation/final_eval.py` | Unified 5-system evaluation |
| | `bin/evaluation/ablation_study.py` | 11-way ablation study |
| | `bin/evaluation/behavioural_test.py` | Gating behaviour verification |
| **Demo** | `bin/demo/demo.py` | Interactive text query demo with explanations |
| **Data prep** | `bin/data/build_manifest.py` | Build AEMS manifest (`--pilot` for a 500-video subset) |
| | `bin/data/extract_frames.py` | Exactly 16 uniform frames per video |
| | `bin/data/extract_audio.py` | Three 10s audio segments per video |
| **Embeddings** | `bin/embeddings/precompute_video_embeddings.py` | CLIP encode → `aems_video_embeddings_v1.pt` |
| | `bin/embeddings/precompute_audio_embeddings.py` | WavLM-Large → `aems_audio_embeddings_wavlm_v1.pt` (1024-d) |
| | `bin/training/train_audio_adapter.py` | WavLM → CLIP adapter → `aems_audio_embeddings_wavlm_clip_v1.pt` (audio branch) |
| | `bin/embeddings/precompute_text_embeddings.py` | CLIP text encode; `--fusion description\|transcript\|fused` |
| | `bin/embeddings/precompute_text_chunks.py` | Per-passage CLIP text encode → `aems_text_chunks_{split}.pt` |
| **Models** | `bin/training/train_gating_network.py` | **Main gating network** (train + eval, ranking loss) |
| | `bin/training/train_temporal_transformer.py` | 2-layer transformer over 16 frames, InfoNCE loss |
| | `bin/training/export_transformer_embeddings.py` | Export transformer embeddings to `.pt` |
| | `bin/training/cleanup_checkpoints.py` | Prune per-epoch checkpoints, keep best + last |
| **Pipeline** | `pipeline/run_aems_pipeline.py` | Orchestrates every stage above |
| **Verification** | `bin/verification/verify_gating.py` | Check gating weights vs keyword expectations |
| | `bin/verification/diagnose_loss.py` | Debug loss/weight behavior during gating training |

## Experimental protocol

See `docs/PROTOCOL.md`: selection happens on the validation split carved from
train, test is scored once, any model whose scores train a downstream model must
be cross-fitted, and a difference is claimed only when a paired bootstrap CI
excludes zero.

## Import conventions

Always use the `src` package for reusable modules:
```python
from src.evaluation.evaluate_retrieval import evaluate_retrieval
from src.encoders.wavlm_encode import WavLMEncoder
from src.encoders.clip_encode import load_clip_model, encode_texts
from src.data.datasets import MSRVTTDataset
from src.data.metadata import load_metadata, get_common_video_ids
from src.models.gating_network import GatingNetwork
from src.models.audio_adapter import AudioAdapter, load_audio_adapter
from src.models.temporal_transformer import TemporalTransformer
from src.config import DEVICE, set_seeds
from src.explainability.explain_retrieval import explain_modality_contributions, explain_gating_decision, format_explanation
from src.routing.query_router import load_search_index, load_gate, search, zscore, fixed_weights
```

## Embedding DBs

Dicts keyed by `video_id`, saved via `torch.save()` and loaded with `torch.load(..., weights_only=False)`:
- `embeddings/video_embeddings.pt` — CLIP visual, 512-dim per video (10K videos, mean-pooled)
- `embeddings/video_embeddings_transformer.pt` — Temporal Transformer output (9,087 videos)
- `embeddings/aems_text_chunks_{train,test}.pt` — per-passage CLIP text embeddings, `(n_chunks, 512)` per video (late-interaction branch)
- `embeddings/aems_audio_embeddings_wavlm_v1.pt` — raw WavLM-Large audio features, 1024-dim (adapter input, not searchable)
- `embeddings/aems_audio_embeddings_wavlm_clip_v1.pt` — **the audio branch**: WavLM projected into CLIP text space, 512-dim
- `embeddings/aems_audio_embeddings_v1.pt` — legacy CLAP audio, 512-dim (CLAP space; `experiments/` only, via `AEMS_CLAP_AUDIO_EMBEDDINGS_PATH`)
- `embeddings/caption_embeddings.pt` — CLIP text, 20×512-dim per video (MAX-aggregated at query time, test split only)
- `embeddings/caption_embeddings_train.pt` — Train split caption embeddings (when available)
- `embeddings/caption_embeddings_test.pt` — Test split caption embeddings

## Memory discipline (6GB GPU constraint)

Pattern used in gating training:
- Embeddings stay on CPU permanently
- CLIP queries: batch-encode on GPU, `.cpu()` immediately
- Similarity computed block-wise (`QUERY_BATCH_SIZE=32`, `VIDEO_BATCH_SIZE=100`)
- `del` + `gc.collect()` + `torch.cuda.empty_cache()` after each batch
- `.float()` on all GPU transfers (avoids float16/32 mismatch)

## Gating network

- Architecture: `src/models/gating_network.py`
- MLP: `LayerNorm(512) → Linear(512→128) → GELU → LayerNorm → Dropout → Linear(128→64) → GELU → LayerNorm → Dropout → Linear(64→4) → Softmax` (one weight per branch in `BRANCHES`; `load_gate` rejects a checkpoint whose head size disagrees)
- Optional modes: `constant_weights` (one global weight vector), `learnable_temp`, `learnable_scale`, `weight_reg`
- `GatingNetworkPerCandidate` is a second class that also consumes per-candidate similarities, predicting weights per query-candidate pair; reachable via `experiments/ablations/run_ablation.py --method per_candidate`
- Training: ranking loss with hard negatives (margin=0.2, 10 negatives/query)
- Constants: `NUM_EPOCHS=15`, `LR=1e-3`, `NUM_TRAIN_QUERIES=300`
- Saved to `models/aems_gating_weights_v1.pth`

## Temporal transformer

- Architecture: `src/models/temporal_transformer.py`
- Input: 16 uniform frames per video (from `frames_uniform/`)
- CLS token + learnable pos embed, 2-layer TransformerEncoder (4 heads, 512 hidden, 2048 FFN), Pre-LayerNorm, GeLU activation, dropout=0.1
- Trains on 85/15 random split from train set only
- Training: mixed precision (`torch.amp`), batch size=512, AdamW (1e-4, wd=0.01), InfoNCE loss (temperature=0.07)
- Checkpoints every epoch in `checkpoints/` + best to `models/temporal_transformer_best.pth`
- Output: `embeddings/video_embeddings_transformer.pt`

## Input data layout

`bin/data/build_manifest.py` walks `AEMS_DATASET_ROOT` (`aems/dataset`), one directory per category:
```
aems/dataset/
└── <category>/
    ├── <video_id>.json   # metadata (description, transcript, Q&A)
    └── <video_id>.mp4    # source video
```

It writes a stratified 85/15 per-category split to `data/processed/aems/metadata/aems_manifest_v1.json`
(or `data/processed/aems_pilot/...` under `--pilot`). Frame extraction requires `ffmpeg`.

## Dependencies

See `requirements.txt`. Key non-obvious ones:
- `torchaudio` (`torchaudio.pipelines.WAVLM_LARGE` for the audio branch)
- `laion_clap` (imported as `laion_clap.CLAP_Module`, `enable_fusion=False`) — `experiments/` only
- `clip` (OpenAI CLIP: `pip install git+https://github.com/openai/CLIP.git`)
- `PIL`, `librosa`, `soundfile`, `moviepy`, `tqdm`

## Code conventions

- Pytest suite in `tests/`; no linting, no typechecking
- Scripts import from `src.*` package, never from sibling scripts
- Pipeline order scripts are in `bin/data/`, `bin/embeddings/`, etc.
- All models in `src/models/`, all encoders in `src/encoders/`
- Shared config in `src/config.py`, file-specific overrides stay in the script
- Seeds: `set_seeds(42)` from `src.config`
- Always set `PYTHONPATH` before running: `export PYTHONPATH="${PYTHONPATH}:$(pwd)"`
