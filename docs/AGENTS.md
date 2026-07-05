# AGENTS.md

## Quick start

```bash
source venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python scripts/<category>/<name>.py
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
│   │   └── clap_encode.py     # CLAPEncoder class
│   ├── data/
│   │   ├── datasets.py        # MSRVTTDataset (PyTorch Dataset)
│   │   └── metadata.py        # JSON loading, split filtering, common video ID utils
│   ├── evaluation/
│   │   └── evaluate_retrieval.py  # evaluate_retrieval(sim_matrix, ...)
│   ├── explainability/        # Explainability module
│   │   └── explain_retrieval.py
│   └── routing/               # Query routing
│       └── query_router.py
├── scripts/
│   ├── data/                  # Data preparation
│   ├── embeddings/            # Embedding precomputation
│   ├── baselines/             # Single-modality/fusion baselines
│   ├── training/              # Model training (gating, transformer)
│   ├── evaluation/            # Evaluation & orchestration
│   ├── queries/               # Query scripts with explainability
│   ├── demo/                  # Interactive demo
│   └── verification/          # Diagnostics & verification
├── configs/                   # YAML config files (future use)
├── data/                      # Raw & processed data (gitignored)
├── embeddings/                # Precomputed .pt files (gitignored)
├── checkpoints/               # Per-epoch checkpoints (gitignored)
├── outputs/                   # Eval results & logs (gitignored)
├── models/                    # Final trained weights (gitignored)
├── notebooks/                 # Jupyter notebooks
└── docs/                      # Documentation
```

## Pipeline order / Data flow

download → parse_captions → extract_frames (or extract_uniform_frames) → extract_audio → precompute_video/audio/caption embeddings → baselines/training → queries/demo

The query scripts (`scripts/queries/`) and demo (`scripts/demo/`) sit at the end of the pipeline. They load precomputed embeddings and trained gating weights, perform retrieval with explainability, and output ranked results with per-modality contribution breakdowns.

## Key scripts

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

## Import conventions

Always use the `src` package for reusable modules:
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

## Embedding DBs

Three dicts keyed by `video_id`, saved via `torch.save()` and loaded with `torch.load(..., weights_only=False)`:
- `embeddings/video_embeddings.pt` — CLIP visual, 512-dim per video (10K videos, mean-pooled)
- `embeddings/video_embeddings_transformer.pt` — Temporal Transformer output (9,087 videos)
- `embeddings/audio_embeddings.pt` — CLAP audio, 512-dim per video (8,809 videos)
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
- MLP: `Linear(512→128) → ReLU → Linear(128→3) → Softmax`
- Training: ranking loss with hard negatives (margin=0.2, 10 negatives/query)
- Constants: `NUM_EPOCHS=15`, `LR=1e-3`, `NUM_TRAIN_QUERIES=300`
- Saved to `models/gating_weights.pth`

## Temporal transformer

- Architecture: `src/models/temporal_transformer.py`
- Input: 16 uniform frames per video (from `frames_uniform/`)
- CLS token + learnable pos embed, 2-layer TransformerEncoder (4 heads, 512 hidden, 2048 FFN), Pre-LayerNorm, GeLU activation, dropout=0.1
- Trains on 85/15 random split from train set only
- Training: mixed precision (`torch.amp`), batch size=512, AdamW (1e-4, wd=0.01), InfoNCE loss (temperature=0.07)
- Checkpoints every epoch in `checkpoints/` + best to `models/temporal_transformer_best.pth`
- Output: `embeddings/video_embeddings_transformer.pt`

## Input data layout

`scripts/data/parse_msrvtt_captions.py` expects:
```
data/raw/msrvtt/
├── videos/              # MSR-VTT video files
└── annotations/
    ├── msrvtt_annotations.json
    ├── train_list.txt
    └── test_list.txt
```

Captions are lowercased during parsing. Both frame extraction scripts require `ffmpeg`.

## Dependencies

See `requirements.txt`. Key non-obvious ones:
- `laion_clap` (imported as `laion_clap.CLAP_Module`, `enable_fusion=False`)
- `clip` (OpenAI CLIP: `pip install git+https://github.com/openai/CLIP.git`)
- `PIL`, `librosa`, `soundfile`, `moviepy`, `tqdm`

## Code conventions

- No test suite, no linting, no typechecking
- Scripts import from `src.*` package, never from sibling scripts
- Pipeline order scripts are in `scripts/data/`, `scripts/embeddings/`, etc.
- All models in `src/models/`, all encoders in `src/encoders/`
- Shared config in `src/config.py`, file-specific overrides stay in the script
- Seeds: `set_seeds(42)` from `src.config`
- Always set `PYTHONPATH` before running: `export PYTHONPATH="${PYTHONPATH}:$(pwd)"`
