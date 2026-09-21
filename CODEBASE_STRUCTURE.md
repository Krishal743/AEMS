# Codebase Structure Guide

This document describes the organization of the team23 AEMS/finevideo retrieval codebase.

## Directory Organization

### 📚 `src/` - Core Library Code
The foundational library for AEMS retrieval system.

```
src/
├── models/              # Neural network architectures
│   ├── gating_network.py        # Multimodal gating network
│   └── temporal_transformer.py   # Temporal transformer for video frames
├── encoders/            # Feature extraction
│   ├── clip_encode.py           # CLIP encoder (vision & text)
│   └── clap_encode.py           # CLAP encoder (audio)
├── data/                # Data loading & processing
│   ├── aems_dataset.py          # AEMS dataset class
│   ├── datasets.py              # Generic dataset utilities
│   └── metadata.py              # Metadata utilities
├── evaluation/          # Evaluation utilities
│   └── evaluate_retrieval.py     # Retrieval metrics
├── routing/             # Query routing & multimodal fusion
│   └── query_router.py          # Query encoding & similarity computation
├── explainability/      # Interpretability tools
│   └── explain_retrieval.py      # Retrieval explanation
└── config.py            # Configuration constants
```

### 🚀 `pipeline/` - Pipeline Orchestration
Main entry points for running the AEMS pipeline.

```
pipeline/
└── run_aems_pipeline.py   # Main 8-stage AEMS pipeline orchestrator
```

**Usage:** `python pipeline/run_aems_pipeline.py`

### 📦 `bin/` - Executable Scripts (Runnable Entrypoints)
Production-ready scripts for data processing, training, evaluation, and inference.

```
bin/
├── data/                # Data preparation
│   ├── build_manifest.py        # Build AEMS dataset manifest
│   ├── extract_frames.py        # Extract video frames (16 uniform)
│   └── extract_audio.py         # Extract & segment audio
│
├── embeddings/          # Feature embedding generation
│   ├── precompute_video_embeddings.py    # CLIP video embeddings
│   ├── precompute_audio_embeddings.py    # CLAP audio embeddings (3 segments)
│   ├── precompute_text_embeddings.py     # CLIP text embeddings
│   └── *_legacy.py                       # Legacy/reference implementations
│
├── training/            # Model training & export
│   ├── train_temporal_transformer.py     # Train video frame transformer
│   ├── train_gating_network.py          # Train multimodal gating network
│   ├── export_transformer_embeddings.py  # Export transformer features
│   └── cleanup_checkpoints.py           # Checkpoint management
│
├── evaluation/          # Evaluation & validation
│   ├── eval_aems_retrieval.py           # Main retrieval evaluation
│   ├── evaluate_priority1.py            # Priority 1 optimizations eval
│   ├── behavioural_test.py              # Behavioral verification
│   ├── ablation_study.py                # Ablation study evaluation
│   ├── final_eval.py                    # Final comprehensive evaluation
│   └── run_final_eval.sh                # Evaluation script
│
├── queries/             # Query interfaces & utilities
│   ├── query_text.py        # Text query interface
│   ├── query_audio.py       # Audio query interface
│   ├── query_video.py       # Video query interface
│   ├── query_image.py       # Image query interface
│   └── query_mixed.py       # Multimodal query interface
│
├── verification/        # Validation & debugging tools
│   ├── verify_gating.py          # Gating network verification
│   └── diagnose_loss.py          # Loss diagnostics
│
└── demo/               # Demonstration scripts
    ├── demo.py                 # Interactive demo
    └── run_demo.sh             # Demo launcher
```

**All scripts in `bin/` are production-ready and can be run directly.**

### 🔬 `experiments/` - Research & Experimental Work
Scripts for research, ablation studies, and optimization experiments. These are not part of the main pipeline.

```
experiments/
├── audio_alignment/         # Audio alignment research
│   ├── build_beats_features.py       # BEATs audio features
│   ├── build_imagebind_features.py   # ImageBind audio features
│   ├── build_wavlm_features.py       # WavLM audio features
│   ├── fit_projection.py             # Audio projection learning
│   ├── validate_projection.py        # Projection validation
│   ├── compare_methods.py            # Method comparison
│   ├── compare_all_methods.py        # Comprehensive comparison
│   └── ... (other alignment experiments)
│
├── audio_optimization/      # Audio embedding improvements
│   ├── improve_audio_embeddings.py          # Embedding optimization
│   ├── improve_audio_similarity.py          # Similarity metric tuning
│   ├── improve_audio_text_alignment.py      # Cross-modal alignment
│   ├── diagnose_audio_embeddings.py         # Audio diagnostics
│   ├── test_audio_alternatives.py           # Alternative methods
│   └── evaluate_comparison.py               # Comparison evaluation
│
├── diagnostics/             # Analysis & debugging scripts
│   ├── run_diagnostics.py                  # Diagnostic suite
│   ├── diagnose_alignment_uniformity.py    # Alignment analysis
│   ├── diagnose_anisotropy.py              # Anisotropy diagnostics
│   ├── diagnose_data_quality.py            # Data quality checks
│   ├── diagnose_hubness.py                 # Hubness analysis
│   ├── diagnose_modality_gap.py            # Modality gap analysis
│   ├── diagnose_retrieval_metrics.py       # Metrics analysis
│   ├── eval_leave_one_out.py               # LOO evaluation
│   ├── eval_loo_gate.py                    # LOO gating eval
│   └── error_analysis.py                   # Error analysis
│
└── ablations/               # Ablation studies
    ├── run_ablation.py              # Ablation runner
    ├── phase0_heuristics.py         # Heuristic baseline
    ├── phase5_best.py               # Best phase evaluation
    ├── run_multiseed.py             # Multi-seed training
    ├── run_c5_tethered.py           # Tethered variant
    ├── sweep_const_weights.py       # Constant weight sweep
    └── sweep_train_weights.py       # Training weight sweep
```

### 📁 Data & Results Directories

```
data/                    # Dataset storage
├── raw/                 # Raw input data
├── processed/           # Processed data
│   └── aems/
│       ├── frames_uniform/      # Extracted video frames (16/video)
│       ├── audio/               # Extracted audio segments
│       └── metadata/            # Manifest & annotations
└── skills/              # Auxiliary data

checkpoints/             # Training checkpoints
├── aems/                # AEMS model checkpoints

models/                  # Final trained models
├── aems_gating_weights_v1.pth
├── aems_temporal_transformer_best_v1.pth

embeddings/             # Precomputed embeddings
├── aems_video_embeddings_v1.pt
├── aems_audio_embeddings_v1.pt
├── aems_video_embeddings_transformer_v1.pt
└── ...

outputs/                # Experiment results
├── aems/                # Main AEMS evaluation results
├── ablations/           # Ablation study results
├── alignment/           # Audio alignment experiment results
├── diagnostics/         # Diagnostic output
├── eval/                # Evaluation results
├── behavioural/         # Behavioral test results
└── logs/                # Training logs

logs/                   # Additional logs
```

### 📖 Documentation & Tests

```
docs/                   # Documentation
├── *.md files          # Design docs, approach docs, analysis

tests/                  # Unit tests
├── test_*.py           # Test files

CODEBASE_STRUCTURE.md   # This file
README.md               # Main README
setup.py               # Package setup
requirements.txt       # Dependencies
```

## Workflow Guide

### Running the Complete Pipeline
```bash
# Run full AEMS pipeline (all 8 stages)
python pipeline/run_aems_pipeline.py
```

### Data Preparation
```bash
# Step 1: Build manifest
python bin/data/build_manifest.py

# Step 2: Extract frames
python bin/data/extract_frames.py

# Step 3: Extract audio
python bin/data/extract_audio.py
```

### Feature Extraction
```bash
# Precompute embeddings
python bin/embeddings/precompute_video_embeddings.py
python bin/embeddings/precompute_audio_embeddings.py
python bin/embeddings/precompute_text_embeddings.py
```

### Model Training
```bash
# Train temporal transformer
python bin/training/train_temporal_transformer.py

# Export transformer features
python bin/training/export_transformer_embeddings.py

# Train gating network
python bin/training/train_gating_network.py
```

### Evaluation
```bash
# Main retrieval evaluation
python bin/evaluation/eval_aems_retrieval.py

# Full evaluation suite
bash bin/evaluation/run_final_eval.sh
```

### Querying
```bash
# Text query
python bin/queries/query_text.py --query "your query"

# Audio query
python bin/queries/query_audio.py --audio audio.wav

# Multimodal query
python bin/queries/query_mixed.py --text "query" --image image.jpg
```

### Research/Experiments
```bash
# Run diagnostics
python experiments/diagnostics/run_diagnostics.py

# Audio alignment experiments
python experiments/audio_alignment/fit_projection.py

# Audio optimization
python experiments/audio_optimization/improve_audio_embeddings.py

# Ablation studies
python experiments/ablations/run_ablation.py
```

## Key Changes from Old Structure

| Old Path | New Path | Type |
|----------|----------|------|
| `scripts/run_aems_pipeline.py` | `pipeline/run_aems_pipeline.py` | Core |
| `bin/data/extract_frames.py` | `bin/data/extract_frames.py` | Core |
| `bin/training/train_gating_network.py` | `bin/training/train_gating_network.py` | Core |
| `bin/evaluation/eval_aems_retrieval.py` | `bin/evaluation/eval_aems_retrieval.py` | Core |
| `experiments/audio_alignment/*` | `experiments/audio_alignment/` | Research |
| `experiments/diagnostics/*` | `experiments/diagnostics/` | Research |
| `experiments/ablations/*` | `experiments/ablations/` | Research |
| `scripts/training/improve_audio_*.py` | `experiments/audio_optimization/` | Research |

## Naming Conventions

- **`bin/` scripts**: AEMS-specific, prefix removed (e.g., `precompute_aems_video_embeddings.py` → `precompute_video_embeddings.py`)
- **`experiments/` scripts**: Research & exploratory, kept as-is
- **`src/` modules**: Core library, no prefix needed

## Python Imports

Scripts in `bin/` and `experiments/` import from `src/`:
```python
from src.models import GatingNetwork
from src.encoders import CLAPEncoder
from src.data import AEMSDataset
```

The `src/` directory is the single source of truth for core functionality.

## Contributing

When adding new scripts:
- **Core AEMS pipeline step**: Place in `bin/<category>/`
- **Research/experiment**: Place in `experiments/<subcategory>/`
- **Core functionality**: Add to `src/`
- **Evaluation metric**: Add to `src/evaluation/`
- **New model**: Add to `src/models/`
- **New encoder**: Add to `src/encoders/`
