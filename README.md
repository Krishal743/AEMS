# AEMS/FineVideo Multimodal Retrieval System

A comprehensive multimodal video retrieval system using CLIP (vision/text) and WavLM (audio, projected into CLIP space by a trained adapter), fused with per-query z-scored similarities.

## 🎯 Quick Start

### Run Complete Pipeline
```bash
python pipeline/run_aems_pipeline.py
```

### Data Processing
```bash
python bin/data/build_manifest.py
python bin/data/extract_frames.py
python bin/data/extract_audio.py
```

### Feature Extraction
```bash
python bin/embeddings/precompute_video_embeddings.py
python bin/embeddings/precompute_audio_embeddings.py   # raw WavLM features
python bin/embeddings/precompute_text_embeddings.py
```

### Model Training
```bash
python bin/training/train_temporal_transformer.py
python bin/training/export_transformer_embeddings.py
python bin/training/train_audio_adapter.py        # WavLM -> CLIP audio branch
python bin/training/train_gating_network.py       # optional: --fusion gate
```

### Evaluation
```bash
python bin/evaluation/eval_aems_retrieval.py
bash bin/evaluation/run_final_eval.sh
```

### Querying
```bash
# Text query
python bin/queries/query_text.py --query "your search query"

# Audio query
python bin/queries/query_audio.py --audio audio.wav

# Multimodal
python bin/queries/query_mixed.py --text "query" --image image.jpg
```

## 📁 Directory Structure

See **[CODEBASE_STRUCTURE.md](CODEBASE_STRUCTURE.md)** for complete documentation.

```
team23/
├── src/                    Core library (models, encoders, data utilities)
├── pipeline/               Pipeline orchestration
├── bin/                    Production-ready executable scripts
│   ├── data/              Data processing
│   ├── embeddings/        Feature extraction
│   ├── training/          Model training
│   ├── evaluation/        Evaluation & testing
│   ├── queries/           Query interfaces
│   ├── verification/      Validation tools
│   └── demo/              Demo scripts
├── experiments/            Research & experimental code
│   ├── audio_alignment/   Audio alignment research
│   ├── audio_optimization/ Audio improvements
│   ├── diagnostics/       Analysis tools
│   └── ablations/         Ablation studies
├── data/                  Dataset storage
├── outputs/               Experiment results & logs
├── checkpoints/           Training checkpoints
├── models/                Final trained models
├── embeddings/            Precomputed embeddings
├── docs/                  Documentation
└── tests/                 Unit tests
```

## 🏗️ Architecture

### Core Components (src/)
- **Models**: Gating networks & temporal transformers
- **Encoders**: CLIP (vision/text) and WavLM-Large + adapter (audio)
- **Fusion**: per-query z-scored branch similarities, fixed weights (`AEMS_FUSION_WEIGHTS`) or the gating network (`--fusion gate`)
- **Data**: Dataset loaders and metadata utilities
- **Routing**: Query encoding and multimodal similarity computation
- **Evaluation**: Retrieval metrics and analysis

### Production Scripts (bin/)
All scripts are ready to run directly:
- **Data**: Build datasets, extract features
- **Training**: Train and export models
- **Evaluation**: Comprehensive evaluation suite
- **Queries**: Query interfaces for different modalities

### Research (experiments/)
- **Audio Alignment**: Various audio projection methods (BEATs, ImageBind, WavLM, etc.)
- **Audio Optimization**: Embedding improvements and similarity tuning
- **Diagnostics**: Analysis tools for debugging and validation
- **Ablations**: Controlled experiments and parameter sweeps

## 📊 Key Features

- **Multimodal Fusion**: Learned gating network dynamically weights visual, audio, and text modalities
- **Temporal Modeling**: Temporal transformer aggregates frame-level visual features
- **Audio Segments**: Multiple audio segments per video for robustness
- **Priority 1 Optimizations**: Angular similarity, temperature/scale learners
- **Production Ready**: Clear separation of code vs research, well-organized pipeline

## 🔧 Configuration

Edit `src/config.py` for:
- Embedding dimensions and model paths
- Batch sizes and learning rates
- Data paths and preprocessing parameters
- AEMS manifest location

## 📈 Results

AEMS test split — 5,097 QA queries over 1,022 videos. Full tables, ablations and
caveats in `docs/RESULTS_2026-10-02.md`.

| Stage | System | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| — | Starting point | 0.389 | 0.509 | 0.554 | 0.449 |
| 1 | Six-branch fusion, fixed weights | 0.687 | 0.774 | 0.794 | 0.728 |
| 2 | **+ fine-tuned cross-encoder (deployed)** | **0.712** | **0.786** | **0.805** | **0.748** |

Single branches: dense E5 passages 0.624, BM25 0.550, CLIP caption 0.389, CLIP
passages 0.365, CLIP visual 0.194, WavLM audio 0.053. Two text signals — one
dense, one lexical — do nearly all the work.

Three things worth knowing before quoting these numbers:

- **Fusion is not adaptive.** The gating network is implemented and available
  (`--fusion gate`) but loses to tuned fixed weights on both splits (test 0.6865
  vs 0.6873), so static weights are deployed.
- **Only text queries are evaluated.** All 5,097 test queries are QA questions;
  the image/video/audio query paths are unmeasured.
- **BM25 is flattered by the benchmark.** Query words appear in their own
  transcript 4.6x more than in a random one, so the questions reuse transcript
  wording.

Results are saved to `outputs/aems/`.

## 🔬 Research & Experimentation

```bash
# Diagnostics
python experiments/diagnostics/run_diagnostics.py

# Audio alignment research
python experiments/audio_alignment/fit_projection.py

# Audio optimization
python experiments/audio_optimization/improve_audio_embeddings.py

# Ablation studies
python experiments/ablations/run_ablation.py
```

## 📚 Documentation

- **[CODEBASE_STRUCTURE.md](CODEBASE_STRUCTURE.md)** - Complete directory guide & workflow examples
- **[docs/AEMS-plan.md](docs/AEMS-plan.md)** - System design and approach
- **[docs/AUDIO_CHOICES.md](docs/AUDIO_CHOICES.md)** - Audio encoding decisions
- **[docs/COMPREHENSIVE_SYSTEM_REPORT.md](docs/COMPREHENSIVE_SYSTEM_REPORT.md)** - Detailed analysis
- **[docs/PRIORITY_1_IMPLEMENTATION.md](docs/PRIORITY_1_IMPLEMENTATION.md)** - Priority 1 optimizations

## 🚀 Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import src; print('Ready!')"
```

## 📝 Notes

- All paths assume execution from project root
- GPU/CUDA available for faster training & inference
- Embeddings are cached in `embeddings/` directory
- Checkpoints saved to `checkpoints/aems/`
- Results saved to `outputs/<category>/`

## 👥 Team

Team 23 - AEMS/FineVideo Retrieval System

---

**Last Updated**: September 20, 2024
