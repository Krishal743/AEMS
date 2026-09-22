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

- **Main Evaluation**: `bin/evaluation/eval_aems_retrieval.py`
- **Priority 1 Eval**: `bin/evaluation/evaluate_priority1.py`
- **Full Suite**: `bash bin/evaluation/run_final_eval.sh`

Results saved to `outputs/aems/` with detailed metrics.

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
