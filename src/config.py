import torch
import random
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dimensionality
TEXT_DIM = 512
HIDDEN_DIM = 128
NUM_FRAMES = 16

# Data paths
CHECKPOINT_DIR = "checkpoints"

# Transformer model paths
TRANSFORMER_BEST_PATH = "models/temporal_transformer_best.pth"

# Memory management
QUERY_BATCH_SIZE = 32
VIDEO_BATCH_SIZE = 100
TOP_K = 10
MAX_QUERIES = 100
MAX_VIDEOS = 100

# Gating training
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
NUM_TRAIN_QUERIES = 300
RANKING_MARGIN = 0.2
NUM_NEGATIVES = 10

# ===== AEMS paths (per Section 2) =====
AEMS_MANIFEST_PATH = "data/processed/aems/metadata/aems_manifest_v1.json"
AEMS_FRAMES_DIR = "data/processed/aems/frames_uniform"
AEMS_AUDIO_DIR = "data/processed/aems/audio"
AEMS_DATASET_ROOT = "aems/dataset"

# ===== AEMS audio extraction =====
AEMS_AUDIO_SR = 48000
AEMS_AUDIO_CLIP_SEC = 10
AEMS_WAVLM_SR = 16000

# ===== CLIP text token budget =====
MAX_CLIP_TEXT_TOKENS = 77
CHUNK_TOKEN_BUDGET = MAX_CLIP_TEXT_TOKENS - 5

# ===== AEMS embedding paths =====
AEMS_VID_EMBEDDINGS_PATH = "embeddings/aems_video_embeddings_v1.pt"
# Audio branch: raw WavLM-Large features (1024-d) are projected into CLIP text
# space by a trained adapter, so the branch is scored with the CLIP query.
AEMS_WAVLM_FEATURES_PATH = "embeddings/aems_audio_embeddings_wavlm_v1.pt"
AEMS_AUDIO_EMBEDDINGS_PATH = "embeddings/aems_audio_embeddings_wavlm_clip_v1.pt"
# Out-of-fold projections of the train videos, for training downstream models
# without the deployed adapter's in-sample optimism (see docs/PROTOCOL.md).
AEMS_AUDIO_OOF_PATH = "embeddings/aems_audio_oof_train_v1.pt"
# Previous CLAP audio branch (CLAP space), still used by experiments/.
AEMS_CLAP_AUDIO_EMBEDDINGS_PATH = "embeddings/aems_audio_embeddings_v1.pt"
AEMS_TEXT_EMBEDDINGS_DESC_PATH_TEMPLATE = "embeddings/aems_text_embeddings_description_{split}.pt"
AEMS_TEXT_EMBEDDINGS_TRANS_PATH_TEMPLATE = "embeddings/aems_text_embeddings_transcript_{split}.pt"
AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE = "embeddings/aems_text_embeddings_fused_{split}.pt"
# Per-chunk text embeddings for the late-interaction (max-sim) branch.
AEMS_TEXT_CHUNKS_PATH_TEMPLATE = "embeddings/aems_text_chunks_{split}.pt"
# Per-frame CLIP embeddings: late interaction over frames, and the only way to
# hold a frame out of the index when evaluating image queries.
AEMS_FRAME_EMBEDDINGS_PATH = "embeddings/aems_frame_embeddings_v1.pt"
# Dense passages from a retrieval-trained text encoder (E5). CLIP's text tower
# was trained on image captions, not documents: on validation it scores 0.397
# R@1 against E5's 0.639 for the same passages.
AEMS_DENSE_TEXT_MODEL = "e5"
# Fine-tuned on AEMS (bin/training/train_dense_retriever.py): val dense-branch
# R@1 0.6392 -> 0.6925. Used when present; the "e5" conventions still apply.
AEMS_DENSE_MODEL_PATH = "models/aems_dense_retriever_v1"
AEMS_DENSE_CHUNK_TOKENS = 400
AEMS_DENSE_PASSAGES_PATH_TEMPLATE = "embeddings/aems_dense_passages_{split}.pt"

# ===== AEMS model paths =====
AEMS_TRANSFORMER_BEST_PATH = "models/aems_temporal_transformer_best_v1.pth"
AEMS_TRANSFORMER_CHECKPOINT_DIR = "checkpoints/aems"
AEMS_GATING_WEIGHTS_PATH = "models/aems_gating_weights_v1.pth"
AEMS_AUDIO_ADAPTER_PATH = "models/aems_audio_adapter_wavlm_v1.pth"
AEMS_PER_CANDIDATE_GATE_PATH = "models/aems_per_candidate_gate_v1.pth"

# ===== Fusion =====
# Each branch's similarities are z-scored per query over the gallery, then
# combined with these weights. Tuned by grid search plus coordinate ascent on the
# validation split (never on test). The learned gate was measured against these
# and lost (0.7075 vs 0.7100), so fixed weights are the deployed default.
# Retuned after the visual branch switched to best-frame scoring. Note the chunk
# branch is back at 0.25: it measured redundant (-0.0009) against the *mean-pooled*
# visual branch, but earns its place again once visual improved, so the earlier
# "drop chunk" result was configuration-specific rather than general.
AEMS_FUSION_WEIGHTS = {"visual": 0.3, "text": 0.2, "chunk": 0.1, "audio": 0.1,
                       "bm25": 0.5, "dense": 1.1}

# Which branches are *reliable* depends on what kind of query is being served,
# and the weights above were tuned on text queries alone. Applying them to an
# audio query cost 0.70 R@1: renormalizing over the reachable branches handed
# the audio branch 14% of the mass and three near-chance branches the other 86%
# (docs/RESULTS_MULTIMODAL_2026-10-03.md). These overrides are tuned per
# modality by bin/training/tune_modality_weights.py on validation videos from
# the train split.
#
# "text" is deliberately absent: it falls back to AEMS_FUSION_WEIGHTS above,
# which were tuned on the deployed 16-frame gallery over all 5,097 questions.
# The benchmark's own text row was tuned on a reduced 8-frame gallery with one
# question per video and is not transferable to the deployed path.
AEMS_MODALITY_WEIGHTS = {
    "image":      {"visual": 1.0, "text": 0.2, "chunk": 0.1, "audio": 0.1, "bm25": 0.5, "dense": 1.1},
    "video":      {"visual": 0.7, "text": 0.2, "chunk": 0.05, "audio": 0.1, "bm25": 0.5, "dense": 1.1},
    "image+text": {"visual": 1.5, "text": 0.2, "chunk": 0.1, "audio": 0.1, "bm25": 0.5, "dense": 1.1},
    # An audio query is a CLIP-space vector, so it *can* be scored against the
    # visual, caption and passage branches. Tuning drove all three to zero: they
    # are near-chance for audio (0.03-0.07 R@1) and act as noise. Reading an
    # audio query as a CLIP query adds nothing over the audio branch alone.
    "audio_clip": {"visual": 0.0, "text": 0.0, "chunk": 0.0, "audio": 0.1, "bm25": 0.5, "dense": 1.1},
}
AEMS_VIDEO_EMBEDDINGS_TRANSFORMER_PATH = "embeddings/aems_video_embeddings_transformer_v1.pt"


def set_seeds(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
