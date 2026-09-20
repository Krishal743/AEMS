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

# ===== CLIP text token budget =====
MAX_CLIP_TEXT_TOKENS = 77
CHUNK_TOKEN_BUDGET = MAX_CLIP_TEXT_TOKENS - 5

# ===== AEMS embedding paths =====
AEMS_VID_EMBEDDINGS_PATH = "embeddings/aems_video_embeddings_v1.pt"
AEMS_AUDIO_EMBEDDINGS_PATH = "embeddings/aems_audio_embeddings_v1.pt"
AEMS_TEXT_EMBEDDINGS_DESC_PATH_TEMPLATE = "embeddings/aems_text_embeddings_description_{split}.pt"
AEMS_TEXT_EMBEDDINGS_TRANS_PATH_TEMPLATE = "embeddings/aems_text_embeddings_transcript_{split}.pt"
AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE = "embeddings/aems_text_embeddings_fused_{split}.pt"

# ===== AEMS model paths =====
AEMS_TRANSFORMER_BEST_PATH = "models/aems_temporal_transformer_best_v1.pth"
AEMS_TRANSFORMER_CHECKPOINT_DIR = "checkpoints/aems"
AEMS_GATING_WEIGHTS_PATH = "models/aems_gating_weights_v1.pth"
AEMS_VIDEO_EMBEDDINGS_TRANSFORMER_PATH = "embeddings/aems_video_embeddings_transformer_v1.pt"


def set_seeds(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
