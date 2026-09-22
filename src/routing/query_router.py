"""Route queries by type and compute multimodal similarities.

Visual and caption embeddings are CLIP; audio embeddings are WavLM features
projected into CLIP text space by the audio adapter. A query supplies up to two
vectors: a CLIP query scored against the visual/caption matrices, and an audio
query scored against the audio matrix (for text queries this is the same CLIP
text vector; for an audio clip it is the adapter-projected WavLM embedding).
Branches a query type cannot reach are left out and get zero weight.

Fusion z-scores each branch's similarities per query over the gallery, so the
branches are on a common scale, then takes a weighted sum. Weights are fixed
(AEMS_FUSION_WEIGHTS) by default, or predicted per query by the gating network.
"""

import torch
import torch.nn.functional as F
import clip

from src.config import (
    AEMS_VID_EMBEDDINGS_PATH,
    AEMS_AUDIO_EMBEDDINGS_PATH,
    AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE,
    AEMS_GATING_WEIGHTS_PATH,
    AEMS_AUDIO_ADAPTER_PATH,
    AEMS_FUSION_WEIGHTS,
)
from src.models.gating_network import GatingNetwork
from src.models.audio_adapter import load_audio_adapter

BRANCHES = ("visual", "text", "audio")


def add_index_args(parser):
    parser.add_argument("--video-embeds", default=AEMS_VID_EMBEDDINGS_PATH)
    parser.add_argument("--audio-embeds", default=AEMS_AUDIO_EMBEDDINGS_PATH)
    parser.add_argument("--caption-embeds",
                        default=AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE.format(split="test"))
    parser.add_argument("--fusion", choices=["fixed", "gate"], default="fixed",
                        help="fixed: AEMS_FUSION_WEIGHTS; gate: per-query gating network")
    parser.add_argument("--gate-weights", default=AEMS_GATING_WEIGHTS_PATH)
    parser.add_argument("--top-k", type=int, default=5)


def load_fusion_gate(args, device):
    """The gating network when --fusion gate, else None (fixed weights)."""
    return load_gate(args.gate_weights, device) if args.fusion == "gate" else None


def load_clip(device):
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()
    return model


def encode_text_query(clip_model, text, device):
    tokens = clip.tokenize([text], truncate=True).to(device)
    with torch.no_grad():
        emb = clip_model.encode_text(tokens)
    return F.normalize(emb.float(), dim=1)


def encode_image_query(clip_model, image_tensor, device):
    with torch.no_grad():
        emb = clip_model.encode_image(image_tensor.to(device))
    return F.normalize(emb.float(), dim=1)


def encode_audio_query(wavlm_encoder, adapter, audio_path, device):
    """Audio clip -> WavLM features -> adapter -> (1, 512) CLIP-text-space vector."""
    feats = wavlm_encoder.encode_file(audio_path).to(device).view(1, -1)
    with torch.no_grad():
        return adapter(feats).float()


def load_adapter(path=AEMS_AUDIO_ADAPTER_PATH, device="cpu"):
    return load_audio_adapter(path, device)


def encode_video_query(clip_model, frame_tensor, device):
    with torch.no_grad():
        embs = F.normalize(clip_model.encode_image(frame_tensor.to(device)).float(), dim=1)
    return F.normalize(embs.mean(dim=0, keepdim=True), dim=1)


def load_search_index(video_path, audio_path, caption_path):
    """Return (video_ids, video_matrix, caption_matrix, audio_matrix), rows L2-normalized."""
    video_db = torch.load(video_path, weights_only=False)
    audio_db = torch.load(audio_path, weights_only=False)
    caption_db = torch.load(caption_path, weights_only=False)
    video_ids = sorted(v for v in video_db if v in audio_db and v in caption_db)

    def matrix(db):
        rows = []
        for v in video_ids:
            e = torch.as_tensor(db[v]).float()
            if e.dim() == 2:  # legacy MSR-VTT: one row per caption, max-pooled
                e = e.max(dim=0).values
            rows.append(e)
        return F.normalize(torch.stack(rows), dim=1)

    return video_ids, matrix(video_db), matrix(caption_db), matrix(audio_db)


def load_gate(path, device):
    gate = GatingNetwork(input_dim=512, hidden_dim=128).to(device)
    missing, unexpected = gate.load_state_dict(
        torch.load(path, map_location=device, weights_only=False), strict=False)
    # strict=False tolerates the temperature/scale buffers older checkpoints lack
    # (forward() never reads them), but a missing learnable weight would silently
    # leave part of the gate randomly initialised.
    missing_params = sorted(set(missing) & {n for n, _ in gate.named_parameters()})
    if missing_params or unexpected:
        raise RuntimeError(f"{path} does not match GatingNetwork: "
                           f"missing={missing_params} unexpected={sorted(unexpected)}")
    gate.eval()
    return gate


def compute_modal_similarities(video_matrix, caption_matrix, audio_matrix,
                               clip_query=None, audio_query=None):
    """Return (sim_v, sim_t, sim_a); a branch is None when the query can't reach it."""
    def score(q, m):
        return None if q is None else (q.to(m.device, m.dtype) @ m.T).squeeze(0)
    return (score(clip_query, video_matrix), score(clip_query, caption_matrix),
            score(audio_query, audio_matrix))


def zscore(sim):
    """Standardize similarities over the gallery (last dim) per query."""
    return (sim - sim.mean(-1, keepdim=True)) / (sim.std(-1, keepdim=True) + 1e-6)


def fixed_weights(weights=None):
    weights = AEMS_FUSION_WEIGHTS if weights is None else weights
    return torch.tensor([float(weights[b]) for b in BRANCHES])


def search(index, clip_query=None, audio_query=None, gate=None, weights=None):
    """Score a query against the index.

    Returns (weights, sim_v, sim_t, sim_a) on CPU, where the similarities are
    z-scored per branch and the fused score is sum(weights[i] * sim_i).
    Unavailable branches get all-zero similarities and zero weight, and the
    remaining weights are renormalized to sum to 1. With a gate and a CLIP
    query, weights come from the gate; otherwise from `weights` (a dict keyed
    by branch, default AEMS_FUSION_WEIGHTS).
    """
    video_ids, video_m, caption_m, audio_m = index
    sims = compute_modal_similarities(video_m, caption_m, audio_m, clip_query, audio_query)
    mask = torch.tensor([s is not None for s in sims], dtype=torch.float32)
    if not mask.any():
        raise ValueError("query produced no embedding for any branch")

    if gate is not None and clip_query is not None:
        device = next(gate.parameters()).device
        with torch.no_grad():
            w = gate(clip_query.to(device).float()).cpu().squeeze(0)
    else:
        w = fixed_weights(weights)
    w = w * mask
    if w.sum() <= 0:  # e.g. an audio-only query while the audio weight is 0
        w = mask
    w = w / w.sum()

    zeros = torch.zeros(len(video_ids))
    sim_v, sim_t, sim_a = (zeros if s is None else zscore(s.float().cpu()) for s in sims)
    return w, sim_v, sim_t, sim_a
