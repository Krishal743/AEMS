"""Route queries by type and compute multimodal similarities.

The three branches live in two different embedding spaces: visual and caption
embeddings are CLIP, audio embeddings are CLAP. A query must therefore be
scored with a CLIP-space vector against the visual/caption matrices and a
CLAP-space vector against the audio matrix. Query types that cannot produce a
vector in one of the spaces (e.g. an image has no CLAP encoding) simply leave
that branch out, and the gate's weight for it is masked to zero.
"""

import numpy as np
import torch
import torch.nn.functional as F
import clip

from src.config import (
    AEMS_VID_EMBEDDINGS_PATH,
    AEMS_AUDIO_EMBEDDINGS_PATH,
    AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE,
    AEMS_GATING_WEIGHTS_PATH,
)
from src.models.gating_network import GatingNetwork


def add_index_args(parser):
    parser.add_argument("--video-embeds", default=AEMS_VID_EMBEDDINGS_PATH)
    parser.add_argument("--audio-embeds", default=AEMS_AUDIO_EMBEDDINGS_PATH)
    parser.add_argument("--caption-embeds",
                        default=AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE.format(split="test"))
    parser.add_argument("--gate-weights", default=AEMS_GATING_WEIGHTS_PATH)
    parser.add_argument("--top-k", type=int, default=5)


def load_clip(device):
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()
    return model


def encode_text_query(clip_model, text, device):
    tokens = clip.tokenize([text], truncate=True).to(device)
    with torch.no_grad():
        emb = clip_model.encode_text(tokens)
    return F.normalize(emb.float(), dim=1)


def encode_clap_text_query(clap_encoder, text, device):
    emb = clap_encoder.encode_text([text])
    emb = torch.as_tensor(np.asarray(emb) if not torch.is_tensor(emb) else emb).float()
    return F.normalize(emb.reshape(1, -1), dim=1).to(device)


def encode_image_query(clip_model, image_tensor, device):
    with torch.no_grad():
        emb = clip_model.encode_image(image_tensor.to(device))
    return F.normalize(emb.float(), dim=1)


def encode_audio_query(clap_encoder, audio_path, device):
    import librosa
    audio, _ = librosa.load(audio_path, sr=48000, mono=True)
    with torch.no_grad():
        emb = clap_encoder.model.get_audio_embedding_from_data(x=audio.astype("float32").reshape(1, -1))
    emb = torch.as_tensor(np.asarray(emb) if not torch.is_tensor(emb) else emb).float()
    return F.normalize(emb.reshape(1, -1), dim=1).to(device)


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
                               clip_query=None, clap_query=None):
    """Return (sim_v, sim_t, sim_a); a branch is None when no query exists in its space."""
    def score(q, m):
        return None if q is None else (q.to(m.device, m.dtype) @ m.T).squeeze(0)
    return (score(clip_query, video_matrix), score(clip_query, caption_matrix),
            score(clap_query, audio_matrix))


def search(index, gate, clip_query=None, clap_query=None):
    """Gate and score a query against the index.

    Returns (weights, sim_v, sim_t, sim_a) on CPU. Unavailable branches get
    all-zero similarities and zero weight, with the remaining weights
    renormalized. The gate needs a CLIP-space query; without one, the available
    branches share weight equally.
    """
    video_ids, video_m, caption_m, audio_m = index
    sims = compute_modal_similarities(video_m, caption_m, audio_m, clip_query, clap_query)
    mask = torch.tensor([s is not None for s in sims], dtype=torch.float32)
    if not mask.any():
        raise ValueError("query produced no embedding in either CLIP or CLAP space")

    if clip_query is None:
        weights = mask
    else:
        device = next(gate.parameters()).device
        with torch.no_grad():
            weights = gate(clip_query.to(device).float()).cpu().squeeze(0) * mask
    weights = weights / weights.sum()

    zeros = torch.zeros(len(video_ids))
    sim_v, sim_t, sim_a = (zeros if s is None else s.float().cpu() for s in sims)
    return weights, sim_v, sim_t, sim_a
