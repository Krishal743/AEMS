"""Route queries by type and compute multimodal similarities."""

import torch
import clip
from src.encoders.clap_encode import CLAPEncoder


def load_clip(device):
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()
    return model


def encode_text_query(clip_model, text, device):
    tokens = clip.tokenize([text]).to(device)
    emb = clip_model.encode_text(tokens)
    emb = emb / emb.norm(dim=1, keepdim=True)
    return emb.float()


def encode_image_query(clip_model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    emb = clip_model.encode_image(image_tensor)
    emb = emb / emb.norm(dim=1, keepdim=True)
    return emb.float()


def encode_audio_query(clap_encoder, audio_path, device):
    import librosa
    audio, sr = librosa.load(audio_path, sr=48000, mono=True)
    audio = audio.astype("float32").reshape(1, -1)
    emb = clap_encoder.model.get_audio_embedding_from_data(x=audio)
    emb_t = torch.from_numpy(emb).float().to(device)
    if emb_t.dim() == 1:
        emb_t = emb_t.unsqueeze(0)
    emb_t = emb_t / emb_t.norm(dim=1, keepdim=True)
    return emb_t


def encode_video_query(clip_model, frame_tensor, device):
    frame_tensor = frame_tensor.to(device)
    with torch.no_grad():
        embs = clip_model.encode_image(frame_tensor)
        embs = embs / embs.norm(dim=1, keepdim=True)
        emb = embs.mean(dim=0, keepdim=True)
    return emb.float()


def compute_modal_similarities(query_embed, video_matrix, audio_matrix, caption_matrix):
    device = query_embed.device
    dtype = query_embed.dtype
    sim_v = (query_embed @ video_matrix.to(device=device, dtype=dtype).T).squeeze(0)
    sim_t = (query_embed @ caption_matrix.to(device=device, dtype=dtype).T).squeeze(0)
    sim_a = (query_embed @ audio_matrix.to(device=device, dtype=dtype).T).squeeze(0)
    return sim_v, sim_t, sim_a
