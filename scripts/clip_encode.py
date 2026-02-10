import torch
import clip
from tqdm import tqdm

@torch.no_grad()
def encode_videos(dataloader, model, device):
    video_embeddings = {}

    for images, _, video_ids in tqdm(dataloader, desc="Encoding videos"):
        images = images.to(device)             # [B, T, 3, H, W]
        B, T, C, H, W = images.shape

        images = images.view(B*T, C, H, W)
        frame_embeds = model.encode_image(images)
        frame_embeds = frame_embeds.view(B, T, -1)

        video_embed = frame_embeds.mean(dim=1)
        video_embed = video_embed / video_embed.norm(dim=-1, keepdim=True)

        for vid, emb in zip(video_ids, video_embed):
            video_embeddings[vid] = emb.cpu()

    return video_embeddings


@torch.no_grad()
def encode_texts(texts, model, device):
    tokens = clip.tokenize(texts).to(device)
    text_embeds = model.encode_text(tokens)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    return text_embeds.cpu()
