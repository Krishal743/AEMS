import torch
import clip
from tqdm import tqdm


def load_clip_model(device="cuda"):
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess


@torch.no_grad()
def encode_texts(texts, model, device):
    tokens = clip.tokenize(texts).to(device)

    text_embeds = model.encode_text(tokens)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    return text_embeds.cpu()