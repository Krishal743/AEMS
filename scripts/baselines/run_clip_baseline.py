import json
import torch
import clip
from tqdm import tqdm
from src.evaluation.evaluate_retrieval import evaluate_retrieval

# ================= CONFIG =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
METADATA = "data/processed/metadata/msrvtt_metadata.json"
VIDEO_EMBEDS = "embeddings/video_embeddings.pt"
TEXT_BATCH_SIZE = 256   # safe for 6GB GPU
# ==========================================

print("Using device:", DEVICE)

# Load CLIP
model, _ = clip.load("ViT-B/32", device=DEVICE)
model.eval()

# Load metadata
with open(METADATA) as f:
    data = json.load(f)

# Use TEST split only
test_data = [d for d in data if d["split"] == "test"]

texts = [d["text"] for d in test_data]
text_video_ids = [d["video_id"] for d in test_data]

print(f"Test captions: {len(texts)}")

# ================= LOAD VIDEO EMBEDDINGS =================
video_embeddings_dict = torch.load(VIDEO_EMBEDS)

# IMPORTANT: fixed, ordered video list
video_ids_unique = sorted(video_embeddings_dict.keys())

video_matrix = torch.stack(
    [video_embeddings_dict[v] for v in video_ids_unique]
)

print(f"Unique videos: {len(video_ids_unique)}")
# ========================================================

# ================= ENCODE TEXT (BATCHED) =================
all_text_embeds = []

with torch.no_grad():
    for i in tqdm(
        range(0, len(texts), TEXT_BATCH_SIZE),
        desc="Encoding text"
    ):
        batch_texts = texts[i:i + TEXT_BATCH_SIZE]
        tokens = clip.tokenize(batch_texts).to(DEVICE)

        embeds = model.encode_text(tokens)
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)

        all_text_embeds.append(embeds.cpu())

text_embeds = torch.cat(all_text_embeds, dim=0)
# ========================================================

# ================= COMPUTE SIMILARITY =================
similarity_matrix = text_embeds @ video_matrix.T
# ========================================================

# ================= EVALUATE =================
metrics = evaluate_retrieval(
    similarity_matrix=similarity_matrix,
    text_video_ids=text_video_ids,
    video_ids=video_ids_unique
)

print("==== MSR-VTT CLIP Baseline ====")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
# ==========================================
