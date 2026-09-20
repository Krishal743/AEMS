import os
import torch
from src.encoders.clap_encode import CLAPEncoder

AUDIO_DIR = "data/processed/audio"
SAVE_PATH = "embeddings/audio_embeddings.pt"

os.makedirs("embeddings", exist_ok=True)

encoder = CLAPEncoder(device="cuda")

embeddings = {}

for file in os.listdir(AUDIO_DIR):
    if not file.endswith(".wav"):
        continue

    vid = file.replace(".wav", "")
    path = os.path.join(AUDIO_DIR, file)

    print(f"Encoding {vid}")
    emb = encoder.encode_audio(path)

    embeddings[vid] = emb

torch.save(embeddings, SAVE_PATH)
print("Saved audio embeddings!")