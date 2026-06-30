import torch
from src.data.datasets import MSRVTTDataset
from src.evaluation.evaluate_retrieval import evaluate_retrieval

from src.encoders.clip_encode import load_clip_model, encode_texts
from src.encoders.clap_encode import CLAPEncoder


def normalize(x):
    return x / x.norm(dim=1, keepdim=True)


def main():
    print("[DATASET] Loading dataset...")
    dataset = MSRVTTDataset(
        metadata_path="data/processed/metadata/msrvtt_metadata.json",
        split="test",
        device="cpu"
    )

    print("[EMBED] Loading embeddings...")
    clip_db = torch.load("embeddings/video_embeddings.pt", weights_only=False)
    audio_db = torch.load("embeddings/audio_embeddings.pt", weights_only=False)

    # -------------------------
    # Video embeddings
    # -------------------------
    print("[PREP] Preparing embeddings...")
    video_ids = list(set(clip_db.keys()) & set(audio_db.keys()))

    clip_video = torch.stack([clip_db[v] for v in video_ids])
    clap_audio = torch.stack([torch.tensor(audio_db[v]) for v in video_ids])

    clip_video = normalize(clip_video)
    clap_audio = normalize(clap_audio)

    # -------------------------
    # Collect text (FAST)
    # -------------------------
    print("[TEXT] Collecting text...")
    texts = []
    text_video_ids = []

    for item in dataset.data:
        vid = item["video_id"]

        if vid not in video_ids:
            continue

        texts.append(item["text"])
        text_video_ids.append(vid)

    # -------------------------
    # Load models
    # -------------------------
    print("[MODEL] Loading models...")
    clip_model, _ = load_clip_model("cuda")
    clap_encoder = CLAPEncoder(device="cuda")

    # -------------------------
    # CLIP text
    # -------------------------
    print("[CLIP] Encoding CLIP text (batched)...")

    BATCH_SIZE = 32 

    clip_text_list = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]

        emb = encode_texts(batch, clip_model, "cuda")  
        clip_text_list.append(emb)

    clip_text = torch.cat(clip_text_list, dim=0)
    clip_text = normalize(clip_text)

    # -------------------------
    # CLAP text (batched)
    # -------------------------
    print("[CLAP] Encoding CLAP text...")
    BATCH_SIZE = 32

    clap_text_list = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        emb = clap_encoder.encode_text(batch)
        clap_text_list.append(torch.tensor(emb))

    clap_text = torch.cat(clap_text_list, dim=0)
    clap_text = normalize(clap_text)

    # -------------------------
    # Similarities
    # -------------------------
    print("[SIM] Computing similarities...")
    sim_clip = clip_text @ clip_video.T
    sim_clap = clap_text @ clap_audio.T

    # -------------------------
    # Fusion
    # -------------------------
    alpha = 0.99
    sim_fused = alpha * sim_clip + (1 - alpha) * sim_clap

    # -------------------------
    # Evaluate
    # -------------------------
    print("[EVAL] Evaluating...")
    results = evaluate_retrieval(
        sim_fused,
        text_video_ids,
        video_ids
    )

    print("\n===== FUSION RESULTS =====")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()