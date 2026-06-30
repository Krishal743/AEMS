import torch
from src.encoders.clap_encode import CLAPEncoder
from src.data.datasets import MSRVTTDataset
from src.evaluation.evaluate_retrieval import evaluate_retrieval

def main():
    print("[INIT] Initializing CLAP encoder...")
    encoder = CLAPEncoder(device="cuda")

    print("[DATA] Loading dataset...")
    dataset = MSRVTTDataset(
        metadata_path="data/processed/metadata/msrvtt_metadata.json",
        split="test",
        device="cpu"
    )

    print("[DATA] Loading audio embeddings...")
    audio_db = torch.load("embeddings/audio_embeddings.pt", weights_only=False)

    # -------------------------
    # Filter to test videos that have audio embeddings
    # -------------------------
    test_video_ids = set(item["video_id"] for item in dataset.data)
    video_ids = [vid for vid in audio_db.keys() if vid in test_video_ids]

    print(f"[FILTER] Using {len(video_ids)}/{len(audio_db)} videos with test split overlap")

    # Build video embeddings for filtered videos
    video_embeds = torch.stack([
        torch.tensor(audio_db[vid]) for vid in video_ids
    ])  # [N_video, D]

    # -------------------------
    # Filter test entries to those with audio embeddings
    # -------------------------
    print("[TEXT] Filtering test data to entries with audio...")

    texts = []
    text_video_ids = []

    for item in dataset.data:
        if item["video_id"] in video_ids:
            texts.append(item["text"])
            text_video_ids.append(item["video_id"])

    print(f"[FILTER] Using {len(texts)}/{len(dataset.data)} test entries with audio")

    # -------------------------
    # Batch encode text
    # -------------------------
    print("[CLAP] Encoding text in batches...")

    BATCH_SIZE = 32

    text_embeds_list = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i + BATCH_SIZE]

        emb = encoder.encode_text(batch_texts)
        emb = torch.tensor(emb)

        text_embeds_list.append(emb)

    text_embeds = torch.cat(text_embeds_list, dim=0)
    

    # -------------------------
    # Normalize embeddings
    # -------------------------
    print("[NORM] Normalizing embeddings...")
    text_embeds = text_embeds / text_embeds.norm(dim=1, keepdim=True)
    video_embeds = video_embeds / video_embeds.norm(dim=1, keepdim=True)

    # -------------------------
    # Compute similarity matrix
    # -------------------------
    print("[SIM] Computing similarity matrix...")
    similarity_matrix = text_embeds @ video_embeds.T

    # -------------------------
    # Evaluate retrieval
    # -------------------------
    print("[EVAL] Computing retrieval metrics...")
    results = evaluate_retrieval(
        similarity_matrix,
        text_video_ids,
        video_ids
    )

    # -------------------------
    # Print results
    # -------------------------
    print("\n===== CLAP Retrieval Results =====")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()