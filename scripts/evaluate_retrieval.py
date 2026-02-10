import torch

def evaluate_retrieval(
    text_embeds,
    video_embeds,
    text_video_ids,
    video_ids,
    ks=(1, 5, 10)
):
    """
    text_embeds: Tensor [N_text, D]
    video_embeds: Tensor [N_video, D]
    text_video_ids: list[str], length N_text
    video_ids: list[str], length N_video (ORDERED to match video_embeds)
    """

    # Cosine similarity (embeddings are already normalized)
    sim = text_embeds @ video_embeds.T  # [N_text, N_video]

    results = {f"R@{k}": 0 for k in ks}

    for i in range(sim.size(0)):
        gt_video = text_video_ids[i]

        # Rank videos for this caption
        ranked_indices = torch.argsort(sim[i], descending=True)

        for k in ks:
            topk_indices = ranked_indices[:k]
            topk_video_ids = [video_ids[j] for j in topk_indices]

            if gt_video in topk_video_ids:
                results[f"R@{k}"] += 1

    # Normalize
    for k in ks:
        results[f"R@{k}"] /= sim.size(0)

    return results
