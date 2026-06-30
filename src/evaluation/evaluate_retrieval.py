import torch

def evaluate_retrieval(similarity_matrix, text_video_ids, video_ids, ks=[1, 5, 10]):
    results = {}
    for k in ks:
        results[f"R@{k}"] = 0.0

    for i in range(similarity_matrix.size(0)):
        gt_video = text_video_ids[i]

        ranked_indices = torch.argsort(similarity_matrix[i], descending=True)

        for k_val in ks:
            k_int = int(k_val)
            topk_list = ranked_indices[:k_int].tolist()
            topk_vids = [video_ids[j] for j in topk_list]

            if gt_video in topk_vids:
                results[f"R@{k_val}"] += 1.0

    num_queries = similarity_matrix.size(0)
    for k_val in ks:
        results[f"R@{k_val}"] /= num_queries

    return results