import torch
import torch.nn.functional as F

# This is the original logic that is broken
def compute_ranking_loss_original(sim_gated, batch_len):
    ranking_losses = []
    NUM_NEGATIVES = 2
    RANKING_MARGIN = 0.2
    for i in range(batch_len):
        correct_score = sim_gated[i, i].unsqueeze(0) # BUG
        neg_scores = sim_gated[i].clone()
        neg_scores[i] = -float("inf")
        topk_scores, _ = torch.topk(neg_scores, min(NUM_NEGATIVES, batch_len - 1))
        for neg_score in topk_scores:
            ranking_losses.append(
                torch.clamp(RANKING_MARGIN - (correct_score - neg_score.unsqueeze(0)), min=0)
            )
    return torch.stack(ranking_losses).mean() if ranking_losses else torch.tensor(0.0)

# This is the NEW logic
def compute_ranking_loss_fixed(sim_gated, pos_indices):
    ranking_losses = []
    NUM_NEGATIVES = 2
    RANKING_MARGIN = 0.2
    for i in range(len(pos_indices)):
        pos_idx = pos_indices[i]
        correct_score = sim_gated[i, pos_idx].unsqueeze(0)
        neg_scores = sim_gated[i].clone()
        neg_scores[pos_idx] = -float("inf")
        topk_scores, _ = torch.topk(neg_scores, min(NUM_NEGATIVES, len(pos_indices) - 1))
        for neg_score in topk_scores:
            ranking_losses.append(
                torch.clamp(RANKING_MARGIN - (correct_score - neg_score.unsqueeze(0)), min=0)
            )
    return torch.stack(ranking_losses).mean() if ranking_losses else torch.tensor(0.0)

def test_ranking_alignment():
    batch_len = 3
    # sim_gated matrix:
    # Query 0: pos index 2 (score 0.9), others 0.1
    # Query 1: pos index 0 (score 0.9), others 0.1
    # Query 2: pos index 1 (score 0.9), others 0.1
    sim_gated = torch.tensor([
        [0.1, 0.1, 0.9],
        [0.9, 0.1, 0.1],
        [0.1, 0.9, 0.1]
    ], dtype=torch.float32)
    pos_indices = torch.tensor([2, 0, 1])

    # Original logic (BUGGY):
    # For query 0, it picks sim_gated[0, 0] = 0.1 as positive (SHOULD BE 0.9)
    loss_orig = compute_ranking_loss_original(sim_gated, batch_len)
    
    # New logic (FIXED):
    # For query 0, it picks sim_gated[0, 2] = 0.9 as positive (CORRECT)
    loss_fixed = compute_ranking_loss_fixed(sim_gated, pos_indices)
    
    # Original loss should be higher because it thinks the positive is a negative
    assert loss_orig > loss_fixed, f"Original loss ({loss_orig}) should be higher than fixed loss ({loss_fixed})"
    print(f"Test Passed: Loss Orig: {loss_orig:.4f}, Loss Fixed: {loss_fixed:.4f}")

if __name__ == "__main__":
    test_ranking_alignment()
