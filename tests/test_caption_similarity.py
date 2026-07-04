import torch
import torch.nn.functional as F

def test_caption_similarity():
    # 3 Queries, 2 Videos, 2 Captions per video
    # Query embeddings (3, 512)
    # Caption embeddings (2, 2, 512)
    
    Q = torch.randn(3, 512)
    Q = F.normalize(Q, p=2, dim=1)
    
    # Videos (V, 2, 512)
    V1_caps = torch.randn(2, 512)
    V1_caps = F.normalize(V1_caps, p=2, dim=1)
    V2_caps = torch.randn(2, 512)
    V2_caps = F.normalize(V2_caps, p=2, dim=1)
    
    caps = torch.stack([V1_caps, V2_caps]) # (2, 2, 512)
    
    # --- Current (WRONG) Implementation: Coordinate-wise Max ---
    # pooled = max(dim=1) over the 2 captions for each video
    pooled = caps.max(dim=1)[0] # (2, 512)
    sim_wrong = Q @ pooled.T # (3, 2)
    
    # --- Correct Implementation: Max Similarity over Captions ---
    # Compute sim for all captions: (Q, 2, 2)
    # (Q, 512) @ (V, 2, 512).T → (Q, V, 2)
    # Use torch.einsum for batch matrix multiplication
    sim_all = torch.einsum('qf,vcf->qvc', Q, caps) # (3, 2, 2)
    sim_correct = sim_all.max(dim=2).values # (3, 2)
    
    # Assert they are different
    assert not torch.allclose(sim_wrong, sim_correct), "Coordinate-wise max and max-over-captions should be different"
    print("Test Passed: Corrected caption similarity identified!")

if __name__ == "__main__":
    test_caption_similarity()
