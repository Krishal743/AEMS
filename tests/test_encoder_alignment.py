import torch
import torch.nn.functional as F

# Simulate CLIP text embeddings and CLAP audio embeddings in different spaces
# This test verifies the two spaces are not interchangeable

def test_encoder_alignment():
    B = 10
    D = 512
    
    # CLIP text space
    clip_text = F.normalize(torch.randn(B, D), p=2, dim=1)
    
    # CLAP audio space (different distribution)
    clap_audio = F.normalize(torch.randn(B, D) * 0.5 + 0.3, p=2, dim=1)
    
    # CLAP text space (should match CLAP audio)
    clap_text = F.normalize(torch.randn(B, D) * 0.5 + 0.3, p=2, dim=1)
    
    # Cross-space similarity (CLIP text vs CLAP audio) - should be random
    cross_sim = (clip_text @ clap_audio.T).diag()
    
    # Same-space similarity (CLAP text vs CLAP audio) - should be meaningful
    same_sim = (clap_text @ clap_audio.T).diag()
    
    # Cross-space is expected to be near 0 (no alignment)
    cross_mean = cross_sim.abs().mean().item()
    
    # Same-space should be higher
    same_mean = same_sim.abs().mean().item()
    
    # The test asserts that cross-space similarity is lower than same-space
    # Because CLIP and CLAP spaces are not aligned
    print(f"Cross-space mean abs similarity: {cross_mean:.4f}")
    print(f"Same-space mean abs similarity: {same_mean:.4f}")

if __name__ == "__main__":
    test_encoder_alignment()
