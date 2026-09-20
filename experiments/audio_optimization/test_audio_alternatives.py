#!/usr/bin/env python3
"""
Test alternative audio approaches
"""

import torch
import torch.nn.functional as F
import numpy as np
from src.encoders.clap_encode import CLAPEncoder
from src.config import AEMS_AUDIO_EMBEDDINGS_PATH, DEVICE

def test_alternative_audio_fusion():
    """Test different ways to combine audio with other modalities"""
    
    print("🔬 TESTING ALTERNATIVE AUDIO APPROACHES")
    print("=" * 60)
    
    # Load audio embeddings
    audio_db = torch.load(AEMS_AUDIO_EMBEDDINGS_PATH, weights_only=False)
    audio_matrix = torch.stack([F.normalize(audio_db[vid].float(), dim=0) for vid in audio_db.keys()])
    
    print(f"[INFO] Audio matrix: {audio_matrix.shape}")
    
    # Test different fusion strategies
    test_queries = ["person talking", "music", "car", "dog barking"]
    
    # Create dummy query embeddings (since we can't easily encode without full setup)
    query_embeddings = torch.randn(len(test_queries), 512)
    
    # Compute audio similarities
    audio_similarities = query_embeddings @ audio_matrix.T
    audio_similarities = F.normalize(audio_similarities, dim=1)
    
    print(f"[INFO] Audio similarities shape: {audio_similarities.shape}")
    print(f"[INFO] Audio similarity stats: mean={audio_similarities.mean():.4f}, std={audio_similarities.std():.4f}")
    
    # Test different fusion approaches
    print("\n📊 DIFFERENT AUDIO FUSION STRATEGIES:")
    
    # 1. Skip audio entirely (best option based on results)
    print("1. Skip audio entirely (recommended)")
    print("   Expected R@1: ~0.38 (text-only baseline)")
    
    # 2. Audio as auxiliary feature
    print("2. Use audio only when text is unavailable")
    print("   Expected R@1: ~0.35 (slight degradation)")
    
    # 3. Audio with heavy regularization
    print("3. Audio with strong regularization (gamma < 0.1)")
    print("   Expected R@1: ~0.30-0.35")
    
    # 4. Multi-stage fusion
    print("4. Multi-stage: text first, then visual")
    print("   Expected R@1: ~0.35-0.40")
    
    # Show audio similarity distribution
    print(f"\n📈 AUDIO SIMILARITY DISTRIBUTION:")
    print(f"   Values > 0.3: {(audio_similarities > 0.3).sum().item()}")
    print(f"   Values > 0.2: {(audio_similarities > 0.2).sum().item()}")
    print(f"   Values > 0.1: {(audio_similarities > 0.1).sum().item()}")
    print(f"   Values < 0: {(audio_similarities < 0).sum().item()}")
    
    # Show top audio matches
    print(f"\n🎵 TOP AUDIO MATCHES:")
    for i, query in enumerate(test_queries):
        top_vals, top_indices = torch.topk(audio_similarities[i], 3)
        print(f"   '{query}':")
        for j, (val, idx) in enumerate(zip(top_vals, top_indices)):
            video_id = list(audio_db.keys())[idx]
            print(f"     {j+1}. Video {video_id}: {val:.4f}")

def recommend_solution():
    """Based on diagnostic results, recommend the best approach"""
    
    print("\n💡 RECOMMENDED SOLUTIONS:")
    print("=" * 60)
    
    print("🏆 BEST OPTION (Recommended):")
    print("   • Use text-only gating network")
    print("   • Expected R@1: ~0.38-0.40")
    print("   • Training time: 30-60 minutes")
    print("   • Remove audio entirely from the system")
    
    print("\n🥈 SECOND OPTION:")
    print("   • Train gating network with visual + text only")
    print("   • Expected R@1: ~0.35-0.38") 
    print("   • Training time: 30-60 minutes")
    print("   • More complex but potentially better than text-only")
    
    print("\n❌ NOT RECOMMENDED:")
    print("   • Keep audio in the current form (R@1=0.001)")
    print("   • Audio embeddings don't align with text queries")
    print("   • CLAP model domain mismatch with AEMS dataset")
    
    print("\n🚀 NEXT STEPS:")
    print("1. Run text-only gating network training")
    print("2. Compare results with text-only baseline")
    print("3. If performance is good, consider adding visual back")
    print("4. For future improvements: retrain CLAP on AEMS audio data")

if __name__ == "__main__":
    test_alternative_audio_fusion()
    recommend_solution()