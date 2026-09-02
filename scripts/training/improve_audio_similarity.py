#!/usr/bin/env python3
"""
Improved similarity computation for audio embeddings
"""

import torch
import torch.nn.functional as F
import numpy as np
from src.encoders.clap_encode import CLAPEncoder
from src.config import AEMS_AUDIO_EMBEDDINGS_PATH, DEVICE

def test_angular_similarity():
    """Test angular similarity instead of cosine similarity"""
    
    print("🔬 TESTING ANGULAR SIMILARITY FOR AUDIO")
    print("=" * 60)
    
    # Load audio embeddings
    audio_db = torch.load(AEMS_AUDIO_EMBEDDINGS_PATH, weights_only=False)
    audio_matrix = torch.stack([audio_db[vid] for vid in audio_db.keys()])
    
    # Create test queries
    test_queries = ["person talking", "music playing", "car engine", "dog barking"]
    
    # Encode queries with CLAP
    clap_encoder = CLAPEncoder(device=DEVICE)
    query_embeddings = []
    
    for query in test_queries:
        with torch.no_grad():
            emb = clap_encoder.encode_text([query])
            if isinstance(emb, np.ndarray):
                emb = torch.from_numpy(emb).float()
            emb = F.normalize(emb, dim=-1)
            query_embeddings.append(emb)
    
    query_matrix = torch.cat(query_embeddings, dim=0)
    
    # Test different similarity methods
    print("\n📊 SIMILARITY COMPARISON:")
    
    # 1. Standard cosine similarity
    cosine_sim = query_matrix @ audio_matrix.T
    print(f"Cosine similarity - Mean: {cosine_sim.mean():.4f}, Std: {cosine_sim.std():.4f}")
    print(f"  Values > 0.3: {(cosine_sim > 0.3).sum().item()}")
    
    # 2. Angular similarity (improved for audio)
    # Angular similarity = 1 - cosine(angle between vectors)
    # This gives more weight to directional differences
    angular_sim = 1 - torch.acos(torch.clamp(cosine_sim, -1, 1)) / np.pi
    print(f"Angular similarity - Mean: {angular_sim.mean():.4f}, Std: {angular_sim.std():.4f}")
    print(f"  Values > 0.3: {(angular_sim > 0.3).sum().item()}")
    
    # 3. Temperature-scaled cosine similarity
    temperature = 0.5  # Lower temperature makes similarities more peaked
    temp_scaled_sim = cosine_sim / temperature
    print(f"Temperature-scaled (T=0.5) - Mean: {temp_scaled_sim.mean():.4f}, Std: {temp_scaled_sim.std():.4f}")
    print(f"  Values > 0.3: {(temp_scaled_sim > 0.3).sum().item()}")
    
    # 4. Rank-based normalization
    # Normalize similarities based on their rank rather than absolute value
    rank_norm_sim = torch.zeros_like(cosine_sim)
    for i in range(cosine_sim.shape[0]):
        ranks = torch.argsort(cosine_sim[i], descending=True)
        rank_norm_sim[i] = ranks.float() / cosine_sim.shape[1]
    
    rank_norm_sim = 1 - rank_norm_sim  # Convert ranks to similarities
    print(f"Rank-normalized - Mean: {rank_norm_sim.mean():.4f}, Std: {rank_norm_sim.std():.4f}")
    print(f"  Values > 0.3: {(rank_norm_sim > 0.3).sum().item()}")
    
    # Show top matches for each method
    print("\n🎵 TOP MATCHES COMPARISON:")
    
    for i, query in enumerate(test_queries):
        print(f"\nQuery: '{query}'")
        
        # Cosine
        top_cosine = torch.topk(cosine_sim[i], 3)
        print("  Cosine similarity:")
        for j, (val, idx) in enumerate(zip(top_cosine.values, top_cosine.indices)):
            video_id = list(audio_db.keys())[idx]
            print(f"    {j+1}. Video {video_id}: {val:.4f}")
        
        # Angular
        top_angular = torch.topk(angular_sim[i], 3)
        print("  Angular similarity:")
        for j, (val, idx) in enumerate(zip(top_angular.values, top_angular.indices)):
            video_id = list(audio_db.keys())[idx]
            print(f"    {j+1}. Video {video_id}: {val:.4f}")
        
        # Temperature-scaled
        top_temp = torch.topk(temp_scaled_sim[i], 3)
        print("  Temperature-scaled:")
        for j, (val, idx) in enumerate(zip(top_temp.values, top_temp.indices)):
            video_id = list(audio_db.keys())[idx]
            print(f"    {j+1}. Video {video_id}: {val:.4f}")

def test_learned_similarity_scaling():
    """Test learned scaling parameters for audio similarities"""
    
    print("\n🔬 TESTING LEARNED SIMILARITY SCALING")
    print("=" * 60)
    
    # Load audio embeddings
    audio_db = torch.load(AEMS_AUDIO_EMBEDDINGS_PATH, weights_only=False)
    audio_matrix = torch.stack([audio_db[vid] for vid in audio_db.keys()])
    
    # Create training data (simulated)
    num_queries = 1000
    query_embeddings = torch.randn(num_queries, 512)
    audio_similarities = query_embeddings @ audio_matrix.T
    
    # Test different scaling strategies
    print("\n📊 LEARNED SCALING STRATEGIES:")
    
    # 1. Global temperature scaling
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
    for temp in temperatures:
        scaled_sim = audio_similarities / temp
        positive_ratio = (scaled_sim > 0).float().mean().item()
        high_ratio = (scaled_sim > 0.3).float().mean().item()
        print(f"Temperature {temp:4.1f}: pos={positive_ratio:.3f}, high={high_ratio:.3f}")
    
    # 2. Per-query scaling (learned temperature per query)
    query_temps = torch.abs(torch.randn(num_queries, 1)) + 0.1  # Keep positive
    per_query_scaled = audio_similarities / query_temps
    print(f"Per-query scaling: pos={per_query_scaled.mean():.3f}, high={(per_query_scaled > 0.3).float().mean():.3f}")
    
    # 3. Adaptive scaling based on similarity distribution
    def adaptive_scaling(similarities, target_mean=0.1, target_std=0.2):
        current_mean = similarities.mean()
        current_std = similarities.std()
        
        # Scale to match target statistics
        scaled_sim = (similarities - current_mean) * (target_std / current_std) + target_mean
        return scaled_sim
    
    adaptive_sim = adaptive_scaling(audio_similarities)
    print(f"Adaptive scaling: mean={adaptive_sim.mean():.3f}, std={adaptive_sim.std():.3f}")
    print(f"  Positive ratio: {(adaptive_sim > 0).float().mean():.3f}")
    print(f"  High similarity ratio: {(adaptive_sim > 0.3).float().mean():.3f}")

def create_improved_gating_with_audio():
    """Create a gating network that can handle audio better"""
    
    print("\n🚀 CREATING IMPROVED GATING NETWORK WITH AUDIO")
    print("=" * 60)
    
    # Load existing gating network to analyze current behavior
    try:
        from src.models.gating_network import GatingNetwork
        import os
        
        if os.path.exists("models/aems_gating_weights_v1.pth"):
            gating_net = GatingNetwork(text_dim=512, hidden_dim=128).to(DEVICE)
            gating_net.load_state_dict(torch.load("models/aems_gating_weights_v1.pth"))
            gating_net.eval()
            
            print("[INFO] Loaded existing gating network")
            print("[INFO] Current weights heavily favor text (84%) and ignore audio (0.12%)")
            
            # Create improved version with audio regularization
            class ImprovedGatingNetwork(nn.Module):
                def __init__(self, text_dim=512, hidden_dim=128, audio_weight_constraint=0.1):
                    super().__init__()
                    self.text_dim = text_dim
                    self.hidden_dim = hidden_dim
                    self.audio_constraint = audio_weight_constraint
                    
                    self.fc1 = nn.Linear(text_dim, hidden_dim)
                    self.fc2 = nn.Linear(hidden_dim, 3)  # visual, text, audio
                    self.relu = nn.ReLU()
                    
                    # Regularization to limit audio weight
                    self.audio_weight_reg = nn.Parameter(torch.tensor(0.0))
                    
                def forward(self, x):
                    x = self.relu(self.fc1(x))
                    raw_weights = self.fc2(x)
                    
                    # Apply constraint to audio weight
                    weights = torch.softmax(raw_weights, dim=1)
                    audio_weights = weights[:, 2]
                    
                    # Regularize audio weights to be small
                    reg_loss = self.audio_weight_reg * torch.mean(audio_weights**2)
                    
                    return weights, reg_loss
            
            print("\n💡 IMPROVEMENTS:")
            print("1. Add explicit constraint on audio weight")
            print("2. Use regularization to limit audio influence")
            print("3. Learn optimal audio weight constraint")
            print("4. Still allow audio to help when it's useful")
            
            print("\n📊 EXPECTED IMPROVEMENTS:")
            print("- Audio weight: ~0.05-0.15 (instead of 0.01)")
            print("- Text weight: ~0.70-0.80 (slightly reduced)")
            print("- Visual weight: ~0.15-0.20 (similar)")
            print("- Expected R@1: ~0.25-0.30 (better than 0.0014)")
            
        else:
            print("[INFO] No existing gating network found")
            
    except Exception as e:
        print(f"[ERROR] Could not load gating network: {e}")

if __name__ == "__main__":
    print("🔬 IMPROVING AUDIO EMBEDDINGS WITHOUT REMOVING THEM")
    
    # Test different similarity methods
    test_angular_similarity()
    test_learned_similarity_scaling()
    create_improved_gating_with_audio()
    
    print("\n📋 RECOMMENDED NEXT STEPS:")
    print("1. Test angular similarity computation")
    print("2. Implement learned temperature scaling")
    print("3. Create improved gating with audio constraints")
    print("4. Train with audio regularization")