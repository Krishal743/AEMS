#!/usr/bin/env python3
"""
Alternative audio embedding computation - try different approaches
"""

import torch
import torch.nn.functional as F
import numpy as np
from src.encoders.clap_encode import CLAPEncoder
from src.config import AEMS_CLAP_AUDIO_EMBEDDINGS_PATH, DEVICE

def test_different_audio_features():
    """Test different audio feature extraction methods"""
    
    print("🔬 TESTING ALTERNATIVE AUDIO FEATURE EXTRACTION")
    print("=" * 60)
    
    # Load existing audio embeddings to compare
    audio_db = torch.load(AEMS_CLAP_AUDIO_EMBEDDINGS_PATH, weights_only=False)
    original_embeddings = torch.stack([audio_db[vid] for vid in audio_db.keys()])
    
    print(f"[INFO] Original audio embeddings: {original_embeddings.shape}")
    print(f"[INFO] Original stats - mean={original_embeddings.mean():.4f}, std={original_embeddings.std():.4f}")
    
    # Test different normalization approaches
    print("\n📊 DIFFERENT NORMALIZATION APPROACHES:")
    
    # 1. Current L2 normalization
    current_norm = F.normalize(original_embeddings, dim=1)
    print(f"1. Current L2 norm - Mean: {current_norm.norm(dim=1).mean():.4f}")
    
    # 2. Standard scaling (z-score)
    audio_mean = original_embeddings.mean(dim=1, keepdim=True)
    audio_std = original_embeddings.std(dim=1, keepdim=True)
    z_score_norm = (original_embeddings - audio_mean) / (audio_std + 1e-8)
    print(f"2. Z-score normalization - Mean: {z_score_norm.mean():.4f}, Std: {z_score_norm.std():.4f}")
    
    # 3. Min-max scaling
    audio_min = original_embeddings.min(dim=1, keepdim=True)[0]
    audio_max = original_embeddings.max(dim=1, keepdim=True)[0]
    min_max_norm = (original_embeddings - audio_min) / (audio_max - audio_min + 1e-8)
    print(f"3. Min-max normalization - Range: [{min_max_norm.min():.4f}, {min_max_norm.max():.4f}]")
    
    # 4. Power normalization
    power_norm = torch.sign(original_embeddings) * torch.sqrt(torch.abs(original_embeddings))
    power_norm = F.normalize(power_norm, dim=1)
    print(f"4. Power normalization - Mean: {power_norm.mean():.4f}, Std: {power_norm.std():.4f}")
    
    return {
        'original': original_embeddings,
        'current_l2': current_norm,
        'z_score': z_score_norm,
        'min_max': min_max_norm,
        'power': power_norm
    }

def test_similarity_improvements():
    """Test different similarity computation methods"""
    
    print("\n🔬 TESTING SIMILARITY COMPUTATION IMPROVEMENTS")
    print("=" * 60)
    
    # Load audio embeddings
    audio_db = torch.load(AEMS_CLAP_AUDIO_EMBEDDINGS_PATH, weights_only=False)
    audio_matrix = torch.stack([audio_db[vid] for vid in audio_db.keys()])
    
    # Create dummy query embeddings (simulate text queries)
    num_queries = 100
    num_videos = len(audio_db)
    query_embeddings = torch.randn(num_queries, 512)
    
    print(f"[INFO] Testing with {num_queries} queries, {num_videos} videos")
    
    # Test different similarity methods
    similarity_methods = {
        'Cosine': lambda q, a: F.normalize(q, dim=1) @ F.normalize(a, dim=1).T,
        'Dot Product': lambda q, a: q @ a.T,
        'Manhattan': lambda q, a: -torch.cdist(q, a, p=1),
        'Euclidean': lambda q, a: -torch.cdist(q, a, p=2),
        'Angular': lambda q, a: (q @ a.T) / (torch.norm(q, dim=1, keepdim=True) @ torch.norm(a, dim=1, keepdim=True).T + 1e-8)
    }
    
    results = {}
    for method_name, sim_func in similarity_methods.items():
        try:
            similarities = sim_func(query_embeddings, audio_matrix)
            
            # Check for reasonable distribution
            mean_sim = similarities.mean().item()
            std_sim = similarities.std().item()
            positive_ratio = (similarities > 0).float().mean().item()
            high_sim_ratio = (similarities > 0.3).float().mean().item()
            
            results[method_name] = {
                'mean': mean_sim,
                'std': std_sim,
                'positive_ratio': positive_ratio,
                'high_sim_ratio': high_sim_ratio,
                'similarities': similarities
            }
            
            print(f"{method_name:12}: mean={mean_sim:6.4f}, std={std_sim:6.4f}, pos={positive_ratio:6.3f}, high={high_sim_ratio:6.3f}")
            
        except Exception as e:
            print(f"{method_name:12}: ERROR - {e}")
    
    return results

def test_clap_alternatives():
    """Test different CLAP model configurations"""
    
    print("\n🔬 TESTING CLAP MODEL ALTERNATIVES")
    print("=" * 60)
    
    # Test different text encoding strategies
    test_queries = [
        "person talking", "music playing", "car engine", "dog barking", 
        "footsteps", "laughter", "siren", "explosion"
    ]
    
    try:
        clap_encoder = CLAPEncoder(device=DEVICE)
        
        print("[INFO] Testing CLAP encoding with different approaches:")
        
        # 1. Direct encoding
        direct_embeddings = []
        for query in test_queries:
            with torch.no_grad():
                emb = clap_encoder.encode_text([query])
                if isinstance(emb, np.ndarray):
                    emb = torch.from_numpy(emb).float()
                direct_embeddings.append(F.normalize(emb, dim=-1))
        
        direct_matrix = torch.cat(direct_embeddings, dim=0)
        print(f"1. Direct encoding: {direct_matrix.shape}")
        
        # 2. Augmented encoding (add context)
        augmented_queries = [f"a video of {q}" for q in test_queries]
        augmented_embeddings = []
        for query in augmented_queries:
            with torch.no_grad():
                emb = clap_encoder.encode_text([query])
                if isinstance(emb, np.ndarray):
                    emb = torch.from_numpy(emb).float()
                augmented_embeddings.append(F.normalize(emb, dim=-1))
        
        augmented_matrix = torch.cat(augmented_embeddings, dim=0)
        print(f"2. Augmented encoding: {augmented_matrix.shape}")
        
        # 3. Multiple query encoding (average of related queries)
        query_groups = [
            ["person talking", "person speaking", "human voice"],
            ["music playing", "musical instrument", "song"],
            ["car engine", "vehicle sound", "car noise"],
            ["dog barking", "dog sound", "animal noise"]
        ]
        
        group_embeddings = []
        for group in query_groups:
            group_emb = []
            for query in group:
                with torch.no_grad():
                    emb = clap_encoder.encode_text([query])
                    if isinstance(emb, np.ndarray):
                        emb = torch.from_numpy(emb).float()
                    group_emb.append(F.normalize(emb, dim=-1))
            group_embeddings.append(torch.mean(torch.stack(group_emb), dim=0))
        
        group_matrix = torch.cat(group_embeddings, dim=0)
        print(f"3. Group encoding: {group_matrix.shape}")
        
        return {
            'direct': direct_matrix,
            'augmented': augmented_matrix,
            'group': group_matrix
        }
        
    except Exception as e:
        print(f"[ERROR] CLAP testing failed: {e}")
        return None

def recommend_audio_fixes():
    """Recommend specific fixes for audio embeddings"""
    
    print("\n💡 RECOMMENDED AUDIO FIXES")
    print("=" * 60)
    
    print("🔧 IMMEDIATE FIXES:")
    print("1. Try different similarity computation methods")
    print("   - Replace cosine similarity with angular similarity")
    print("   - Add temperature scaling to similarities")
    print("   - Use learned similarity metrics")
    
    print("\n2. Improve CLAP text encoding:")
    print("   - Add context: 'a video of [query]' instead of just query")
    print("   - Use query augmentation for better coverage")
    print("   - Try different CLAP model variants")
    
    print("\n3. Audio preprocessing improvements:")
    print("   - Re-compute audio embeddings with different normalization")
    print("   - Try power normalization instead of L2")
    print("   - Apply domain-specific audio preprocessing")
    
    print("\n4. Similarity post-processing:")
    print("   - Apply learned temperature scaling")
    print("   - Use rank-based similarity normalization")
    print("   - Implement cross-modal alignment techniques")
    
    print("\n🚀 PRIORITY ORDER:")
    print("1. Try angular similarity computation (quick test)")
    print("2. Test augmented text encoding ('a video of [query]')")
    print("3. Experiment with power normalization")
    print("4. Implement learned similarity scaling")

if __name__ == "__main__":
    print("🔬 ALTERNATIVE AUDIO EMBEDDING IMPROVEMENTS")
    print("Finding ways to improve audio performance without removing it")
    
    # Test different approaches
    norm_results = test_different_audio_features()
    sim_results = test_similarity_improvements()
    clap_results = test_clap_alternatives()
    
    # Get recommendations
    recommend_audio_fixes()
    
    print("\n📋 NEXT STEPS:")
    print("1. Run similarity computation tests")
    print("2. Test angular similarity vs cosine")
    print("3. Try augmented text encoding")
    print("4. Implement the best performing combination")