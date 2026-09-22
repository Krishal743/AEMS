#!/usr/bin/env python3
"""
Audio embedding diagnostics - step-by-step validation
"""

import torch
import torch.nn.functional as F
import numpy as np
from src.encoders.clap_encode import CLAPEncoder
from src.config import AEMS_CLAP_AUDIO_EMBEDDINGS_PATH, DEVICE

def check_audio_embeddings():
    """Step 1: Check if audio embeddings exist and have reasonable properties"""
    print("=" * 60)
    print("STEP 1: AUDIO EMBEDDING VALIDATION")
    print("=" * 60)
    
    # Load audio embeddings
    print(f"[LOAD] Loading audio embeddings from {AEMS_CLAP_AUDIO_EMBEDDINGS_PATH}")
    audio_db = torch.load(AEMS_CLAP_AUDIO_EMBEDDINGS_PATH, weights_only=False)
    
    print(f"[INFO] Number of audio embeddings: {len(audio_db)}")
    print(f"[INFO] Keys (video IDs): {list(audio_db.keys())[:5]}...")  # First 5 keys
    
    # Check embedding properties
    all_embeddings = []
    for vid, emb in audio_db.items():
        all_embeddings.append(emb)
    
    all_embeddings = torch.stack(all_embeddings)
    print(f"[INFO] Audio embedding shape: {all_embeddings.shape}")
    print(f"[INFO] Audio embedding dtype: {all_embeddings.dtype}")
    print(f"[INFO] Audio embedding device: {all_embeddings.device}")
    
    # Check statistics
    print(f"[INFO] Mean: {all_embeddings.mean():.6f}")
    print(f"[INFO] Std: {all_embeddings.std():.6f}")
    print(f"[INFO] Min: {all_embeddings.min():.6f}")
    print(f"[INFO] Max: {all_embeddings.max():.6f}")
    
    # Check if embeddings are normalized
    norms = torch.norm(all_embeddings, dim=1)
    print(f"[INFO] L2 norms - Mean: {norms.mean():.6f}, Std: {norms.std():.6f}")
    print(f"[INFO] Norms close to 1.0: {(norms > 0.9).sum()}/{len(norms)}")
    
    # Check for NaN/inf values
    has_nan = torch.isnan(all_embeddings).any()
    has_inf = torch.isinf(all_embeddings).any()
    print(f"[INFO] Has NaN values: {has_nan}")
    print(f"[INFO] Has Inf values: {has_inf}")
    
    return audio_db, all_embeddings

def check_clap_encoder():
    """Step 2: Test CLAP encoder functionality"""
    print("\n" + "=" * 60)
    print("STEP 2: CLAP ENCODER VALIDATION")
    print("=" * 60)
    
    try:
        print("[INIT] Initializing CLAP encoder...")
        clap_encoder = CLAPEncoder(device=DEVICE)
        print("[OK] CLAP encoder initialized successfully")
        
        # Test with a simple query
        test_queries = ["a person talking", "music playing", "car engine sound"]
        
        print(f"[TEST] Testing CLAP encoder with {len(test_queries)} queries...")
        for i, query in enumerate(test_queries):
            print(f"[TEST] Query {i+1}: '{query}'")
            try:
                with torch.no_grad():
                    emb = clap_encoder.encode_text([query])
                    if isinstance(emb, np.ndarray):
                        emb = torch.from_numpy(emb).float()
                    emb = F.normalize(emb, dim=-1)
                    
                    print(f"[OK] Encoded shape: {emb.shape}")
                    print(f"[OK] Encoded norm: {torch.norm(emb):.6f}")
                    
                    # Check for issues
                    if torch.isnan(emb).any():
                        print("[ERROR] NaN values in encoded embedding!")
                    if torch.isinf(emb).any():
                        print("[ERROR] Inf values in encoded embedding!")
                        
            except Exception as e:
                print(f"[ERROR] Failed to encode query '{query}': {e}")
                
    except Exception as e:
        print(f"[ERROR] Failed to initialize CLAP encoder: {e}")
        return None
    
    return clap_encoder

def check_similarity_computation(audio_db, test_queries, clap_encoder):
    """Step 3: Test similarity computation between audio and text"""
    print("\n" + "=" * 60)
    print("STEP 3: SIMILARITY COMPUTATION VALIDATION")
    print("=" * 60)
    
    # Build audio matrix
    print("[BUILD] Building audio matrix...")
    audio_matrix = torch.stack([F.normalize(audio_db[vid].float(), dim=0) for vid in audio_db.keys()])
    print(f"[INFO] Audio matrix shape: {audio_matrix.shape}")
    
    # Encode test queries
    print("[ENCODE] Encoding test queries...")
    query_embeddings = []
    for query in test_queries:
        with torch.no_grad():
            emb = clap_encoder.encode_text([query])
            if isinstance(emb, np.ndarray):
                emb = torch.from_numpy(emb).float()
            emb = F.normalize(emb, dim=-1)
            query_embeddings.append(emb)
    
    query_matrix = torch.cat(query_embeddings, dim=0)
    print(f"[INFO] Query matrix shape: {query_matrix.shape}")
    
    # Compute similarities
    print("[COMPUTE] Computing audio-text similarities...")
    similarities = query_matrix @ audio_matrix.T
    
    print(f"[INFO] Similarity matrix shape: {similarities.shape}")
    print(f"[INFO] Similarity stats - Mean: {similarities.mean():.6f}, Std: {similarities.std():.6f}")
    print(f"[INFO] Similarity range: [{similarities.min():.6f}, {similarities.max():.6f}]")
    
    # Check for reasonable similarity distribution
    print(f"[INFO] Similarities > 0.5: {(similarities > 0.5).sum()}")
    print(f"[INFO] Similarities > 0.1: {(similarities > 0.1).sum()}")
    print(f"[INFO] Similarities < -0.1: {(similarities < -0.1).sum()}")
    
    # Check top similarities for each query
    print("\n[TOP SIMILARITIES] Top 3 similarities for each query:")
    for i, query in enumerate(test_queries):
        top_vals, top_indices = torch.topk(similarities[i], 3)
        print(f"Query '{query}':")
        for j, (val, idx) in enumerate(zip(top_vals, top_indices)):
            video_id = list(audio_db.keys())[idx]
            print(f"  {j+1}. Video {video_id}: {val:.6f}")
    
    return similarities

def check_ground_truth_alignment():
    """Step 4: Check if audio embeddings align with expected ground truth"""
    print("\n" + "=" * 60)
    print("STEP 4: GROUND TRUTH ALIGNMENT CHECK")
    print("=" * 60)
    
    # This would require loading the metadata and checking if relevant videos
    # actually contain the expected audio content
    
    print("[INFO] This step requires manual verification:")
    print("1. Check if metadata correctly labels video content")
    print("2. Verify that audio extraction worked correctly")
    print("3. Ensure CLAP model understands the audio-text relationship")
    print("4. Check if there's a domain mismatch (e.g., trained on different data)")

if __name__ == "__main__":
    print("🔍 AUDIO EMBEDDING DIAGNOSTICS")
    print("This script will help identify why audio embeddings perform poorly")
    
    # Step 1: Check audio embeddings
    audio_db, audio_embeddings = check_audio_embeddings()
    
    # Step 2: Check CLAP encoder
    clap_encoder = check_clap_encoder()
    
    if clap_encoder is not None and audio_db is not None:
        # Step 3: Check similarity computation
        test_queries = ["person talking", "music", "car", "dog barking", "footsteps"]
        similarities = check_similarity_computation(audio_db, test_queries, clap_encoder)
        
        # Step 4: Ground truth alignment
        check_ground_truth_alignment()
        
        print("\n" + "=" * 60)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 60)
        print("✅ Audio embeddings loaded successfully")
        print("✅ CLAP encoder working")
        print("✅ Similarity computation functional")
        print("🤔 Check if the computed similarities make sense for your domain")
        print("🔬 If similarities look random, the issue might be:")
        print("   - CLAP model not trained on your audio domain")
        print("   - Audio extraction issues")
        print("   - Text-audio mismatch in your dataset")
    else:
        print("❌ Found critical issues - need to fix before proceeding")