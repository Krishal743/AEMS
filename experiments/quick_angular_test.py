#!/usr/bin/env python3
"""
Quick test of angular similarity vs cosine similarity on audio-heavy queries
Small-scale validation before full implementation
"""

import torch
import torch.nn.functional as F
import numpy as np
import clip
from src.encoders.clap_encode import CLAPEncoder
from src.evaluation.evaluate_retrieval import evaluate_retrieval
from src.config import AEMS_MANIFEST_PATH, AEMS_VID_EMBEDDINGS_PATH, AEMS_AUDIO_EMBEDDINGS_PATH, AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE, DEVICE
from src.data.metadata import load_metadata, filter_by_split

def select_audio_heavy_queries():
    """Select audio-heavy queries from test data"""
    print("🔍 SELECTING AUDIO-HEAVY QUERIES")
    
    # Load metadata
    metadata = load_metadata(AEMS_MANIFEST_PATH)
    test_items = filter_by_split(metadata, "test")
    
    # Audio-heavy query patterns
    audio_keywords = ['sound', 'audio', 'music', 'talking', 'speaking', 'noise', 
                     'barking', 'engine', 'footsteps', 'laughter', 'siren', 
                     'explosion', 'whisper', 'shouting', 'singing', 'playing']
    
    audio_heavy_queries = []
    audio_heavy_query_video_ids = []
    
    # Find queries with audio-related content
    for item in test_items:
        vid = item["video_id"]
        for q in item["qa_questions"]:
            # Check if query contains audio keywords
            if any(keyword in q.lower() for keyword in audio_keywords):
                audio_heavy_queries.append(q)
                audio_heavy_query_video_ids.append(vid)
                if len(audio_heavy_queries) >= 20:  # Limit to 20 queries
                    break
        if len(audio_heavy_queries) >= 20:
            break
    
    print(f"Selected {len(audio_heavy_queries)} audio-heavy queries")
    for i, query in enumerate(audio_heavy_queries[:10]):
        print(f"  {i+1}. '{query}'")
    
    return audio_heavy_queries, audio_heavy_query_video_ids

def select_video_subset(audio_heavy_query_video_ids, max_videos=250):
    """Select subset of videos for testing"""
    print(f"\n🎬 SELECTING VIDEO SUBSET (max {max_videos})")
    
    # Load all embeddings to check availability
    video_db = torch.load(AEMS_VID_EMBEDDINGS_PATH, weights_only=False)
    text_db_test = torch.load(AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE.format(split="test"), weights_only=False)
    
    all_video_ids = list(video_db.keys())
    
    # Filter videos that have both video and text embeddings
    available_videos = []
    for vid in all_video_ids:
        if vid in text_db_test:
            available_videos.append(vid)
    
    print(f"Total videos with both embeddings: {len(available_videos)}/{len(all_video_ids)}")
    
    # Include videos that are in our query set and available
    relevant_videos = []
    for vid in audio_heavy_query_video_ids:
        if vid in available_videos:
            relevant_videos.append(vid)
    
    # Add random videos to reach target size
    import random
    random.seed(42)  # For reproducibility
    remaining_slots = max_videos - len(relevant_videos)
    other_videos = [vid for vid in available_videos if vid not in relevant_videos]
    selected_other = random.sample(other_videos, min(remaining_slots, len(other_videos)))
    
    selected_videos = relevant_videos + selected_other
    print(f"Selected {len(selected_videos)} videos:")
    print(f"  - {len(relevant_videos)} query-relevant videos")
    print(f"  - {len(selected_other)} random videos")
    
    return selected_videos, video_db

def compute_embeddings(audio_heavy_queries, selected_videos, video_db):
    """Compute all necessary embeddings for the test"""
    print("\n🧠 COMPUTING EMBEDDINGS")
    
    # Load text embeddings for selected videos
    text_db_test = torch.load(AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE.format(split="test"), weights_only=False)
    
    # Build video matrices
    video_matrix = torch.stack([F.normalize(video_db[vid].float(), dim=0) for vid in selected_videos])
    text_matrix = torch.stack([F.normalize(text_db_test[vid].float(), dim=0) for vid in selected_videos])
    
    print(f"Video matrix: {video_matrix.shape}")
    print(f"Text matrix: {text_matrix.shape}")
    
    # Encode queries
    print("Encoding queries...")
    clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
    clip_model.eval()
    
    def encode_clip_queries(texts, batch_size=16):
        all_emb = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            tokens = clip.tokenize(batch, truncate=True).to(DEVICE)
            with torch.no_grad():
                emb = clip_model.encode_text(tokens)
                emb = F.normalize(emb, dim=-1)
            all_emb.append(emb.cpu())
        return torch.cat(all_emb, dim=0).float()  # Convert to float32
    
    query_clip = encode_clip_queries(audio_heavy_queries)
    print(f"Query embeddings: {query_clip.shape}")
    
    # Encode queries with CLAP for audio
    print("Encoding queries with CLAP...")
    clap_encoder = CLAPEncoder(device=DEVICE)
    
    def encode_clap_queries(texts, batch_size=16):
        all_emb = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            with torch.no_grad():
                emb = clap_encoder.encode_text(batch)
            if isinstance(emb, np.ndarray):
                emb = torch.from_numpy(emb).float()
            elif isinstance(emb, torch.Tensor):
                emb = F.normalize(emb, dim=-1)
            all_emb.append(emb.cpu())
        return torch.cat(all_emb, dim=0).float() if all_emb else torch.zeros(len(texts), 512)  # Convert to float32
    
    query_clap = encode_clap_queries(audio_heavy_queries)
    print(f"CLAP embeddings: {query_clap.shape}")
    
    return video_matrix, text_matrix, query_clip, query_clap, selected_videos

def test_similarity_methods(video_matrix, text_matrix, query_clip, query_clap, selected_videos, audio_heavy_query_video_ids, audio_heavy_queries):
    """Test different similarity computation methods"""
    print("\n🔬 TESTING SIMILARITY METHODS")
    
    # Ensure all tensors are float32
    video_matrix = video_matrix.float()
    text_matrix = text_matrix.float()
    query_clip = query_clip.float()
    query_clap = query_clap.float()
    
    # Create video ID to index mapping
    video_id_to_idx = {vid: i for i, vid in enumerate(selected_videos)}
    
    # Get ground truth indices for our queries
    ground_truth_indices = []
    for vid in audio_heavy_query_video_ids:
        if vid in video_id_to_idx:
            ground_truth_indices.append(video_id_to_idx[vid])
    
    print(f"Ground truth videos found: {len(ground_truth_indices)}/{len(audio_heavy_query_video_ids)}")
    
    # Method 1: Baseline (Cosine similarity)
    print("\n1️⃣ BASELINE: COSINE SIMILARITY")
    
    # Visual similarities
    sim_v_cosine = query_clip @ video_matrix.T
    sim_v_cosine = F.normalize(sim_v_cosine, dim=1)
    
    # Text similarities  
    sim_t_cosine = query_clip @ text_matrix.T
    sim_t_cosine = F.normalize(sim_t_cosine, dim=1)
    
    # Audio similarities
    sim_a_cosine = query_clap @ F.normalize(video_matrix, dim=1).T
    
    # Equal fusion
    sim_cosine_fusion = (sim_v_cosine + sim_t_cosine + sim_a_cosine) / 3
    
    print(f"   Visual - Mean: {sim_v_cosine.mean():.4f}, High (>0.3): {(sim_v_cosine > 0.3).sum().item()}")
    print(f"   Text - Mean: {sim_t_cosine.mean():.4f}, High (>0.3): {(sim_t_cosine > 0.3).sum().item()}")
    print(f"   Audio - Mean: {sim_a_cosine.mean():.4f}, High (>0.3): {(sim_a_cosine > 0.3).sum().item()}")
    print(f"   Fusion - Mean: {sim_cosine_fusion.mean():.4f}, High (>0.3): {(sim_cosine_fusion > 0.3).sum().item()}")
    
    # Method 2: Angular similarity (proposed)
    print("\n2️⃣ PROPOSED: ANGULAR SIMILARITY")
    
    def angular_similarity(query_emb, video_emb):
        """Compute angular similarity"""
        cosine_sim = query_emb @ video_emb.T
        # Convert to angular similarity (1 - angle/π)
        angular_sim = 1 - torch.acos(torch.clamp(cosine_sim, -1, 1)) / np.pi
        return angular_sim
    
    # Visual similarities (angular)
    sim_v_angular = angular_similarity(query_clip.float(), video_matrix)
    
    # Text similarities (angular)
    sim_t_angular = angular_similarity(query_clip.float(), text_matrix)
    
    # Audio similarities (angular)
    sim_a_angular = angular_similarity(query_clap.float(), video_matrix)
    
    # Temperature-scaled audio similarities
    temperature = 0.5
    sim_a_angular_temp = sim_a_angular / temperature
    
    # Equal fusion with temperature-scaled audio
    sim_angular_fusion = (sim_v_angular + sim_t_angular + sim_a_angular_temp) / 3
    
    print(f"   Visual - Mean: {sim_v_angular.mean():.4f}, High (>0.3): {(sim_v_angular > 0.3).sum().item()}")
    print(f"   Text - Mean: {sim_t_angular.mean():.4f}, High (>0.3): {(sim_t_angular > 0.3).sum().item()}")
    print(f"   Audio - Mean: {sim_a_angular.mean():.4f}, High (>0.3): {(sim_a_angular > 0.3).sum().item()}")
    print(f"   Audio (T=0.5) - Mean: {sim_a_angular_temp.mean():.4f}, High (>0.3): {(sim_a_angular_temp > 0.3).sum().item()}")
    print(f"   Fusion - Mean: {sim_angular_fusion.mean():.4f}, High (>0.3): {(sim_angular_fusion > 0.3).sum().item()}")
    
    # Method 3: Audio-only comparison
    print("\n3️⃣ AUDIO-ONLY COMPARISON")
    
    # Evaluate retrieval performance
    methods = {
        'Cosine Fusion': sim_cosine_fusion,
        'Angular Fusion': sim_angular_fusion,
        'Cosine Audio': sim_a_cosine,
        'Angular Audio': sim_a_angular,
        'Angular Audio (T=0.5)': sim_a_angular_temp,
    }
    
    results = {}
    for method_name, similarities in methods.items():
        try:
            # For this test, use simple retrieval (top-1 matches)
            correct_matches = 0
            total_queries = len(audio_heavy_queries)
            
            for i in range(total_queries):
                # Get top-1 prediction
                top_idx = torch.argmax(similarities[i])
                predicted_video = selected_videos[top_idx]
                
                # Check if it's correct (simplified - check if in ground truth)
                if predicted_video in audio_heavy_query_video_ids:
                    correct_matches += 1
            
            r1 = correct_matches / total_queries
            results[method_name] = {
                'R@1': r1,
                'mean_sim': similarities.mean().item(),
                'high_sim': (similarities > 0.3).sum().item()
            }
            
            print(f"   {method_name:25}: R@1={r1:.4f}, Mean={similarities.mean():.4f}, High={(similarities > 0.3).sum().item()}")
            
        except Exception as e:
            print(f"   {method_name:25}: ERROR - {e}")
    
    return results, sim_cosine_fusion, sim_angular_fusion

def analyze_top_examples(sim_cosine_fusion, sim_angular_fusion, audio_heavy_queries, selected_videos, top_k=3):
    """Analyze top examples for a few queries"""
    print(f"\n📊 TOP EXAMPLES ANALYSIS (showing top {top_k} for first 5 queries)")
    
    for i in range(min(5, len(audio_heavy_queries))):
        query = audio_heavy_queries[i]
        
        print(f"\nQuery: '{query}'")
        
        # Get top results for cosine method
        cosine_top_vals, cosine_top_indices = torch.topk(sim_cosine_fusion[i].float(), top_k)
        print("  Cosine Fusion Top:")
        for j, (val, idx) in enumerate(zip(cosine_top_vals, cosine_top_indices)):
            video_id = selected_videos[idx]
            print(f"    {j+1}. Video {video_id}: {val:.4f}")
        
        # Get top results for angular method  
        angular_top_vals, angular_top_indices = torch.topk(sim_angular_fusion[i].float(), top_k)
        print("  Angular Fusion Top:")
        for j, (val, idx) in enumerate(zip(angular_top_vals, angular_top_indices)):
            video_id = selected_videos[idx]
            print(f"    {j+1}. Video {video_id}: {val:.4f}")

def main():
    print("🚀 QUICK ANGULAR SIMILARITY TEST")
    print("=" * 50)
    print("Testing angular similarity vs cosine similarity on audio-heavy queries")
    print("Target: 200-300 videos, 15-20 audio-heavy queries")
    
    try:
        # Step 1: Select audio-heavy queries
        audio_heavy_queries, audio_heavy_query_video_ids = select_audio_heavy_queries()
        
        # Step 2: Select video subset
        selected_videos, video_db = select_video_subset(audio_heavy_query_video_ids, max_videos=250)
        
        # Step 3: Compute embeddings
        video_matrix, text_matrix, query_clip, query_clap, selected_videos = compute_embeddings(
            audio_heavy_queries, selected_videos, video_db
        )
        
        # Step 4: Test similarity methods
        results, sim_cosine_fusion, sim_angular_fusion = test_similarity_methods(
            video_matrix, text_matrix, query_clip, query_clap,
            selected_videos, audio_heavy_query_video_ids, audio_heavy_queries
        )
        
        # Step 5: Analyze top examples
        # Need to recompute fusion methods for analysis
        sim_v_cosine = query_clip.float() @ video_matrix.T
        sim_t_cosine = query_clip.float() @ text_matrix.T
        sim_a_cosine = query_clap.float() @ F.normalize(video_matrix, dim=1).T
        sim_cosine_fusion = (sim_v_cosine + sim_t_cosine + sim_a_cosine) / 3
        
        def angular_similarity(query_emb, video_emb):
            cosine_sim = query_emb @ video_emb.T
            angular_sim = 1 - torch.acos(torch.clamp(cosine_sim, -1, 1)) / np.pi
            return angular_sim
        
        sim_v_angular = angular_similarity(query_clip.float(), video_matrix)
        sim_t_angular = angular_similarity(query_clip.float(), text_matrix)
        sim_a_angular = angular_similarity(query_clap.float(), video_matrix)
        sim_a_angular_temp = sim_a_angular / 0.5
        sim_angular_fusion = (sim_v_angular + sim_t_angular + sim_a_angular_temp) / 3
        
        analyze_top_examples(sim_cosine_fusion, sim_angular_fusion, audio_heavy_queries, selected_videos)
        
        # Step 6: Summary
        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print("=" * 50)
        
        print("Performance Comparison:")
        for method, metrics in results.items():
            print(f"  {method:25}: R@1={metrics['R@1']:.4f}")
        
        # Find best improvement
        baseline_r1 = results.get('Cosine Fusion', {}).get('R@1', 0)
        best_r1 = max([m['R@1'] for m in results.values()])
        
        if baseline_r1 > 0:
            improvement = (best_r1 - baseline_r1) / baseline_r1
            print(f"\nImprovement over baseline: {improvement:.1f}x")
        
        print(f"\nRecommendation:")
        if best_r1 > baseline_r1 * 2:  # 2x improvement
            print("✅ ANGULAR SIMILARITY SHOWS SIGNIFICANT IMPROVEMENT")
            print("   Proceed with full implementation in main training script")
        elif best_r1 > baseline_r1:
            print("⚠️  ANGULAR SIMILARITY SHOWS MODEST IMPROVEMENT") 
            print("   Consider further refinement or alternative approaches")
        else:
            print("❌ ANGULAR SIMILARITY DOES NOT SHOW IMPROVEMENT")
            print("   Re-evaluate approach or consider alternative methods")
            
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()