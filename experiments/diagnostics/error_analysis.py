"""
Error analysis: where does V+C fusion succeed vs fail?
Only needs CLIP (not CLAP) to avoid long loading times.
"""
import json, torch, os, gc
import clip
import numpy as np
from collections import defaultdict, Counter
from src.evaluation.evaluate_retrieval import evaluate_retrieval

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    os.makedirs("outputs/diagnostics", exist_ok=True)

    print("Loading metadata...")
    with open("data/processed/metadata/msrvtt_metadata.json") as f:
        metadata = json.load(f)
    test_items = [m for m in metadata if m["split"] == "test"]
    test_video_ids = set(m["video_id"] for m in test_items)

    print("Loading embeddings...")
    video_db = torch.load("embeddings/video_embeddings.pt", weights_only=False)
    caption_test_db = torch.load("embeddings/caption_embeddings_test.pt", weights_only=False)
    common = [v for v in video_db if v in caption_test_db and v in test_video_ids]
    print(f"Common videos: {len(common)}")

    test_items_f = [m for m in test_items if m["video_id"] in common]
    test_texts = [m["text"] for m in test_items_f]
    test_vids = [m["video_id"] for m in test_items_f]
    print(f"Queries: {len(test_texts)}")

    # Build video text->index mapping for LOO
    from collections import defaultdict
    video_captions = defaultdict(list)
    for m in test_items_f:
        video_captions[m["video_id"]].append(m["text"])
    cap_index = {}
    for vid, caps in video_captions.items():
        for i, t in enumerate(caps):
            cap_index[(vid, t)] = i

    # Modality matrices
    print("Building matrices...")
    vid_m = torch.stack([torch.as_tensor(video_db[v]).float() for v in common])
    vid_m = torch.nn.functional.normalize(vid_m, p=2, dim=1)
    del video_db
    gc.collect()

    # CLIP text embeddings
    print("Encoding queries...")
    clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
    clip_model.eval()
    text_embeds = []
    for i in range(0, len(test_texts), 256):
        batch = test_texts[i:i+256]
        tokens = clip.tokenize(batch).to(DEVICE)
        with torch.no_grad():
            emb = clip_model.encode_text(tokens).float()
            emb = emb / emb.norm(dim=1, keepdim=True)
        text_embeds.append(emb.cpu())
    text_embeds = torch.cat(text_embeds, dim=0)
    sim_v = text_embeds @ vid_m.T
    del clip_model
    gc.collect()

    # Sim_t LOO
    print("Caption similarity (LOO)...")
    sim_t_loo_list = []
    for vid in common:
        cap_embeds = torch.as_tensor(caption_test_db[vid]).float()
        cap_embeds = torch.nn.functional.normalize(cap_embeds, p=2, dim=1).to(DEVICE)
        sims = text_embeds.to(DEVICE) @ cap_embeds.T
        max_sims = sims.max(dim=1).values.clone()
        for qi, (qt, qv) in enumerate(zip(test_texts, test_vids)):
            if qv == vid:
                ci = cap_index.get((vid, qt))
                if ci is not None:
                    row = sims[qi].clone()
                    row[ci] = -float('inf')
                    max_sims[qi] = row.max()
        sim_t_loo_list.append(max_sims.cpu())
        del cap_embeds, sims
        gc.collect()
    sim_t_loo = torch.stack(sim_t_loo_list, dim=1)
    del caption_test_db, text_embeds
    gc.collect()

    sim_eq = (sim_v + sim_t_loo) / 2
    del sim_v, sim_t_loo
    gc.collect()

    # Per-query analysis
    print("\n=== ERROR ANALYSIS ===")

    def get_rank(sim, vids, common_vids):
        ranks = []
        for i in range(len(vids)):
            ranked = torch.argsort(sim[i], descending=True)
            pos = (ranked.tolist().index(list(common_vids).index(vids[i])))
            ranks.append(pos)
        return np.array(ranks)

    ranks = get_rank(sim_eq, test_vids, common)

    # Rank distribution
    print(f"\nRank distribution of correct video under V+C equal fusion:")
    buckets = [(0, "R@1"), (1, "R@2"), (2, "R@5"), (5, "R@10"),
               (10, "R@50"), (50, "R@100"), (100, "R@500"), (500, "R@2635")]
    prev = 0
    for threshold, label in buckets:
        count = ((ranks >= prev) & (ranks < threshold)).sum() if threshold < 2635 else (ranks >= prev).sum()
        pct = 100 * count / len(ranks)
        print(f"  {label:<10} {count:>6} ({pct:>5.2f}%)")
        prev = threshold

    # Categories of success/failure
    def per_query_rank(sim_2d, vids, common_vids):
        """Returns rank per query."""
        ranks = []
        for i in range(len(vids)):
            gt = vids[i]
            ranked = torch.argsort(sim_2d[i], descending=True)
            pos = (ranked.tolist().index(list(common_vids).index(gt)))
            ranks.append(pos)
        return np.array(ranks)

    # Divide into categories by rank
    cat_r1 = ranks == 0
    cat_near = (ranks >= 1) & (ranks < 10)
    cat_far = (ranks >= 10) & (ranks < 100)
    cat_very_far = ranks >= 100

    print(f"\nSuccess categories:")
    print(f"  R@1 (perfect):        {cat_r1.sum():>6} ({100*cat_r1.sum()/len(ranks):.2f}%)")
    print(f"  Near miss (R@2-10):   {cat_near.sum():>6} ({100*cat_near.sum()/len(ranks):.2f}%)")
    print(f"  Far (R@11-100):       {cat_far.sum():>6} ({100*cat_far.sum()/len(ranks):.2f}%)")
    print(f"  Very far (R@>100):    {cat_very_far.sum():>6} ({100*cat_very_far.sum()/len(ranks):.2f}%)")

    # Show example queries from each category
    print(f"\n=== EXAMPLE QUERIES ===")

    # Helper: find video topic from its captions
    def video_theme(metadata, vid, split="test"):
        caps = [m["text"] for m in metadata if m["video_id"] == vid and m["split"] == split]
        return caps[:3]

    def show_category(name, indices, limit=5):
        print(f"\n--- {name} (showing up to {limit}) ---")
        for idx in indices[:limit]:
            rank = ranks[idx]
            print(f"  Q: '{test_texts[idx][:70]}...' " if len(test_texts[idx]) > 70 else f"  Q: '{test_texts[idx]}'")
            gt_vid = test_vids[idx]
            # What was the top-1 retrieved video?
            ranked = torch.argsort(sim_eq[idx], descending=True)
            top1_vid = common[ranked[0].item()]
            top1_caps = video_theme(metadata, top1_vid, "test")
            print(f"  GT: {gt_vid}  |  Retrieved: {top1_vid} (rank={rank})")
            print(f"  Top-1 captions: {[c[:50] for c in top1_caps[:2]]}")
            print()

    # Random samples from each category
    np.random.seed(42)
    r1_indices = np.where(cat_r1)[0]
    near_indices = np.where(cat_near)[0]
    far_indices = np.where(cat_far)[0]
    very_far_indices = np.where(cat_very_far)[0]

    np.random.shuffle(r1_indices)
    np.random.shuffle(near_indices)
    np.random.shuffle(far_indices)
    np.random.shuffle(very_far_indices)

    show_category("R@1 SUCCESSES", r1_indices[:5])
    show_category("NEAR MISSES (R@2-10)", near_indices[:5])
    show_category("FAR FAILURES (R@11-100)", far_indices[:5])
    show_category("VERY FAR FAILURES (R@>100)", very_far_indices[:5])

    # Text length analysis
    print("\n=== TEXT LENGTH ANALYSIS ===")
    lengths = np.array([len(t.split()) for t in test_texts])
    for name, mask in [("All", np.ones_like(lengths, dtype=bool)),
                       ("R@1", cat_r1),
                       ("R@2-10", cat_near),
                       ("R@11-100", cat_far),
                       ("R@>100", cat_very_far)]:
        if mask.sum() > 0:
            print(f"  {name:<15} mean_len={lengths[mask].mean():.1f} std={lengths[mask].std():.1f}")

    # Most common words in failures vs successes
    from collections import Counter
    print("\n=== VOCABULARY ANALYSIS ===")
    def word_counts(indices):
        words = []
        for idx in indices:
            words.extend(test_texts[idx].lower().split())
        return Counter(words)

    # Remove stopwords
    stopwords = {"a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
                 "have", "has", "had", "do", "does", "did", "will", "would", "could",
                 "should", "may", "might", "can", "shall", "to", "of", "in", "for",
                 "on", "with", "at", "by", "from", "as", "and", "or", "but", "not",
                 "no", "it", "its", "this", "that", "these", "those", "i", "you",
                 "he", "she", "they", "we", "my", "your", "his", "her", "their",
                 "our", "me", "him", "them", "us", "who", "what", "which", "how"}

    def filtered_counts(counter):
        return {w: c for w, c in counter.items() if w not in stopwords and len(w) > 2}

    r1_words = filtered_counts(word_counts(r1_indices[:5000]))
    fail_words = filtered_counts(word_counts(very_far_indices))

    print("  Top words in R@1 successes:")
    for w, c in sorted(r1_words.items(), key=lambda x: -x[1])[:10]:
        print(f"    {w:<20} {c}")

    print("  Top words in failures (R>100):")
    for w, c in sorted(fail_words.items(), key=lambda x: -x[1])[:10]:
        print(f"    {w:<20} {c}")

    # What distinguishes failures from successes
    r1_set = set(r1_words.keys())
    fail_set = set(fail_words.keys())
    only_in_failures = fail_set - r1_set
    only_in_r1 = r1_set - fail_set

    print(f"\n  Words ONLY in failures (not in top-5k successes):")
    for w in sorted(only_in_failures, key=lambda x: -fail_words.get(x, 0))[:10]:
        print(f"    {w}: {fail_words.get(w, 0)}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    metrics = evaluate_retrieval(sim_eq, test_vids, common, ks=[1, 5, 10, 50])
    print(f"Equal V+C fusion (LOO):")
    print(f"  R@1={metrics['R@1']:.4f} R@5={metrics['R@5']:.4f} R@10={metrics['R@10']:.4f} R@50={metrics['R@50']:.4f}")
    print(f"  Median rank: {np.median(ranks):.0f}")
    print(f"  Mean rank:   {np.mean(ranks):.2f}")
    print(f"  Queries in top-10: {(ranks < 10).mean()*100:.2f}%")
    print(f"  Queries in top-100: {(ranks < 100).mean()*100:.2f}%")

if __name__ == "__main__":
    main()
