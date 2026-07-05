# Diagnostic Report — AEMS Multimodal Retrieval

## Summary

The AEMS adaptive gating system was systematically audited, corrected (4 bugs), 
retrained, and evaluated under both the original (leaky) and a corrected 
leave-one-out protocol. Results show that:

1. **57% of caption retrieval success was exact-text cheating** — queries matched their own caption in the gallery, inflating R@1 from 39.7% to 92.7%.
2. **Audio (CLAP) is noise** on MSR-VTT (R@1=1.02%), a dataset characteristic.
3. **The 2-layer MLP gating network collapses** to a single modality regardless of training regime and does **not** beat simple Equal V+C fusion.
4. **Temporal transformer underperforms mean pooling** (R@1 20.3% vs 22.7%).

## Corrected Implementation (recovery-v2 branch)

### Bug Fixes Applied (Phase 2A)

| # | Bug | Files Changed | Impact |
|---|-----|---------------|--------|
| 1 | Train/val split at caption level (leakage) | `train_temporal_transformer.py` | Val R@1 dropped from ~0.9 to 0.436 after fix |
| 2 | Ranking loss used diagonal index (wrong for batching) | `retrain_gating.py`, `run_query_routing.py` | Gradients computed on wrong videos |
| 3 | Caption similarity used `max(dim=0)` across all captions | 5 eval/train files | Wrong pooling; switched to `max(dim=-1)` per-video |
| 4 | Audio branch used CLIP text (wrong modality) | 6 eval/train files | Replaced with CLAP text encoder |

### Heuristics Removed (Phase 3)
- `CAPTION_PENALTY=0.70`, `ENTROPY_PENALTY=0.05`, `ALIGNMENT_REWARD=0.1`
- Caption weight square penalty
- All set to zero in `run_query_routing.py` (the sole file with penalties)

## Evaluation Under Leave-One-Out Protocol

### Honest R@1 (no exact caption matching)

| System | Standard R@1 | LOO R@1 |
|--------|:-----------:|:-------:|
| Visual only | 0.2165 | 0.2165 |
| Caption only | 0.9270 | 0.3972 |
| Audio only | 0.0102 | 0.0102 |
| **Equal V+C** | 0.8919 | **0.4303** |
| Equal V+T+A | 0.3048 | 0.1288 |
| Adaptive (standard-trained) | 0.9273 | 0.3983 |
| AdaGate-LOO (LOO-trained) | — | 0.2496 |

### Key Findings

**Equal V+C fusion (43.03%) is the strongest baseline.** It beats:
- Visual alone (21.65%) by +21.4pp
- Caption alone (39.72%) by +3.3pp
- Both adaptive gating variants (39.83%, 24.96%)

**The gating network fails to outperform equal fusion** under either training regime:
- Standard-trained gate → collapses to caption-only (w_t ≈ 0.999)
- LOO-trained gate → collapses to visual-only (w_v ≈ 0.940)
- Neither learns per-query adaptive weighting; both collapse to a single modality

**Audio provides no benefit** (R@1=1.02%), and adding it via equal fusion degrades V+C from 43.03% to 12.88%.

## Error Analysis (Equal V+C Fusion, LOO)

### Rank Distribution
- R@1: 42.39% (25,347 queries)
- R@2-10: 28.93% (17,298 queries, near misses)
- R@11-100: 19.71% (11,784 queries)
- R@>100: 8.97% (5,365 queries)
- Median rank: **1** (more than half of queries rank the correct video first)
- In top-10: 71.3%
- In top-100: 91.0%

### What Makes Queries Hard

| Category | Mean Length | Example |
|----------|:-----------:|---------|
| R@1 successes | 10.1 words | `"a man is making a crab dish in the kitchen"` |
| R@2-10 | 9.2 words | `"a cabin in minecraft"` |
| R@11-100 | 8.4 words | `"gameplay footage of someone playing a game"` |
| R@>100 | 7.9 words | `"someone is playing a game"` |

**Short, generic queries fail most.** The top-1 retrieved video is typically semantically related but not the correct one (e.g., a car-driving video for a "man talking about a car" query).

## Temporal Transformer

| System | R@1 | R@5 | R@10 |
|--------|:---:|:---:|:----:|
| Mean pool | 0.2274 | 0.4316 | 0.5277 |
| Transformer | 0.2032 | 0.4373 | 0.5530 |

- Embedding similarity: only 0.28 cosine — the transformer learns a very different representation
- Low similarity + worse R@1 = transformer has not converged to a better embedding space
- Possible causes: insufficient training (4 epochs), tiny capacity (2M params), or frozen CLIP features already capture temporal information via mean pooling

## Archived Checkpoints

| File | Description |
|------|-------------|
| `models/gating_weights_meanpool.pth` | Standard-trained (penalty-free, exact-match regime) |
| `models/gating_weights_loo.pth` | LOO-trained (1 epoch, val R@1=0.2411) |

## Commitment

**Adaptive gating is an investigated hypothesis, not a successful contribution.**
The 2-layer MLP gating network (512→128→3, softmax) using CLIP text embeddings 
as input cannot learn per-query modality weighting that exceeds simple equal fusion.

## Future Work — Per-Video Gating (Design Sketch)

### Problem
The current `GatingNetwork` computes `w = MLP(clip_text_embed(query))` — a function 
of the query alone. It cannot distinguish "this query has a distinctive visual match" 
from "this query has a distinctive caption match" because it sees neither the video 
features nor the similarity scores. The optimal weight for query `q` and video `v` 
is a function of the pair `(q, v)`, but the current architecture treats it as a 
function of `q` alone, forcing a single global policy.

### Proposed Architecture

```
query_embed = CLIP_text(query)          # [512]
video_embed = CLIP_video(video)          # [512]
caption_embeds = CLIP_text(captions)     # [20, 512]
audio_embed = CLAP_audio(audio)          # [512]   (if available)

sim_v = dot(query_embed, video_embed)         # scalar
sim_t = max(dot(query_embed, caption_embeds)) # scalar
sim_a = dot(CLAP_text(query), audio_embed)    # scalar

# Per-video gating: weights depend on (query, video) pair
pair_features = [
    query_embed,           # what the query is about
    video_embed,           # what the video looks like
    sim_v, sim_t, sim_a,   # how well they match per modality
    abs(sim_v - sim_t),    # modality disagreement signal
    max(sim_v, sim_t, sim_a),  # best modality score
]

gating_input = concat(pair_features)  # [512*2 + 5] ≈ [1029]
w_v, w_t, w_a = MLP(gating_input)    # softmax over modalities

score = w_v * sim_v + w_t * sim_t + w_a * sim_a
```

### Key Differences from Current Architecture

| Aspect | Current | Proposed |
|--------|---------|----------|
| Gate input | Query CLIP text only (512-dim) | Query + video + scores (1029-dim) |
| Weight policy | Global per query | Per (query, video) pair |
| Scale | O(N) weights for N queries | O(N×M) weights for N queries × M videos |
| Training | Ranking loss over gated scores | Same ranking loss (drop-in) |
| Inference | One forward pass per query | Must compute per query×video |

### Implementation Strategy

Since per-video gating requires computing weights for all `N × M` query-video 
pairs, it is O(N×M) during inference. For the test set (52,694 × 2,635 ≈ 139M pairs), 
this requires batching but is computationally feasible.

The gating network architecture stays the same (2-layer MLP), only its input 
changes. Training uses the same ranking loss with hard negatives. The expectation 
is that per-video gating can learn to up-weight visual similarity for queries 
where `sim_v` is high, and caption similarity for queries where `sim_t` is high,
outperforming equal fusion.

### Cross-Attention Fusion (Alternative)

An alternative to per-video gating is replacing the weighted sum with a 
query-key-value attention mechanism:

```
Q = query_embed                               # [512]
K = [video_embed, cap_embed, audio_embed]      # [3, 512]
V = [sim_v, sim_t, sim_a]                     # [3]

attn = softmax(Q @ K^T / sqrt(512))            # [3]
score = attn @ V                               # scalar
```

This is more parameter-efficient than per-video gating (no MLP at all) and 
provides a principled way to let the query attend to the most relevant modality.

### Dataset Consideration

MSR-VTT audio is essentially random relative to queries (R@1=1.02%). A 
meaningful evaluation of audio-visual fusion requires an audio-rich dataset 
such as AudioCaps or VGGSound. The same pipeline applies: only the audio 
encoder changes.
