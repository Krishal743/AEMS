# System Pseudocode

---

## 1. Implementation Overview

### 1.1 Project Summary
**Project Name**: Query-Conditioned Adaptive Video Retrieval
**Task**: Text-to-Video retrieval on MSR-VTT dataset using multimodal embeddings with learned query-adaptive routing
**Core Idea**: Use CLIP (visual), CLAP (audio), and caption embeddings with a gating network that learns which modality to trust per query

### 1.2 Technologies Used

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Visual Encoder** | OpenAI CLIP (ViT-B/32) | Encode video frames to 512-dim embeddings |
| **Audio Encoder** | LAION CLAP | Encode audio to 512-dim embeddings |
| **Text Encoder** | OpenAI CLIP (ViT-B/32) | Encode query text and captions |
| **Frame Extraction** | ffmpeg | Extract frames from videos at uniform timestamps |
| **Audio Extraction** | ffmpeg | Extract audio from videos |
| **Video Loading** | PIL (Pillow) | Load and preprocess extracted frames |
| **Audio Loading** | librosa | Load audio at 48kHz for CLAP |
| **Deep Learning** | PyTorch | Gating network, temporal transformer |

### 1.3 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PHASE 1: DATA PREPARATION                │
│  • Download MSR-VTT videos (10K videos, 20 captions each) │
│  • Extract uniform 16 frames per video (spans full duration)│
│  • Extract audio per video                                  │
│  • Encode with CLIP/CLAP → Precomputed embeddings          │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 PHASE 2: BASELINE RETRIEVAL                 │
│  • sim_v: query . video_embeddings (CLIP)                 │
│  • sim_t: query . caption_embeddings MAX (CLIP)              │
│  • sim_a: query . audio_embeddings (CLAP)                   │
│  • Equal fusion (0.33 each) → baseline R@1=0.76            │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│               PHASE 3: GATING NETWORK                        │
│  • Input: 512-dim text embedding (query)                    │
│  • Architecture: MLP(512→128→ReLU→3) with softmax          │
│  • Training: Ranking loss with hard negatives               │
│  • Output: [w_v, w_t, w_a] weights that sum to 1         │
│  • Best R@1: 0.60 (matches visual-only baseline)         │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│          PHASE 4: TEMPORAL TRANSFORMER (IN PROGRESS)         │
│  • Problem: Mean-pooled frames lose temporal information       │
│  • Solution: 2-layer transformer over 16 CLIP frame embeddings│
│  • Architecture: CLS token + learnable positional embeddings│
│  • Training: InfoNCE loss, temperature=0.07                 │
│  • Expected: Better temporal modeling → improved R@1         │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.4 Key Implementation Details

| Detail | Value/Config |
|--------|------------|
| Dataset | MSR-VTT (10K videos, 20 captions/video) |
| Train Split | ~6K videos |
| Test Split | ~4K videos |
| Video Embedding | 512-dim (CLIP ViT-B/32) |
| Audio Embedding | 512-dim (CLAP) |
| Caption Embedding | 512-dim × 20 captions/video |
| Gating Hidden | 128-dim |
| Gating Output | 3-way softmax |
| Transformer | 2 layers, 4 heads, 512-dim |
| Frame Count | 16 per video (uniform) |
| Temperature | 0.07 (InfoNCE) |
| Training Epochs | 8 |
| Learning Rate | 1e-4 |
| Batch Size | 32 |

### 1.5 Pipeline Flow

```
QUERY TEXT
    │
    ▼
┌──────────────────┐
│ CLIP encode text │ ──→ 512-dim query embedding
└──────────────────┘
    │
    ├──────────────────┐
    │                  ▼
┌──────────────┐    ┌──────────────────┐
│ Gating      │    │ Compute        │
│ Network     │    │ similarities   │
│ [w_v,w_t,w_a]│    │ sim_v,sim_t,sim_a
└──────────────┘    └──────────────────┘
    │                  │
    └────────┬─────────┘
             ▼
      ┌──────────────────┐
      │ Weighted Fusion │
      │ w_v*sim_v +   │
      │ w_t*sim_t +   │
      │ w_a*sim_a    │
      └──────────────────┘
             │
             ▼
      ┌──────────────────┐
      │ Ranking       │
      │ Top-K        │
      └──────────────────┘
```

---

## 2. Data Preparation Pipeline

### 2.1 Video Frame Extraction
```
FUNCTION extract_uniform_frames(video_path, num_frames=16):
    duration = get_video_duration(video_path)
    FOR i FROM 0 TO num_frames-1:
        timestamp = (i + 0.5) * duration / num_frames
        ffmpeg_extract_frame(video_path, timestamp, output_file)
    RETURN frame_count == num_frames
```

### 2.2 Audio Extraction
```
FUNCTION extract_audio(video_path, output_path):
    ffmpeg_extract_audio(video_path, output_path)
    RETURN audio_file_exists
```

---

## 3. Embedding Encoding Pipeline

### 3.1 CLIP Video Encoding
```
FUNCTION encode_videos_clip(frames_batch):
    # frames_batch: [B, T, 3, H, W]
    images = frames_batch.flatten(0, 1)          # [B*T, 3, H, W]
    frame_embeds = clip_model.encode_image(images)
    frame_embeds = frame_embeds.view(B, T, -1)   # [B, T, 512]
    video_embed = frame_embeds.mean(dim=1)      # Mean pooling
    video_embed = normalize(video_embed)         # L2 normalize
    RETURN video_embed
```

### 3.2 CLIP Text Encoding
```
FUNCTION encode_text_clip(texts):
    tokens = clip.tokenize(texts).to(device)
    text_embeds = clip_model.encode_text(tokens)
    text_embeds = normalize(text_embeds)
    RETURN text_embeds
```

### 3.3 CLAP Audio Encoding
```
FUNCTION encode_audio_clap(audio_path):
    audio, sr = librosa.load(audio_path, sr=48000)
    audio = audio.astype("float32").reshape(1, -1)
    audio_embed = clap_model.get_audio_embedding_from_data(audio)
    RETURN audio_embed  # [512]
```

---

## 4. Similarity Computation

### 2-Way Similarity (Query vs Video)
```
FUNCTION compute_similarity_2way(query_embeds, video_embeds):
    # query_embeds: [N, 512], video_embeds: [M, 512]
    sim_matrix = query_embeds @ video_embeds.T  # [N, M]
    RETURN sim_matrix
```

### 3-Way Similarity (Video + Caption + Audio)
```
FUNCTION compute_similarity_3way(query_embed, video_embed, caption_embeds, audio_embed):
    sim_v = query_embed @ video_embed.T                    # [1]
    sim_t = (query_embed @ caption_embeds.T).max()         # MAX over captions
    sim_a = query_embed @ audio_embed.T                    # [1]
    RETURN sim_v, sim_t, sim_a
```

---

## 5. Gating Network

### 5.1 Architecture
```
CLASS GatingNetwork(nn.Module):
    FUNCTION __init__():
        self.fc1 = Linear(512, 128)    # text_embed → hidden
        self.fc2 = Linear(128, 3)        # hidden → [w_v, w_t, w_a]
    
    FUNCTION forward(text_embed):
        h = ReLU(self.fc1(text_embed))
        logits = self.fc2(h)
        weights = softmax(logits, dim=-1)  # [w_v, w_t, w_a], sum=1
        RETURN weights
```

### 5.2 Training (Ranking Loss)
```
FUNCTION train_gating_ranking(model, train_data, video_db, audio_db, caption_db):
    FOR epoch IN range(NUM_EPOCHS):
        FOR batch IN train_batches:
            query_embeds = encode_text_clip(batch.captions)
            
            # Compute similarities for all candidates
            FOR video IN batch.positive_videos:
                sim_v = query @ video_db[video].T
                sim_t = (query @ caption_db[video].T).max()
                sim_a = query @ audio_db[video].T
                
                weights = model(query)
                score = weights[0]*sim_v + weights[1]*sim_t + weights[2]*sim_a
                positive_scores.append(score)
            
            # Hard negative mining
            FOR video IN hard_negatives:
                score = weighted_fusion(...)
                negative_scores.append(score)
            
            # Ranking loss
            loss = ranking_loss(positive_scores, negative_scores)
            loss.backward()
            optimizer.step()
```

### 5.3 Inference
```
FUNCTION retrieve_with_gating(query_text, model, video_db, ...):
    query_embed = encode_text_clip([query_text])
    weights = model(query_embed)  # [w_v, w_t, w_a]
    
    FOR video IN database:
        sim_v = query @ video_db[video].T
        sim_t = max(query @ caption_db[video].T)
        sim_a = query @ audio_db[video].T
        
        score = weights[0]*sim_v + weights[1]*sim_t + weights[2]*sim_a
        rankings.append((video, score))
    
    RETURN top_k(rankings, k=10)
```

---

## 6. Temporal Transformer

### 6.1 Architecture
```
CLASS TemporalTransformer(nn.Module):
    FUNCTION __init__(num_frames=16, hidden_dim=512):
        self.cls_token = Parameter(random.randn(1, 1, hidden_dim) * 0.02)
        self.pos_embedding = Parameter(random.randn(1, num_frames+1, hidden_dim) * 0.02)
        self.transformer = TransformerEncoder(
            d_model=hidden_dim,
            nhead=4,
            num_layers=2,
            dim_feedforward=2048
        )
        self.projection = Linear(hidden_dim, hidden_dim)
    
    FUNCTION forward(frame_embeds):
        # frame_embeds: [B, T, 512]
        batch_size = frame_embeds.size(0)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = concatenate([cls_tokens, frame_embeds], dim=1)  # [B, T+1, 512]
        x = x + self.pos_embedding                          # Add positional
        x = self.transformer(x)                             # [B, T+1, 512]
        cls_output = x[:, 0]                                # Extract CLS
        output = normalize(self.projection(cls_output))     # [B, 512]
        RETURN output
```

### 6.2 Training (InfoNCE)
```
FUNCTION train_transformer(model, train_data, clip_model):
    FOR epoch IN range(NUM_EPOCHS):
        FOR batch IN train_batches:
            frames = load_uniform_frames(batch.video_ids)     # [B, 16, 3, 224, 224]
            captions = batch.captions
            
            # Encode with frozen CLIP
            frame_embeds = clip_model.encode_image(frames.flatten(0,1))
            frame_embeds = frame_embeds.view(-1, 16, 512)
            frame_embeds = normalize(frame_embeds)
            
            text_embeds = clip_model.encode_text(tokenize(captions))
            text_embeds = normalize(text_embeds)
            
            # Forward through transformer
            video_embeds = model(frame_embeds.float())
            
            # InfoNCE loss
            logits = video_embeds @ text_embeds.T / temperature
            labels = arange(len(logits))
            loss = (cross_entropy(logits, labels) + cross_entropy(logits.T, labels)) / 2
            
            loss.backward()
            optimizer.step()
```

---

## 7. Retrieval Evaluation

### 7.1 Recall@K
```
FUNCTION compute_recall(sim_matrix, k):
    # sim_matrix: [N_queries, N_videos]
    correct = 0
    FOR i IN range(N_queries):
        top_k_indices = argsort(sim_matrix[i], descending=True)[:k]
        IF correct_video_index[i] IN top_k_indices:
            correct += 1
    RETURN correct / N_queries
```

### 7.2 Block-wise Computation (Memory Efficient)
```
FUNCTION retrieve_blockwise(query_embeds, video_embeds, query_batch=32, video_batch=100):
    results = []
    FOR q_start IN range(0, num_queries, query_batch):
        query_batch = query_embeds[q_start:q_start+query_batch]
        FOR v_start IN range(0, num_videos, video_batch):
            video_batch = video_embeds[v_start:v_start+video_batch]
            sim = query_batch @ video_batch.T
            process(sim)
            del sim; gc.collect()
```

---

## 8. Complete Retrieval Pipeline

```
FUNCTION full_retrieval(query_text, video_db, audio_db, caption_db, gating_model):
    # 1. Encode query
    query_embed = encode_text_clip([query_text])
    
    # 2. Get gating weights
    weights = gating_model(query_embed)  # [w_v, w_t, w_a]
    
    # 3. Compute scores for all videos
    scores = {}
    FOR video_id IN video_db.keys():
        sim_v = query_embed @ video_db[video_id].T
        sim_t = max(query_embed @ caption_db[video_id].T)
        sim_a = query_embed @ audio_db[video_id].T
        
        score = weights[0]*sim_v + weights[1]*sim_t + weights[2]*sim_a
        scores[video_id] = score
    
    # 4. Rank and return top-K
    RETURN top_k(scores, k=10)
```
