# Pipeline

## Order

```
download → parse_captions → extract_frames → extract_audio
  → precompute video/audio/caption embeddings
  → baselines (CLIP, CLAP, fusion, three-branch)
  → temporal transformer training / gating network training
  → final multimodal evaluation
```

## Step-by-Step

### 1. Download
```bash
python scripts/data/download_msrvtt.py
```
Downloads MSR-VTT from HuggingFace to `data/raw/msrvtt/`.

### 2. Parse captions
```bash
python scripts/data/parse_msrvtt_captions.py
```
Builds `data/processed/metadata/msrvtt_metadata.json` from annotations.

### 3a. Extract frames (fps=1, max=15)
```bash
python scripts/data/extract_frames_msrvtt.py
```
Output: `data/processed/video/frames/`

### 3b. Extract uniform frames (exactly 16)
```bash
python scripts/data/extract_uniform_frames.py
```
Output: `data/processed/video/frames_uniform/`

### 4. Extract audio
```bash
python scripts/data/extract_audio_msrvtt.py
```
Output: `data/processed/audio/`

### 5a. Precompute video embeddings
```bash
python scripts/embeddings/precompute_video_embeddings.py
```
Output: `embeddings/video_embeddings.pt`

### 5b. Precompute audio embeddings
```bash
python scripts/embeddings/precompute_audio_embeddings.py
```
Output: `embeddings/audio_embeddings.pt`

### 5c. Precompute caption embeddings
```bash
python scripts/embeddings/precompute_caption_embeddings.py
```
Output: `embeddings/caption_embeddings.pt`

### 6. Run baselines
```bash
python scripts/baselines/run_clip_baseline.py
python scripts/baselines/run_clap_baseline.py
python scripts/baselines/run_fusion_baseline.py
python scripts/baselines/run_three_branch.py
```

### 7. Train gating network
```bash
python scripts/training/run_query_routing.py
```
Output: `models/gating_weights.pth`

### 8. Train temporal transformer
```bash
python scripts/training/train_temporal_transformer.py
```
Output: `embeddings/video_embeddings_transformer.pt`, `models/temporal_transformer_best.pth`

### 9. Final evaluation
```bash
bash scripts/evaluation/run_final_eval.sh
```
Output: `outputs/eval/`
