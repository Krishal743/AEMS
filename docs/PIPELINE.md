# Pipeline

The AEMS pipeline, as implemented by `pipeline/run_aems_pipeline.py`. Run the
whole thing with `python pipeline/run_aems_pipeline.py --stage all`, or one
stage at a time with `--stage <name>`. Every stage below is also runnable on
its own; set `PYTHONPATH=.` first.

## Order

```
preprocess → extract_frames → extract_audio → embed
  → train_audio_adapter → train_transformer → export_transformer
  → train_gating → eval
```

## Stages

### 1. preprocess — build the manifest
```bash
python bin/data/build_manifest.py          # --pilot for a 500-video subset
```
Output: `data/processed/aems/metadata/aems_manifest_v1.json` from `aems/dataset/`.

### 2. extract_frames
```bash
python bin/data/extract_frames.py --resume
```
Output: exactly 16 uniform frames per video in `data/processed/aems/frames_uniform/`.

### 3. extract_audio
```bash
python bin/data/extract_audio.py --resume
```
Output: `data/processed/aems/audio/<video_id>.wav`.

### 4. embed
```bash
python bin/embeddings/precompute_video_embeddings.py
python bin/embeddings/precompute_audio_embeddings.py
for split in train test; do
  for fusion in description transcript fused; do
    python bin/embeddings/precompute_text_embeddings.py --split $split --fusion $fusion
  done
done
```
Outputs:
- `embeddings/aems_video_embeddings_v1.pt` — CLIP ViT-B/32, mean-pooled over 16 frames.
- `embeddings/aems_audio_embeddings_wavlm_v1.pt` — raw WavLM-Large features (1024-d), three fixed 10 s segments. These are adapter *inputs*, not the searchable branch.
- `embeddings/aems_text_embeddings_{description,transcript,fused}_{train,test}.pt` — CLIP text.

### 5. train_audio_adapter
```bash
python bin/training/train_audio_adapter.py
```
Trains the WavLM → CLIP-text adapter with symmetric InfoNCE. Positives are each
train video's description plus its QA questions. The checkpoint is chosen by
text→audio MRR on a validation split carved from the **train** videos, so the
test split is never used for selection.

Outputs:
- `models/aems_audio_adapter_wavlm_v1.pth`
- `embeddings/aems_audio_embeddings_wavlm_clip_v1.pt` — the searchable audio branch, 512-d in CLIP text space.

### 6. train_transformer / export_transformer (optional visual variant)
```bash
python bin/training/train_temporal_transformer.py --epochs 12
python bin/training/export_transformer_embeddings.py
```
Outputs: `models/aems_temporal_transformer_best_v1.pth`,
`embeddings/aems_video_embeddings_transformer_v1.pt`. Used by
`--visual-variant transformer`; the default `meanpool` does not need this.

### 7. train_gating (optional)
```bash
python bin/training/train_gating_network.py --epochs 15
```
Output: `models/aems_gating_weights_v1.pth`. Only needed for `--fusion gate`;
the default fixed-weight fusion does not use it. Training fuses branches the
same way search does: per-query z-scores, then gate weights.

### 8. eval
```bash
python bin/evaluation/eval_aems_retrieval.py --bootstrap
```
Outputs `outputs/aems/eval_results_<visual>_<text>_v1.json` and
`outputs/aems/summary_table_v1.md`, comparing visual-only, text-only,
audio-only, equal fusion, fixed fusion, and adaptive gating.

## Fusion

Each branch's similarities are z-scored per query across the gallery, then
combined as `w_v·sim_v + w_t·sim_t + w_a·sim_a`. Weights come from
`AEMS_FUSION_WEIGHTS` in `src/config.py` by default, or from the gating network
with `--fusion gate`. Re-tune the fixed weights on a validation split whenever
the branches change; never on test.

## Queries and demo

```bash
python bin/queries/query_text.py  --query "your text"
python bin/queries/query_image.py --image  query.jpg
python bin/queries/query_audio.py --audio  clip.wav
python bin/queries/query_video.py --video  clip.mp4
python bin/queries/query_mixed.py --text "your text" --image query.jpg
python bin/demo/demo.py --query "your text" --top-k 5
python bin/evaluation/behavioural_test.py
```

Text queries score all three branches with one CLIP vector, because the audio
branch is projected into CLIP text space. Audio-clip queries are encoded with
WavLM and the adapter and reach the audio branch only. Image and video queries
skip the audio branch.
