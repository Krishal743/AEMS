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
for split in train test; do python bin/embeddings/precompute_text_chunks.py --split $split; done
for split in train test; do
  for fusion in description transcript fused; do
    python bin/embeddings/precompute_text_embeddings.py --split $split --fusion $fusion
  done
done
```
Outputs:
- `embeddings/aems_video_embeddings_v1.pt` — CLIP ViT-B/32, mean-pooled over 16 frames.
- `embeddings/aems_audio_embeddings_wavlm_v1.pt` — raw WavLM-Large features (1024-d), three fixed 10 s segments. These are adapter *inputs*, not the searchable branch.
- `embeddings/aems_text_embeddings_{description,transcript,fused}_{train,test}.pt` — CLIP text, mean-pooled per video.
- `embeddings/aems_text_chunks_{train,test}.pt` — per-passage CLIP embeddings (~30 per video) for the max-sim branch.

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

### 7. train_gating
```bash
python bin/training/train_gating_network.py
```
Output: `models/aems_gating_weights_v1.pth`. Gating is the default fusion mode,
so this is required for the deployed configuration; `--fusion fixed` runs
without it. Training fuses branches the same way search does (per-query
z-scores, then gate weights) and cross-fits the audio branch so the gate is not
trained on adapter-fitted audio. The run warns if the gate collapses onto one
modality or fails to beat tuned fixed weights on validation.

### 8. eval
```bash
python bin/evaluation/eval_aems_retrieval.py --bootstrap
```
Outputs `outputs/aems/eval_results_<visual>_<text>_v1.json` and
`outputs/aems/summary_table_v1.md`, comparing visual-only, text-only,
passage-only, audio-only, equal fusion, fixed fusion, and adaptive gating.

## Fusion

Each branch's similarities are z-scored per query across the gallery, then
combined as `w_v·sim_v + w_t·sim_t + w_c·sim_c + w_a·sim_a` over the visual,
caption, passage and audio branches. Weights come from
`AEMS_FUSION_WEIGHTS` in `src/config.py` by default, or from the gating network
with `--fusion gate`. Re-tune the fixed weights on a validation split whenever
the branches change; never on test.

## Two-stage retrieval

Stage 1 (above) scores every video. `--rerank` adds a stage 2 over the top
`--rerank-top-k` candidates only:

- `gate` — the per-candidate gating network
  (`bin/training/train_per_candidate_gate.py`), which predicts fusion weights
  per query-candidate pair from scores stage 1 already computed. Nearly free,
  worth +0.009 R@1.
- `cross` — a cross-encoder that reads query and passage together
  (`cross-encoder/ms-marco-MiniLM-L6-v2` by default). Worth +0.125 R@1 at
  ~12 ms/query.

Compare them with `python bin/evaluation/eval_rerankers.py`, which tunes depth,
passages per candidate and the blend weight on validation and scores test once.

## Queries and demo

```bash
python bin/queries/query_text.py  --query "your text"
python bin/queries/query_image.py --image  query.jpg
python bin/queries/query_audio.py --audio  clip.wav
python bin/queries/query_video.py --video  clip.mp4
python bin/queries/query_mixed.py --text "your text" --image query.jpg
python bin/demo/demo.py --query "your text" --top-k 5
python bin/evaluation/behavioural_test.py

# optional stage-2 reranking of the shortlist (off by default)
python bin/queries/query_text.py --query "your text" --rerank gate    # ~0.1 ms/query
python bin/queries/query_text.py --query "your text" --rerank cross   # ~12 ms/query
```

Text queries score all four branches with one CLIP vector, because the audio
branch is projected into CLIP text space. Audio-clip queries are encoded with
WavLM and the adapter and reach the audio branch only. Image and video queries
skip the audio branch.
