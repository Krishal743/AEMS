# AEMS-plan.md — Canonical Implementation Specification

## Specification Status

| Field | Value |
|---|---|
| **Specification version** | v1.0 |
| **Approval status** | **Fully approved** (all 8 sections approved) |
| **Scope** | Phase A — Port retrieval pipeline to the AEMS corpus, retrain original architecture unchanged, evaluate on a structurally leakage-free benchmark. |
| **Approval date** | 2026-07-20 |
| **Approved by** | All design decisions ratified via iterative section-by-section review |
| **Related documents** | `docs/PROJECT_CHECKPOINT.md` (MSR-VTT prior results), `docs/DIAGNOSTIC_REPORT.md` (gating collapse analysis + Phase B sketches), `docs/ARCHITECTURE.md` (system overview), `docs/AGENTS.md` (project conventions), `docs/PIPELINE.md` (MSR-VTT execution order) |
| **Supersedes** | All other design docs in `docs/` for Phase A implementation. Implementation must follow this document rather than chat history. |

---

## 0. Locked Project Constraints

These decisions were settled during the brainstorming phase and are NOT revisitable in Phase A:

| # | Decision | Rationale |
|---|---|---|
| **C-1** | **Primary goal**: Beat the old system's honest LOO R@1=43.03% (Equal V+C fusion on MSR-VTT) using the new AEMS corpus. | Frames "improve the results" as a measurable target. |
| **C-2** | **Dataset split**: Stratified 85/15 random across all 17 fine categories, seeded with `set_seeds(42)`. | Statistically sound; per-category balance; minor categories still contribute to test. Yields ~5,361 train / ~946 test videos. |
| **C-3** | **Text representation per video** = `text_description` (human-authored one-sentence summary) **fused with** `text_transcript` (full `text_to_speech`). Fusion happens at embedding time, not at manifest time. Raw fields preserved separately in the manifest. | Preserves richness; enables deferred-fusion ablations without re-parsing. |
| **C-4** | **Evaluation queries** = `qa_questions` (5 per test video ≈ 4,730 queries). The Q&A modality is structurally disjoint from `text_description`/`text_transcript`, so exact-text leakage is impossible by construction. | Mirrors the honest LOO regime that exposed 43.03% on MSR-VTT; eliminates Bug 2's root cause by dataset construction. |
| **C-5** | **Architecture unchanged**: 3-modality (Visual, Text, Audio) fusion with adaptive MLP gating `GatingNetwork(512→128→3)`. Temporal Transformer also unchanged (`TemporalTransformer` with 16 frames, 2-layer, 4-head, 512-hidden). | Per the user's mandate: "Retrain all models from scratch using the new dataset. The goal is to evaluate whether the original adaptive query-routing architecture generalizes under a structurally leakage-free benchmark before introducing any architectural modifications." |
| **C-6** | **Only if** the MLP gating again fails to outperform equal fusion do we proceed to Phase B (per-video gating / cross-attention). Phase B is a separate spec. | Empirically rigorous: isolate the dataset variable before introducing architectural variables. |
| **C-7** | **Compute posture**: Pilot on 500-video subset → scale to full 6,307. Mirror the existing project methodology. Sequential ~7–10 days elapsed. | De-risks large-scale compute; respects the 24 GB GPU. |
| **C-8** | **Repo top-level file**: This document, `AEMS-plan.md`, is the single source of truth. Implementation follows this file, not chat messages. Updated incrementally as sections are approved. | Consistency, maintainability, reproducibility. |

---

## 1. Data Layout & Manifest Schema

### 1.1 Directory layout (parallel to MSR-VTT, no overlap)

```
data/
├── raw/
│   ├── msrvtt/                      (existing, untouched)
│   │   ├── annotations/
│   │   └── videos/
│   └── aems/                        (NOT used — direct paths chosen)
└── processed/
    ├── audio/                       (existing msrvtt audio clips)
    ├── metadata/
    │   └── msrvtt_metadata.json     (existing)
    └── aems/                        (NEW)
        ├── frames_uniform/
        │   └── <video_id>/
        │       ├── frame_0000.jpg  ... frame_0015.jpg
        ├── audio/
        │   └── <video_id>.wav
        └── metadata/
            └── aems_manifest_v1.json
```

**Direct paths** (no symlinks): per Q1.1, the manifest stores each video's `video_path` and `json_path` pointing directly into `aems/dataset/<Category>/<id>.{json,mp4}`. There is no `data/raw/aems/` directory.

### 1.2 Manifest schema — `data/processed/aems/metadata/aems_manifest_v1.json`

A single flat JSON array (mirrors `msrvtt_metadata.json`'s flat-array shape), one entry per video, sorted by `(content_fine_category, video_id)`. Each record:

```jsonc
{
  "video_id": "10013",
  "source_dataset": "aems",
  "split": "train",
  "content_fine_category": "Science Explainers",
  "content_parent_category": "Education",
  "duration_seconds": 602,
  "resolution": "640x360",
  "fps": 30.0,
  "original_video_filename": "O-w6CSLiRsA.mp4",
  "original_json_filename": "O-w6CSLiRsA.json",
  "video_path": "aems/dataset/Science Explainers/10013.mp4",
  "json_path":  "aems/dataset/Science Explainers/10013.json",
  "frames_dir": "data/processed/aems/frames_uniform/10013",
  "audio_path": "data/processed/aems/audio/10013.wav",
  "text_description": "A video featuring Bobby who reacts to ...",
  "text_transcript": "save 10 with my code Bobby 10 ...",
  "text_transcript_word_count": 1707,
  "timecoded_text_to_speech": [ /* raw array, unused in Phase A */ ],
  "youtube_title": "Water REACTS to Quran. Wow ...",
  "youtube_tags": ["christian reacts to quran", ...],
  "youtube_description": "#Islam #christianity ...",
  "qa_questions": [
    "What is the discount code mentioned ...",
    ...
  ],
  "qa_answers": [
    "The discount code is 'bobby10'.",
    ...
  ]
}
```

### 1.3 Critical invariants the manifest enforces

1. **Raw text fields preserved separately** (per refinement 1): `text_description`, `text_transcript`, `qa_questions`, `qa_answers`, `timecoded_text_to_speech` are distinct keys. Fusion happens at embedding time, not manifest time.
2. **`qa_questions` are stored but never used as video text.** They live in the manifest for traceability and for the eval script to consume as queries; the embedding precomputation scripts will not read this field. This is the structural guarantee of leakage-free evaluation.
3. **`source_dataset: "aems"`** lets the codebase keep both manifests in memory for cross-dataset sanity checks.
4. **`timecoded_text_to_speech` is included as a raw field** per Q1.3, but is unused in Phase A.
5. **`split` field is stratified per `content_fine_category`**, deterministic via `set_seeds(42)` then per-category `random.Random(42 + category_index).shuffle(category_videos)`. Train ≈ 5,361, test ≈ 946.
6. **`frames_dir` and `audio_path` are populated *before* the embeddings stage runs.** Empty `frames_dir` (extraction failed) → record skipped downstream with a warning, never silently dropped.

### 1.4 What stays unchanged vs. MSR-VTT

The flat-array shape and presence of `{video_id, split, frames_dir}` keys mean `src/data/metadata.py::load_metadata` can consume this manifest with a single new `path` argument — no signature change required. `filter_by_split`, `get_unique_video_ids`, `filter_items_by_videos` work unchanged.

---

## 2. Ingestion Layer & Frame/Audio Extraction

### 2.1 New files

| File | Purpose |
|---|---|
| `bin/data/build_manifest.py` | Walks `aems/dataset/<Cat>/<id>.{json,mp4}`, applies stratified 85/15 split via `set_seeds(42)`, writes `aems_manifest_v1.json` |
| `bin/data/extract_frames.py` | Reads manifest, extracts 16 uniform frames/video via `ffmpeg` → `data/processed/aems/frames_uniform/<id>/frame_XXXX.jpg` |
| `bin/data/extract_audio.py` | Reads manifest, extracts **three** 10-second segments (begin/middle/end) at 48 kHz mono via `moviepy` + `librosa` → `data/processed/aems/audio/<id>.wav` |
| `src/data/aems_dataset.py` | `AEMSDataset(torch.utils.data.Dataset)` — loads raw data ONLY (no CLIP/preprocessing inside) |

### 2.2 Modified files

| File | Change |
|---|---|
| `src/config.py` | Add AEMS path constants (see §2.6) |

No other `src/` files touched in this section.

### 2.3 Manifest builder — `bin/data/build_manifest.py`

**Algorithm:**
```
walk aems/dataset/ for subdirs (the 17 categories)
for each category (with category_index):
    collect all *.json files in that category dir
    for each json file:
        load json
        verify sibling .mp4 exists
        extract fields: video_id (json stem, e.g. "10013"),
                        content_fine_category, content_parent_category,
                        duration_seconds, resolution, fps,
                        text_description (= content_metadata.description),
                        text_transcript (= text_to_speech),
                        text_transcript_word_count,
                        timecoded_text_to_speech (raw, full array),
                        youtube_title, youtube_tags, youtube_description,
                        qa_questions (= [q.question for q in content_metadata.qAndA]),
                        qa_answers   (= [q.answer   for q in content_metadata.qAndA])
    stratified 85/15 split PER CATEGORY
        random.Random(42 + category_index).shuffle(category_videos)
        train_size = int(0.85 * len(category_videos))
        records[:train_size].split = "train"
        records[train_size:].split = "test"
    emit records with split="train" or "test"
write all records (sorted by category, then video_id) to aems_manifest_v1.json
```

**Correctness guarantees:**
- **Per-category stratification** — avoids the failure mode where a global 85/15 shuffle puts all of a large category like Marketing Strategies (887 videos) in train but leaves none for category-balanced test analysis.
- **Seed discipline** — `set_seeds(42)` once at script start, then `random.Random(42 + category_index).shuffle(...)` per category: deterministic and per-category reproducible.
- **Atomic writes** — write to `aems_manifest_v1.json.tmp` then `os.replace()`, never partial.
- **Schema validation** — every record must have non-empty `text_description` and a non-empty `qa_questions` list, else skip with a logged warning and a `failed_videos.txt`.

**Outputs:**
- `data/processed/aems/metadata/aems_manifest_v1.json` (~6,307 entries)
- `data/processed/aems/metadata/_build_log.txt` (skips, warnings, per-category counts)
- Expected split sizes: train ≈ 5,361, test ≈ 946

### 2.4 Frame extraction — `bin/data/extract_frames.py`

**Mirrors** `extract_uniform_frames.py` line-by-line, with these changes only:
- `METADATA_PATH = src.config.AEMS_MANIFEST_PATH`
- `FRAMES_ROOT = src.config.AEMS_FRAMES_DIR`
- `video_path` read from each manifest entry (resolves directly to `aems/dataset/<Cat>/<id>.mp4` — no symlink)
- Resume/idempotency: if `frames_dir` already contains 16 valid `.jpg` files, skip (existing convention)
- Failure list written to `data/processed/aems/_frames_failed.txt`

**Tool**: `ffmpeg` (matches existing MSR-VTT pipeline exactly, memory-safe streaming, no new deps).

**At scale**: 6,307 videos × 16 frames × ~5s timeout = worst case ~14 hours sequential. Realistic average ~3.5 hours. Pilot (500 videos) ≈ 20 minutes. **Frame extraction may run in the background after the pilot succeeds**, per refinement 2.

### 2.5 Audio extraction — `bin/data/extract_audio.py` (three-segment variant per refinement 1)

**Algorithm:**
```
TARGET_SR = 48000
SEGMENT_DURATION = 10  # seconds
NUM_SEGMENTS = 3       # begin, middle, end

for each manifest record:
    audio_path = record["audio_path"]
    load video via moviepy.editor.VideoFileClip(record["video_path"])
    if video.audio is None: log skip, continue
    full_audio, sr = librosa.load audio from video at TARGET_SR, mono=True
    duration_sec = len(full_audio) / TARGET_SR

    if duration_sec <= SEGMENT_DURATION:
        # Short video: encode the available audio once, reuse the same embedding
        # → produces same final embedding as 3× identical forward passes,
        #   but cleaner and slightly more efficient (per refinement 6)
        encode full_audio[0 : SEGMENT_DURATION*SR] once, pad if shorter
        replicate the single resulting CLAP embedding 3 times into the mean
    else:
        # Three 10s segments at begin, middle, end
        seg_offsets = [
            0,                                                   # begin:  [0, 10s]
            int((duration_sec - SEGMENT_DURATION) / 2 * SR),    # middle: centered
            int((duration_sec - SEGMENT_DURATION) * SR)          # end:    [-10s, end]
        ]
        encode each segment independently, L2-normalize each, average, re-normalize
    
    write final audio embedding to disk
```

**Notes:**
- Audio extraction uses the three-segment strategy (begin / middle / end), refined from the original MSR-VTT center-10s approach. AEMS videos are much longer educational content where center-10s alone is unrepresentative.
- For short videos (duration ≤ 10s), encode the available audio once and reuse that embedding rather than performing three identical forward passes. Same final result, slightly cleaner code.
- Skip-if-no-audio: log to `data/processed/aems/_audio_failed.txt`.

**Outputs:**
- `data/processed/aems/audio/<video_id>.wav` (full demuxed audio, 48 kHz mono)
- `data/processed/aems/_audio_failed.txt` — log of skips

### 2.6 Dataset class — `src/data/aems_dataset.py`

The existing `MSRVTTDataset.__getitem__` returns `(images, text, video_id)` — a single text string. To preserve raw text fields separately (per refinement 1), `AEMSDataset` must return all raw text fields.

**New class** `AEMSDataset(torch.utils.data.Dataset)`:
```python
class AEMSDataset(Dataset):
    def __init__(self, manifest_path, split="train", num_frames=16):
        self.metadata = filter_by_split(load_metadata(manifest_path), split=split)
        self.num_frames = num_frames
        # NOTE: NO CLIP loading here per refinement 4.
        # Encoder loading belongs to the embedding-generation scripts.
    
    def __len__(self) -> int: ...
    
    def __getitem__(self, idx) -> dict:
        item = self.metadata[idx]
        frames = sorted(glob(os.path.join(item["frames_dir"], "frame_*.jpg")))
        assert len(frames) == self.num_frames, f"missing frames for {item['video_id']}"
        # NOTE: returns RAW PIL images (or paths), not preprocessed tensors.
        # Preprocessing is the embedding script's responsibility.
        images = [Image.open(f).convert("RGB") for f in frames]
        return {
            "images": images,                    # list[ PIL.Image ]  (raw)
            "video_id": item["video_id"],
            "text_description": item["text_description"],
            "text_transcript": item["text_transcript"],
            "qa_questions": item["qa_questions"],   # list of 5 strings
            "qa_answers":   item["qa_answers"],      # list of 5 strings
            "audio_path": item["audio_path"],
        }
```

**Per refinement 4**: the dataset is **dumb** — it loads raw data only. CLIP model loading, image preprocessing, text preprocessing all belong to the embedding generation scripts (Section 3, 4). This keeps the dataset independent of any particular encoder.

**Per refinement 5**: the `assert len(frames) == num_frames` check is acceptable as-is (matches existing project conventions). Replacing with explicit runtime exceptions is a minor code-quality improvement, not a functional requirement — reserved for a future cleanup.

**Dict-return format** (per refinement 3): the dataset contracts via named fields rather than positional tuples. The richer schema is better represented as a dict.

### 2.7 New `src/config.py` additions

```python
# ===== AEMS paths (per Section 2) =====
AEMS_MANIFEST_PATH      = "data/processed/aems/metadata/aems_manifest_v1.json"
AEMS_FRAMES_DIR         = "data/processed/aems/frames_uniform"
AEMS_AUDIO_DIR          = "data/processed/aems/audio"
AEMS_DATASET_ROOT       = "aems/dataset"

# ===== AEMS audio extraction =====
AEMS_AUDIO_SR            = 48000
AEMS_AUDIO_CLIP_SEC      = 10
AEMS_AUDIO_NUM_SEGMENTS  = 3

# ===== CLIP text token budget (per Section 3 refinement) =====
MAX_CLIP_TEXT_TOKENS     = 77
CHUNK_TOKEN_BUDGET       = MAX_CLIP_TEXT_TOKENS - 5    # = 72; named, not hardcoded
```

### 2.8 What stays dataset-agnostic

After this section, downstream stages see only:
- `AEMSDataset` (or the manifest directly via `load_metadata`)
- The dict shape `__getitem__` returns
- The on-disk `frames_dir` + `audio_path` fields

Sections 3–6 reference `text_description` and `text_transcript` by name, never `content_metadata.description` or `text_to_speech`. **The ingestion layer is the only place in the codebase that knows about AEMS's JSON schema.**

---

## 3. Text Embedding Fusion Strategy

### 3.1 The central problem

CLIP-ViT-B/32's text encoder has a **hard 77-token limit** (verified: `clip.tokenize` raises `RuntimeError` past 77 tokens, no silent truncation). AEMS transcripts range from 85 to 1,770 words (~110 to ~2,300 tokens). The raw `text_to_speech` is **not directly encodable** by CLIP for most videos.

The MSR-VTT pipeline never faced this because captions were 10–15 words each.

### 3.2 Three precomputed fusion variants

Per refinement 1 ("precompute all three text embedding variants"), three variants are written to disk. The downstream evaluation can run ablations across all three without re-precomputing.

| Variant | Source | Algorithm | Output file |
|---|---|---|---|
| **D — Description-only** | `text_description` (always CLIP-safe ≤ 40 words) | Encode directly, L2-normalize | `embeddings/aems_text_embeddings_description_<split>.pt` |
| **T — Transcript-chunk-mean** | `text_transcript` (long, must be chunked) | Split into ≤`CHUNK_TOKEN_BUDGET`-token windows (non-overlapping, consecutive); encode each, L2-normalize; average; re-normalize | `embeddings/aems_text_embeddings_transcript_<split>.pt` |
| **F — Fused (mean-of-means)** | Both | Encode description as 1 vector; encode transcript chunks as N vectors and average into 1 transcript vector; fuse `normalize(0.5·e_desc + 0.5·e_trans)`. | `embeddings/aems_text_embeddings_fused_<split>.pt` |

**Canonical text branch for the gating network**: **F-mean** (per Section 3 decision 1 — precompute all three, decision 2 — F-mean is canonical). The gating network trains against F-mean and evaluates against F-mean. The D and T variants exist for ablation studies only.

### 3.3 Why F-mean over naive average

A naive `(1+N)`-average is bad: description becomes negligible for long videos (a 2,300-token transcript splits into ~32 chunks — description would contribute 1/33 of the final vector). F-mean weights description and transcript equally:

```
e_desc = normalize(CLIP_text(description))                  # [512]
e_chunks = normalize(CLIP_text(transcript_chunks))          # [N, 512]
e_transcript = normalize(e_chunks.mean(dim=0))               # [512]
text_embed_video = normalize(0.5 * e_desc + 0.5 * e_transcript)  # [512]
```

This is a **mean-of-means**: description and transcript contribute equally regardless of how many chunks the transcript splits into. Symmetric, principled, no hyperparameter.

### 3.4 Chunking rule

- **Non-overlapping consecutive windows** (per Section 3 decision 3): simpler, no compute overhead.
- **Token budget**: `CHUNK_TOKEN_BUDGET = MAX_CLIP_TEXT_TOKENS - 5 = 72` tokens (named constant per refinement, NOT hardcoded 72).
- Per-chunk encoding via `clip.tokenize([chunk])` (single string per call to dodge the hard-limit raise).

### 3.5 Fallback behavior (per Section 3 decision 4)

| Condition | Behavior |
|---|---|
| Empty transcript | Use description-only (equivalent to variant D for that video) |
| Empty description | Use transcript-only (equivalent to variant T for that video) |
| Both empty | Skip video from text-DB, log to `_text_failed.txt` |
| ASR-noisy transcript (e.g. bilingual Pronunciation Guides) | **No filter in Phase A** — pass all transcripts through. R@1 differences between D and T variants become the diagnostic finding for Phase B. |

### 3.6 New file

`bin/embeddings/precompute_text_embeddings.py` — reads manifest, encodes text-DB entries, writes one `.pt` per variant per split.

### 3.7 Script interface

```
python bin/embeddings/precompute_text_embeddings.py \
    --split {train,test} \
    --fusion {description,transcript,fused}
```

Each output is `{video_id: Tensor[512]}` keyed dict (one embedding per video, even though internally multiple chunks are averaged). The retrieval math `query_embed @ text_db.T` works unchanged.

**All three variants precomputed for both splits** — total disk ~6,307 videos × 3 variants × 2 splits × 512 × 4B ≈ 145 MB. Trivial.

### 3.8 Embedding-stage architecture (per refinement 4: dataset is dumb)

```python
# bin/embeddings/precompute_text_embeddings.py — sketch
model, _ = clip.load("ViT-B/32", device=DEVICE)            # embedding script owns CLIP load
model.eval()

manifest = load_metadata(AEMS_MANIFEST_PATH)
records = filter_by_split(manifest, split=args.split)

out = {}
for item in tqdm(records):
    vid = item["video_id"]
    if args.fusion == "description":
        e = encode_one(model, item["text_description"])
    elif args.fusion == "transcript":
        e = encode_chunks_and_mean(model, item["text_transcript"])
    elif args.fusion == "fused":
        e_desc = encode_one(model, item["text_description"])
        e_trans = encode_chunks_and_mean(model, item["text_transcript"])
        if e_desc is None:   e = e_trans
        elif e_trans is None: e = e_desc
        else: e = F.normalize(0.5 * e_desc + 0.5 * e_trans, dim=-1)
    if e is not None:
        out[vid] = e.cpu()

torch.save(out, OUT_PATH)
```

### 3.9 New `src/config.py` additions

```python
AEMS_TEXT_EMBEDDINGS_DESC_PATH_TEMPLATE   = "embeddings/aems_text_embeddings_description_{split}.pt"
AEMS_TEXT_EMBEDDINGS_TRANS_PATH_TEMPLATE  = "embeddings/aems_text_embeddings_transcript_{split}.pt"
AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE  = "embeddings/aems_text_embeddings_fused_{split}.pt"
```
(Filled with `.format(split=...)` at use sites.)

### 3.10 What stays unchanged

- The retrieval math `query_embed @ text_db.T` is unchanged — `text_db` is still `{vid: Tensor[512]}`.
- `src/encoders/clip_encode.py::encode_texts` is unchanged (used for query encoding in Section 6).
- The gating network sees a single `sim_t` scalar (text-branch similarity), just as before. The fusion decision is invisible to it.

### 3.11 Why this satisfies "raw fields preserved separately"

The manifest has `text_description` and `text_transcript` as distinct fields. The precompute script reads each independently and emits three variant `.pt` files. Changing the fusion strategy later is a one-line `--fusion` flag re-run, no manifest rebuild, no dataset re-parse.

---

## 4. Visual & Audio Embedding Precomputation

### 4.1 Two new scripts, both mirror existing patterns

| New file | Mirrors | Architecture change |
|---|---|---|
| `bin/embeddings/precompute_video_embeddings.py` | `precompute_video_embeddings.py` | Identical algorithm; only paths swap |
| `bin/embeddings/precompute_audio_embeddings.py` | `precompute_audio_embeddings.py` | Three-segment variant (per §2.5) |

### 4.2 Visual embedding — `precompute_aems_video_embeddings.py`

**Per refinement 1**: Separate AEMS precomputation scripts cleanly isolate AEMS from the existing pipeline while preserving the same output format and downstream retrieval logic.

**Per refinement 2**: Encoder loading inside the embedding scripts themselves (CLIP loaded directly via `clip.load("ViT-B/32")`, CLAP via `CLAPEncoder`). No `src/encoders/` edits.

**Algorithm** (mirrors `precompute_video_embeddings.py` line-by-line):
```
DEVICE = "cuda" if cuda.is_available() else "cpu"
BATCH_SIZE = 64    # configurable via --batch-size (per refinement 5); 24 GB GPU can handle
                   # 16 frames × 64 batch × ~0.5 MB tensor = ~512 MB per batch — safe
load CLIP ViT-B/32
load AEMS_MANIFEST_PATH
iterate manifest records (dedup by video_id; manifest is already 1-record/video):
    for given video_id:
        frame_files = sorted(glob(frames_dir / "frame_*.jpg"))
        if len(frame_files) != NUM_FRAMES: log skip, continue
        for each frame-batch of BATCH_SIZE frames:
            images = stack of PIL → preprocess — CPU
            embeds = model.encode_image(images.to(DEVICE))  # [B, 512]
            embeds = F.normalize(embeds, dim=-1)            # [B, 512]
            collect to CPU
        frame_embeds = cat all batches                       # [16, 512]
        video_embed = frame_embeds.mean(dim=0)               # [512]
        video_embed = F.normalize(video_embed, dim=0)         # [512]
        out[video_id] = video_embed.cpu()
torch.save(out, OUT_PATH)

# Per refinement: processing summary
print(f"Processed: {len(out)}")
print(f"Skipped:    {skipped_count}  ({skip_reasons})")
```

**Per refinement 7**: continue skipping missing/invalid frame data with logging rather than aborting preprocessing.

**Outputs (per refinement 8 — versioned filenames)**:
- `embeddings/aems_video_embeddings_v1.pt` — `{video_id: Tensor[512]}`, ~13 MB
- `data/processed/aems/_video_embed_failed.txt`

**Time estimate**: 6,307 videos × ~30 ms forward ≈ 3 minutes total. Trivial.

### 4.3 Audio embedding — `precompute_aems_audio_embeddings.py` (three-segment variant)

**Per refinement 4 (Section 2)**: Three 10s segments at begin/middle/end, averaging the three normalized CLAP embeddings into one final embedding.

**Algorithm:**
```
DEVICE = "cuda" if cuda.is_available() else "cpu"
TARGET_SR = 48000
SEGMENT_DURATION = 10
NUM_SEGMENTS = 3

encoder = CLAPEncoder(device=DEVICE)
load AEMS_MANIFEST_PATH
iterate manifest records:
    audio_path = record["audio_path"]
    if not exists(audio_path): log skip, continue
    
    # Load full audio once (per refinement)
    audio, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    duration_sec = len(audio) / TARGET_SR
    
    segment_embeds = []
    if duration_sec <= SEGMENT_DURATION:
        # Refinement 6: encode-once-reuse for short videos
        # Cleaner than performing three identical CLAP forward passes
        segment = audio[0 : SEGMENT_DURATION * TARGET_SR]
        if len(segment) < SEGMENT_DURATION * TARGET_SR:
            segment = np.pad(segment, (0, SEGMENT_DURATION * TARGET_SR - len(segment)))
        x = segment.astype("float32").reshape(1, -1)
        with torch.no_grad():
            emb = encoder.model.get_audio_embedding_from_data(x=x)
            emb = F.normalize(emb, dim=-1).squeeze(0).cpu()
        # Replicate the single embedding 3× into the mean (mathematically equivalent to 3× forward)
        audio_embed = emb   # mean of three identical vectors = the vector itself
    else:
        seg_offsets = [
            0,                                                  # begin
            int((duration_sec - SEGMENT_DURATION) / 2 * TARGET_SR),  # middle
            int((duration_sec - SEGMENT_DURATION) * TARGET_SR)       # end
        ]
        for offset_samples in seg_offsets:
            segment = audio[offset_samples : offset_samples + SEGMENT_DURATION * TARGET_SR]
            if len(segment) < SEGMENT_DURATION * TARGET_SR:
                segment = np.pad(segment, (0, SEGMENT_DURATION * TARGET_SR - len(segment)))
            x = segment.astype("float32").reshape(1, -1)
            with torch.no_grad():
                emb = encoder.model.get_audio_embedding_from_data(x=x)
                emb = F.normalize(emb, dim=-1).squeeze(0).cpu()
            segment_embeds.append(emb)
        audio_embed = torch.stack(segment_embeds).mean(dim=0)
        audio_embed = F.normalize(audio_embed, dim=0)
    
    out[record["video_id"]] = audio_embed

torch.save(out, OUT_PATH)

# Per refinement: processing summary
print(f"Processed: {len(out)}")
print(f"Skipped:    {skipped_count}  ({skip_reasons})")
```

### 4.4 New `src/config.py` additions

```python
# Versioned embedding paths (per refinement 8)
AEMS_VID_EMBEDDINGS_PATH        = "embeddings/aems_video_embeddings_v1.pt"
AEMS_AUDIO_EMBEDDINGS_PATH      = "embeddings/aems_audio_embeddings_v1.pt"
AEMS_TEXT_EMBEDDINGS_DESC_PATH_TEMPLATE   = "embeddings/aems_text_embeddings_description_{split}.pt"
AEMS_TEXT_EMBEDDINGS_TRANS_PATH_TEMPLATE  = "embeddings/aems_text_embeddings_transcript_{split}.pt"
AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE  = "embeddings/aems_text_embeddings_fused_{split}.pt"

# Tokenization constants
MAX_CLIP_TEXT_TOKENS     = 77
CHUNK_TOKEN_BUDGET       = MAX_CLIP_TEXT_TOKENS - 5    # 72

# Audio extraction constants
AEMS_AUDIO_SR            = 48000
AEMS_AUDIO_CLIP_SEC      = 10
AEMS_AUDIO_NUM_SEGMENTS  = 3

# Model output paths (Section 5)
AEMS_TRANSFORMER_BEST_PATH      = "models/aems_temporal_transformer_best_v1.pth"
AEMS_TRANSFORMER_CHECKPOINT_DIR = "checkpoints/aems"
AEMS_GATING_WEIGHTS_PATH        = "models/aems_gating_weights_v1.pth"
AEMS_VIDEO_EMBEDDINGS_TRANSFORMER_PATH = "embeddings/aems_video_embeddings_transformer_v1.pt"
```

### 4.5 Why both new scripts load CLIP/CLAP themselves (per refinement 2)

The existing MSR-VTT precompute scripts already load CLIP directly (line 17 of `precompute_video_embeddings.py`). Refinement 4 calls for encoder loading in the embedding scripts, not the dataset class. The current pattern satisfies this — `MSRVTTDataset` is the violator (loads CLIP in `__init__`), and the new `AEMSDataset` (per Section 2's refinement) does NOT load CLIP. **No MSR-VTT cleanup proposed** — this is an AEMS-only change.

### 4.6 Output consistency guarantee

All three AEMS embedding files (`aems_video_embeddings_v1.pt`, `aems_audio_embeddings_v1.pt`, `aems_text_embeddings_<variant>_<split>.pt`) use the same `{video_id: Tensor[512]}` shape with L2-normalized values. Downstream consumers (baselines, gating, transformer, eval) use exactly the same `dict → stack → tensor` idioms the existing code uses.

### 4.7 What stays unchanged

- `src/encoders/clap_encode.py` — used as-is (`CLAPEncoder` class)
- `src/encoders/clip_encode.py` — used as-is for query encoding (Section 6)
- `clip.load("ViT-B/32")` duplicated across files (existing known wart; spec-scope rule says no cleanup in this plan)
- The retrieval math is fully symmetric to MSR-VTT

---

## 5. Transformer + Gating Retraining

### 5.1 Strict implementation order (per refinement)

The training pipeline MUST execute in this order:

```
1. Train the Temporal Transformer
2. Export the final transformer embeddings using the frozen best checkpoint
3. Train the Gating Network using those final embeddings
4. Run evaluation
```

This guarantees the gating network always trains against the same frozen embedding space that will be used during evaluation.

### 5.2 New files

| File | Mirrors | Architecture change |
|---|---|---|
| `bin/training/train_temporal_transformer.py` | `train_temporal_transformer.py` (418 lines) | **None** |
| `bin/training/export_transformer_embeddings.py` | lines 391-416 of `train_temporal_transformer.py` (factored out per refinement) | **None** |
| `bin/training/train_gating_network.py` | `run_query_routing.py` (445 lines) | **None** |

`bin/training/export_transformer_embeddings.py` is a separate script that loads `aems_temporal_transformer_best_v1.pth` and produces `aems_video_embeddings_transformer_v1.pt`. Per refinement 4: "Export the transformer embeddings only after training has completed and the best checkpoint has been selected. The exported embeddings should then be used consistently by downstream components."

### 5.3 The temporal transformer (Step 1)

**Per refinement 1**: Separate AEMS training scripts, preserving the original implementation while allowing a clean AEMS training pipeline.

**Per refinement 2**: Architecture, optimizer, scheduler, losses, and checkpointing unchanged.

**What changes (constants only)**:
- `METADATA_PATH` → `src.config.AEMS_MANIFEST_PATH`
- `FRAMES_DIR` → `src.config.AEMS_FRAMES_DIR`
- `OUTPUT_PATH` → not used in training script; export is a separate step
- Best-checkpoint path → `models/aems_temporal_transformer_best_v1.pth`
- Per-epoch checkpoint dir → `checkpoints/aems/` (new subdirectory, isolated from MSR-VTT's `checkpoints/`)

**What is structurally identical**:
- 16-frame uniform sampling
- CLIP-ViT-B/32 frozen encoder, `[B, 16, 3, 224, 224]` → `encode_clip_frames` → `[B, 16, 512]` → `TemporalTransformer` → `[B, 512]` L2-normalized
- `TemporalTransformer(16 frames, 2-layer, 4-head, 512 hidden, 2048 FFN, dropout 0.1, CLS+pos)` — frozen architecture
- InfoNCE loss (temperature=0.07), AdamW (1e-4, wd=0.01), mixed precision, cosine schedule, gradient clip=1.0
- Validation split within manifest's `"train"` split: 85/15 random at video level, seeded `set_seeds(42)` (mirrors existing pattern)
- Frame cache preloaded into RAM (same pattern as existing)

**Per refinement 3 — Training-text supervision**:

The positive text for InfoNCE training is `item["text_description"]` — the human-authored one-sentence summary. This mirrors the role of MSR-VTT's `item["text"]` (a single clean caption). The fused description+transcript representation (F-mean) remains an **evaluation-time database representation**, not a training target.

Rationale:
- `text_description` is CLIP-token-safe (always ≤ 40 words).
- It is human-authored and high-quality (ranked 5/5 informativeness in the corpus analysis).
- Using fused text at training time would inject transcript ASR noise into contrastive supervision.
- The transformer learns *text-video alignment*; description is the cleanest possible alignable text.

### 5.4 Embedding export (Step 2) — `bin/training/export_transformer_embeddings.py`

**Per refinement 4**: After Step 1 completes and the best checkpoint is selected, run a dedicated export script.

**Algorithm:**
```
Load TemporalTransformer architecture
Load weights from aems_temporal_transformer_best_v1.pth
Set to eval mode

# Run over ALL valid videos in the manifest (train + test splits combined, deduped by video_id)
# This produces the embedding DB used by downstream eval
all_video_embeds = {}
for video_id in tqdm(sorted(unique_video_ids)):
    frames = load_uniform_frames(video_id)  # [16, 3, 224, 224]
    frame_embeds = encode_clip_frames(frames.unsqueeze(0))   # [1, 16, 512]
    video_embed = model(frame_embeds)                       # [1, 512]
    all_video_embeds[video_id] = video_embed.squeeze(0).cpu()
    
    # Periodic save for crash recovery (matches existing pattern)
    if len(all_video_embeds) % 1000 == 0:
        torch.save(all_video_embeds, AEMS_VIDEO_EMBEDDINGS_TRANSFORMER_PATH)

torch.save(all_video_embeds, AEMS_VIDEO_EMBEDDINGS_TRANSFORMER_PATH)
```

**Output**: `embeddings/aems_video_embeddings_transformer_v1.pt` — `{video_id: Tensor[512]}` dict.

This is the transformer's contribution to the eval-time embedding DB. The mean-pool CLIP embedding (`aems_video_embeddings_v1.pt` from Section 4) is the other visual embedding variant. Both stored; experiments can run on either.

### 5.5 The gating network (Step 3) — `bin/training/train_gating_network.py`

**Per refinement 1**: Separate AEMS training script.

**Per refinement 7**: Architecture and loss unchanged. Replacing the 20-caption maximum operation with a single fused text embedding (per Section 3) is a data representation change, not an architectural change.

**Critical training-query decision** — what plays the role of `query` in the gating ranking loss:

The original `run_query_routing.py` trained on MSR-VTT test-split captions, which caused the leakage archived as Bug 2's root cause. **For AEMS Phase A, training and eval queries are Q&A questions** — structurally disjoint from the video's text-DB entry (description+transcript).

| Aspect | Original `run_query_routing.py` | AEMS `train_aems_gating.py` |
|---|---|---|
| `METADATA_PATH` | MSR-VTT manifest | AEMS manifest |
| `video_db_v` | `video_embeddings.pt` | `aems_video_embeddings_v1.pt` OR `aems_video_embeddings_transformer_v1.pt` (per implementation choice) |
| `video_db_a` | `audio_embeddings.pt` | `aems_audio_embeddings_v1.pt` |
| `caption_db` | `caption_embeddings.pt` (20 captions/video) | `aems_text_embeddings_fused_<split>.pt` (1 fused text/video) |
| Training queries | test-split captions (`item["text"]`) | **train-split Q&A questions** (5/video ≈ 26,805 queries) |
| Eval queries | same test-split captions | **test-split Q&A questions** (5/video ≈ 4,730 queries) |
| `sim_t` computation | `max(query · caption_db[vid])` over 20 captions | `query · text_db[vid]` (single fused text embed) — simpler, no max |
| `sim_a` computation | `query_clap · audio_db[vid]` — same | unchanged |
| `MAX_QUERIES=100`, `MAX_VIDEOS=100` caps | small to fit 6GB GPU | **Removed on 24 GB GPU.** Full train-query count (~26,805) and full test-video count (~946) used in a single forward; same block-batch memory discipline pattern (`QUERY_BATCH_SIZE=32`, `VIDEO_BATCH_SIZE=100`) |
| `NUM_TRAIN_QUERIES=300` cap | limited memory | **Removed** (per refinement 5), use all ~26,805 train queries per epoch |
| `NUM_EPOCHS=15` | fixed | **15 initial, but monitor validation performance and allow early stopping (or extend) based on convergence** (per refinement 8) |
| Keyword-overrides block | commented out | stays commented out |
| Output gate weights | `models/gating_weights.pth` | `models/aems_gating_weights_v1.pth` |

**Everything else byte-identical to the original gating script.** The architecture, the ranking loss formula `clamp(0.2 - (correct_score - neg_score), min=0)`, the hard-negative mining (top-10 negatives excluding the correct idx), Adam@1e-3 — all preserved.

**Per refinement 6**: Train the gating network against the canonical fused text database, since this is the same representation used during evaluation. Training and evaluation remain structurally consistent.

#### Why the architecture stays identical even though `sim_t` simplified

The original `sim_t = max(query_embed · caption_tensor[vid])` operates over 20 caption embeddings per video. AEMS text-DB has **1 fused text embedding per video** (Section 3), so `sim_t = query_embed · text_db[vid]` is a single scalar — no max. **This is not an architecture change**: the gate still sees a scalar `sim_t`, `sim_v`, `sim_a`. The internal computation that produced `sim_t` is invisible to the gate. Removing the `max(..., dim=-1).values` call is a data-layout change, not an architectural one.

### 5.6 Evaluation (Step 4)

Covered in Section 6 (Baselines & Evaluation Protocol).

### 5.7 New `src/config.py` additions for Section 5

(Already enumerated in §4.4 — `AEMS_TRANSFORMER_BEST_PATH`, `AEMS_TRANSFORMER_CHECKPOINT_DIR`, `AEMS_GATING_WEIGHTS_PATH`, `AEMS_VIDEO_EMBEDDINGS_TRANSFORMER_PATH`.)

---

## 6. Baselines & Evaluation Protocol (leakage-free)

### 6.1 The leakage-free evaluation protocol (the centerpiece of Phase A)

**Per refinement 1**: Keep the leakage-free evaluation protocol exactly as proposed. Using Q&A questions as queries and description+transcript as the text database is the central methodological improvement of the project and remains the canonical evaluation setup.

| Quantity | Source | Structurally disjoint from? |
|---|---|---|
| Eval queries (~4,730) | `qa_questions` of test-split videos (5/video) | — |
| Video text DB entries | `text_description` + `text_transcript` fused (F-mean) of all videos | Any Q&A question (different annotation field) |

The Q&A questions are human-authored interrogatives that ask about the video's content (e.g., *"What is the discount code mentioned at the beginning of the video?"*); the description/transcript are declarative summaries. They are disjoint text modalities in the source JSON schema and remain disjoint through the manifest and embedding pipeline.

This is the realization of the LOO regime the old system attempted but couldn't structurally enforce (because MSR-VTT's only text was captions). It is also the single most important methodological change in Phase A.

### 6.2 One consolidated evaluation script (per refinement 2)

**Per refinement 2**: Single consolidated `eval_aems_retrieval.py` script. All five systems share query encoding and embedding DBs — 1× encoding pass instead of 5×, and ensures all systems are evaluated under identical conditions.

**New file**: `bin/evaluation/eval_aems_retrieval.py`

**Algorithm:**
```
# Inputs
manifest = load_metadata(AEMS_MANIFEST_PATH)
test_records = filter_by_split(manifest, "test")           # ~946 videos

# Build query set: 5 Q&A per test video → ~4,730 queries
queries = []        # list of strings
query_video_ids = [] # parallel: ground-truth video for each query
query_categories = [] # parallel: content_fine_category for stratification (per refinement 6)
for record in test_records:
    for q in record["qa_questions"]:
        queries.append(q)
        query_video_ids.append(record["video_id"])
        query_categories.append(record["content_fine_category"])
assert len(queries) == len(query_video_ids) == len(query_categories)

# Load all embedding DBs (per Sections 4 & 5)
video_db_v_meanpool    = torch.load(AEMS_VID_EMBEDDINGS_PATH, weights_only=False)
video_db_v_transformer = torch.load(AEMS_VIDEO_EMBEDDINGS_TRANSFORMER_PATH, weights_only=False)
audio_db               = torch.load(AEMS_AUDIO_EMBEDDINGS_PATH, weights_only=False)
text_db_desc           = torch.load(AEMS_TEXT_EMBEDDINGS_DESC_PATH.format(split="test"), weights_only=False)
text_db_trans          = torch.load(AEMS_TEXT_EMBEDDINGS_TRANS_PATH.format(split="test"), weights_only=False)
text_db_fused          = torch.load(AEMS_TEXT_EMBEDDINGS_FUSED_PATH.format(split="test"), weights_only=False)
gate_weights           = torch.load(AEMS_GATING_WEIGHTS_PATH, weights_only=False)  # if exists

# CLI flags select visual variant + text variant
parser.add_argument("--visual-variant", default="meanpool", choices=["meanpool", "transformer"])
parser.add_argument("--text-variant",   default="fused",     choices=["description", "transcript", "fused"])

# Encode queries via CLIP + CLAP
clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
clap_encoder = CLAPEncoder(device=DEVICE)
query_embeds_clip = encode_clip_text(clip_model, queries)            # [N_q, 512]
query_embeds_clap = encode_clap_text(clap_encoder, queries)          # [N_q, 512]

# Common video IDs (per existing three_branch.py pattern)
video_ids = sorted(
    set(video_db_v_meanpool.keys()) & set(video_db_v_transformer.keys())
    & set(audio_db.keys()) & set(text_db_fused.keys())
)
# → expected ~946 test videos

# Stack to [N_v, 512] matrices, L2-normalize (on CPU; matrices are small)
video_matrix_v_meanpool    = stack_and_normalize([video_db_v_meanpool[v]    for v in video_ids])
video_matrix_v_transformer  = stack_and_normalize([video_db_v_transformer[v] for v in video_ids])
audio_matrix                = stack_and_normalize([audio_db[v]                for v in video_ids])
text_matrix_fused           = stack_and_normalize([text_db_fused[v]           for v in video_ids])
text_matrix_desc            = stack_and_normalize([text_db_desc[v]            for v in video_ids])
text_matrix_trans           = stack_and_normalize([text_db_trans[v]           for v in video_ids])

# Compute three similarity matrices (one per modality pair)
sim_v = query_embeds_clip @ video_matrix_v_<variant>.T          # [N_q, N_v]
sim_t = query_embeds_clip @ text_matrix_<variant>.T              # [N_q, N_v]  -- single scalar, no max
sim_a = query_embeds_clap @ audio_matrix.T                       # [N_q, N_v]

# Five evaluation systems
# 1) Visual only       → R@1/R@5/R@10 over sim_v
# 2) Text only          → R@1/R@5/R@10 over sim_t
# 3) Audio only         → R@1/R@5/R@10 over sim_a
# 4) Equal fusion       → R@1/R@5/R@10 over (sim_v + sim_t + sim_a) / 3
# 5) Adaptive MLP gating → load gate weights from aems_gating_weights_v1.pth,
#                          weights = GatingNetwork(query_embed_clip)
#                          gated_score = w_v·sim_v + w_t·sim_t + w_a·sim_a
#                          R@1/R@5/R@10

# Per refinement 6: category-stratified R@1
for category in sorted(set(query_categories)):
    cat_mask = [i for i, c in enumerate(query_categories) if c == category]
    cat_sim_v = sim_v[cat_mask]
    cat_sim_t = sim_t[cat_mask]
    ... (compute per-category R@1 for all five systems)

# Per refinement (optional scientific enhancement): bootstrap CIs
#   for each system, for each of R@1/R@5/R@10:
#     bootstrap_resample indices [0, N_q) with replacement B=1000 times,
#     compute metric on resample, take 2.5/97.5 percentiles → 95% CI
#   ↓ enables statistically meaningful comparisons between systems

# Per refinement (auto summary table): emit a clean text/markdown table
#   with R@1/R@5/R@10 for all five systems + canonical config + checkpoint IDs
#   to outputs/aems/summary_table_v1.md alongside the JSON output

# Outputs
results = {
    "dataset": "aems",
    "n_queries": len(queries),
    "n_videos": len(video_ids),
    "visual_variant": args.visual_variant,
    "text_variant": args.text_variant,
    "checkpoint_ids": {
        "visual_meanpool":   "aems_video_embeddings_v1.pt",
        "visual_transformer": "aems_video_embeddings_transformer_v1.pt",
        "audio":              "aems_audio_embeddings_v1.pt",
        "text_fused":         "aems_text_embeddings_fused_test.pt",
        "gating_weights":     "aems_gating_weights_v1.pth",
    },
    "systems": {
        "visual_only":      {"R@1": ..., "R@5": ..., "R@10": ...,
                             "R@1_CI95": [low, high], "R@5_CI95": [...], "R@10_CI95": [...]},
        "text_only":        {...},
        "audio_only":       {...},
        "equal_fusion":     {...},
        "adaptive_gating":  {"R@1": ..., "R@5": ..., "R@10": ...,
                             "w_v_mean": ..., "w_t_mean": ..., "w_a_mean": ...,
                             "w_v_std":  ..., "w_t_std":  ..., "w_a_std":  ...},
    },
    "category_stratified": {
        "<category_name>": {
            "n_queries": ..., "n_videos": ...,
            "visual_only_R@1": ..., "text_only_R@1": ..., ...
        },
        ...
    },
}
write results to outputs/aems/eval_results_<visual_variant>_<text_variant>_v1.json
write summary_table_v1.md with formatted comparison
print formatted table to stdout
```

### 6.3 The five evaluation systems (per refinement 3)

**Per refinement 3**: Keep the five-system comparison suite. Clear progression for evaluating the contribution of each modality and the benefit of adaptive fusion.

| # | System | Sim computation | Notes |
|---|---|---|---|
| 1 | Visual only | `sim_v` (query CLIP-text · video CLIP-visual) | Two variants: meanpool, transformer |
| 2 | Text only | `sim_t` (query CLIP-text · text-DB F-mean) | Three ablation variants: D, T, F |
| 3 | Audio only | `sim_a` (query CLAP-text · audio CLAP) | Per DIAGNOSTIC_REPORT, expected near-zero (was 1.02% on MSR-VTT) |
| 4 | Equal fusion | `(sim_v + sim_t + sim_a) / 3` | The "strong baseline" that beat adaptive gating on MSR-VTT honest LOO (43.03%) |
| 5 | Adaptive MLP gating | `w_v·sim_v + w_t·sim_t + w_a·sim_a` where `w = GatingNetwork(query_embed_clip)` | The Phase A object of investigation |

### 6.4 Canonical configuration (per refinements 4 & 5)

- **Canonical visual variant**: `meanpool` (per refinement 4) — consistent with PROJECT_CHECKPOINT findings that meanpool outperformed transformer on MSR-VTT, while still allowing AEMS to test whether the transformer benefits from the stronger dataset.
- **Canonical text variant**: `fused` (per refinement 5) — description-only and transcript-only evaluated as ablations.

### 6.5 Ablation matrix (CLI flag combinations)

After the canonical run (meanpool + fused), the script is re-invoked for ablations:

| Combination | `--visual-variant` | `--text-variant` | Why |
|---|---|---|---|
| Canonical | meanpool | fused | The canonical Phase A run |
| Visual alt | transformer | fused | Tests whether transformer embeddings beat meanpool on AEMS |
| Text D | meanpool | description | Text-branch ablation: description-only |
| Text T | meanpool | transcript | Text-branch ablation: transcript-only |
| Text F (canonical repeat) | meanpool | fused | Sanity check on F dominance; ⊆ canonical |

Total: 4 invocations of the same script, ~minutes each (after embeddings are precomputed). No retraining required.

### 6.6 Category-stratified Recall (per refinement 6)

**Per refinement 6**: Report category-stratified R@1 (or R@5 if per-category query count is too small for R@1 stability) in addition to overall metrics. AEMS was designed with meaningful content categories, and evaluating them separately will strengthen the analysis.

**Per-category reporting**:
- For each of the 17 `content_fine_category` values, filter queries to that category.
- Compute R@1 (and R@5/R@10) for all five systems on the filtered subset.
- If a category has fewer than ~50 test queries (Pronunciation Guides ≈ 44 × 5 = 220 — actually plenty), R@5 is computed in addition to R@1 for stability. With 5 questions × ~44 videos = ~220 queries even for the smallest category, R@1 is statistically meaningful — but R@5 is reported alongside to catch any category where R@1 fails but R@5 succeeds (diagnostic for "near misses").
- Categories with very low per-category R@1 are diagnostic: `Pronunciation Guides` (bilingual ASR noise) may show text-branch weakness; `TED Talks` (long-form structured content) may show text-branch strength.

### 6.7 Comparison anchor (per refinement 7)

**Per refinement 7**: Keep the comparison to the honest MSR-VTT baseline (Equal Fusion R@1 = 43.03%), but present it as a comparison point rather than a mandatory success threshold. Since AEMS is a different and potentially more challenging dataset, overall Recall values may not be directly comparable in absolute terms.

| System (MSR-VTT, honest LOO) | R@1 |
|---|---|
| Visual only | 21.65% |
| Caption only | 39.72% |
| Audio only | 1.02% |
| Equal V+C fusion | **43.03%** ← comparison point (not a threshold) |
| Adaptive (standard-trained) | 39.83% |
| AdaGate-LOO | 24.96% |

### 6.8 Automatic summary table (per refinement)

**Per refinement**: Alongside the JSON outputs, automatically generate a concise summary table of the final results (all five systems, R@1/R@5/R@10, canonical configuration, and checkpoint identifiers). This will make experiments much easier to compare and reproduce.

**Output**: `outputs/aems/summary_table_v1.md` — a markdown table:

```markdown
# AEMS Phase A — Retrieval Results (Canonical Configuration)

**Configuration**: visual=meanpool, text=fused, n_queries=4730, n_videos=946
**Checkpoints**:
- visual:   embeddings/aems_video_embeddings_v1.pt
- audio:    embeddings/aems_audio_embeddings_v1.pt
- text:     embeddings/aems_text_embeddings_fused_test.pt
- gating:   models/aems_gating_weights_v1.pth (if applicable)

| System | R@1 | R@1 95% CI | R@5 | R@5 95% CI | R@10 | R@10 95% CI |
|---|---|---|---|---|---|---|
| Visual only | 0.XXXX | [0.XX, 0.XX] | ... | ... | ... | ... |
| Text only   | 0.XXXX | [...] | ... | ... | ... | ... |
| Audio only  | 0.XXXX | [...] | ... | ... | ... | ... |
| Equal fusion | 0.XXXX | [...] | ... | ... | ... | ... |
| Adaptive gating | 0.XXXX | [...] | ... | ... | ... | ... |

**Gating weight diagnostic** (mean ± std across 4,730 test queries):
| w_v | w_t | w_a |
|---|---|---|
| 0.XXX ± 0.XXX | 0.XXX ± 0.XXX | 0.XXX ± 0.XXX |

## Category-stratified R@1 (all five systems, by content_fine_category)
| Category | n_q | Visual | Text | Audio | Equal | Adaptive |
|---|---|---|---|---|---|---|
| Academic Lectures | NNN | 0.XX | 0.XX | 0.XX | 0.XX | 0.XX |
| Cooking Tutorials | NNN | ... | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... | ... |

## Comparison anchor (MSR-VTT honest LOO, from DIAGNOSTIC_REPORT.md)
| System | MSR-VTT R@1 | AEMS R@1 |
|---|---|---|
| Visual only | 21.65% | 0.XXXX |
| Text/caption only | 39.72% | 0.XXXX |
| Audio only | 1.02% | 0.XXXX |
| Equal fusion | 43.03% | 0.XXXX |
| Adaptive gating | 39.83% | 0.XXXX |
```

### 6.9 Bootstrap confidence intervals (optional scientific enhancement)

**Per refinement (optional)**: If practical, compute bootstrap confidence intervals for R@1/5/10 to help determine whether observed improvements between systems are statistically meaningful.

**Algorithm:**
```python
# For each system, for each of R@1/R@5/R@10:
B = 1000   # bootstrap iterations
N = len(queries)
metric_samples = []
for _ in range(B):
    idx = np.random.randint(0, N, size=N)   # resample with replacement
    sampled_sim = sim_matrix[idx]           # [N, N_v]
    sampled_gt  = [query_video_ids[i] for i in idx]
    m = compute_recall_at_k(sampled_sim, sampled_gt, video_ids, k=K)
    metric_samples.append(m)
ci_low, ci_high = np.percentile(metric_samples, [2.5, 97.5])
```

**Per-system CI computation**: ~5 systems × 3 metrics × 1000 bootstrap iterations × 4,730 queries × 946 videos per bootstrapped retrieval = a few minutes of compute. **Acceptable cost** for the statistical rigor of distinguishing, e.g., adaptive gating R@1=0.28 ± 0.02 vs equal fusion R@1=0.26 ± 0.02 (where the CIs overlap → not a statistically meaningful improvement). The CIs make the gating-vs-equal-fusion comparison scientific rather than anecdotal.

**Seeded** with `set_seeds(42)` for reproducibility.

### 6.10 Output artifacts (per refinement)

| Path | Content |
|---|---|
| `outputs/aems/eval_results_<visual>_<text>_v1.json` | Per-system R@1/R@5/R@10 + 95% CIs + per-query top-K retrievals for error analysis (canonical config + each ablation) |
| `outputs/aems/summary_table_v1.md` | Clean markdown summary table of canonical-config results + category stratification + comparison anchor (auto-generated) |
| `outputs/aems/gating_weights_summary.json` | Mean/std of the 3 gating weights across all test queries (the diagnostic fields for the architectural-collapse check) |
| `outputs/aems/_eval_log.txt` | Run log |

### 6.11 New file inventory for Section 6

| File | Purpose |
|---|---|
| `bin/evaluation/eval_aems_retrieval.py` | Unified five-system evaluation (new). Combines baselines + gating eval + ablation flags + summary table + bootstrap CIs + category stratification into one script. |

### 6.12 What this section deliberately does NOT do

- **No behavioural-verification script** (the old `bin/evaluation/behavioural_test.py`). Per the locked project constraints, behavioural verification is reserved for Phase B if Phase A's adaptive gating fails to beat equal fusion.
- **No ablation_study.py port** (the old script that ran 11-way ablations). The four CLI flag combinations in §6.5 cover the scientifically relevant subset without re-implementing the full 11-way matrix.
- **No `final_eval.py` port** (the old "unified 5-system evaluation" script). `eval_aems_retrieval.py` IS the unified-5-system evaluation for AEMS — there is no need for a separate `final_eval_aems.py`.

### 6.13 What stays unchanged vs. the old pipeline

- `src/evaluation/evaluate_retrieval.py::evaluate_retrieval` — used verbatim
- `src/encoders/clip_encode.py::encode_texts` — used verbatim for query encoding
- `src/encoders/clap_encode.py::CLAPEncoder.encode_text` — used verbatim for audio-query encoding
- `src/models/gating_network.py::GatingNetwork` — used verbatim (reloaded from disk)
- The retrieval math, the L2-normalization conventions, the Recall@K computation — all unchanged

## 7. Pilot-then-Scale Rollout & File Inventory

### 7.1 The pilot phase (500 videos, stratified, ~24 hours)

**Per refinement 1**: Keep the pilot-first workflow. Validating the complete pipeline on a smaller stratified subset before committing to the full dataset is the correct engineering approach.

**Per refinement 2**: Keep the pilot outputs completely separate from the full dataset outputs (`data/processed/aems_pilot/` vs. `data/processed/aems/`) to avoid accidental mixing of artifacts.

**Per refinement 3**: Keep the proposed pilot success criteria. The focus is on verifying that the entire pipeline executes correctly, embeddings are valid, losses decrease, and retrieval functions correctly — not on targeting absolute performance metrics.

The pilot validates the pipeline end-to-end before committing ~10 days of compute to the full corpus.

**Pilot assembly**:
- Pick a stratified subset of 500 videos: ~30 per category × 17 categories.
- Same seed discipline (`set_seeds(42)`)
- Use the same `build_aems_manifest.py` script with a `--pilot` flag that limits the train/test split each category's video count to `min(30, len(cat))` for test samples and `min(270, len(cat))` for train. Total ~500 videos.
- Outputs go to `data/processed/aems_pilot/` (separate from the eventual `data/processed/aems/`).

### 7.2 Pilot execution sequence (sequential, ~24h elapsed)

| Step | Script | Estimated wall-clock | Notes |
|---|---|---|---|
| P-1 | `bin/data/build_manifest.py --pilot` | ~2 min | Writes `aems_pilot/manifest_v1.json` — 500-video stratified subset |
| P-2 | `bin/data/extract_frames.py --manifest aems_pilot/manifest_v1.json` | ~20 min | 500 × 16 frames × ffmpeg. Resume-on-rerun. |
| P-3 | `bin/data/extract_audio.py --manifest aems_pilot/manifest_v1.json` | ~15 min | 500 × 3 segments × moviepy/librosa. |
| P-4 | `bin/embeddings/precompute_video_embeddings.py --manifest aems_pilot/manifest_v1.json` | ~1 min | 500 × CLIP forward. |
| P-5 | `bin/embeddings/precompute_audio_embeddings.py --manifest aems_pilot/manifest_v1.json` | ~1 min | 500 × 3 CLAP forwards. |
| P-6 | `bin/embeddings/precompute_text_embeddings.py --split train --fusion description` (and `transcript`, `fused`) | ~3 min | 3 fusion variants × pilot train split. |
| P-7 | Repeat P-6 with `--split test` | ~2 min | 3 fusion variants × pilot test split. |
| P-8 | `bin/training/train_temporal_transformer.py --manifest aems_pilot/manifest_v1.json --epochs 4` | ~1 h | **Per refinement 7**: Reduce to 4 transformer epochs. Pilot exists to validate correctness and convergence, not maximize performance. |
| P-9 | `bin/training/export_transformer_embeddings.py --manifest aems_pilot/manifest_v1.json` | ~2 min | Export embeddings from best pilot checkpoint. |
| P-10 | `bin/training/train_gating_network.py --manifest aems_pilot/manifest_v1.json --epochs 15` | ~30 min | Train on pilot Q&A queries; verify ranking loss decreases and weights do not collapse instantly. |
| P-11 | `bin/evaluation/eval_aems_retrieval.py --pilot` (canonical config: meanpool + fused) | ~5 min | Report R@1/R@5/R@10 for all five systems. Bootstrap CIs optional in pilot. |

### 7.3 Pilot success criteria (gate before running full corpus)

**Per refinement 3**: keep the proposed pilot success criteria.

1. **No script crashes** on real AEMS data (manifest build, frame/audio extraction, all 3 embedding precomputes, transformer training, export, gating training, evaluation).
2. **Frame extraction rate ≥ 95%** (i.e., ≥475/500 videos yield 16 valid frames). Some AEMS MP4s may be malformed — the failure log file documents why.
3. **Audio extraction rate ≥ 90%** (≥450/500). The three-segment moviepy path is novel and may hit edge cases on bilingual videos with unusual encoding.
4. **Transformer val R@1 ≥ 5%** by epoch 4 on the pilot (just sanity check that the loss decreases and the model is learning; absolute number doesn't matter at this scale).
5. **Gating loss decreases monotonically over 15 epochs** (even on 500 videos, the loss curve should be downward).
6. **Evaluation R@1 > 0 for visual_only, text_only, equal_fusion** (i.e., retrieval is not broken at the eval layer).
7. **No NaN in any embedding file or training log.**

If any criterion fails: triage on the pilot before scaling. If all pass, the full-corpus run is safe to start.

### 7.4 Full-corpus execution sequence (after pilot success, ~7–10 days elapsed)

Per the refinement in Section 2 (Decision 2 from that turn): once the pilot succeeds, **frame extraction may run in the background** while later-stage development continues. Model training only begins after preprocessing has completed.

**Per refinement 4**: Keep the dependency-ordered execution sequence exactly as proposed. The preprocessing → embedding generation → transformer training → embedding export → gating training → evaluation pipeline is clear and reproducible.

**Phase A-1: Preprocessing (background + foreground, ~3–4 days)**

| Step | Script | Wall-clock | Foreground / background |
|---|---|---|---|
| F-1 | `bin/data/build_manifest.py` (no `--pilot`) | ~10 min | Foreground |
| F-2 | `bin/data/extract_frames.py` | ~3.5 h | **Background** — per refinement 2 from Section 2 |
| F-3 | `bin/data/extract_audio.py` | ~12 h | **Background** — three-segment extraction is ~3× slower than the old center-10s |
| F-4 | `bin/embeddings/precompute_video_embeddings.py` | ~3 min | Foreground (after F-2 completes) |
| F-5 | `bin/embeddings/precompute_audio_embeddings.py` | ~8 min | Foreground (after F-3 completes) |
| F-6 | `bin/embeddings/precompute_text_embeddings.py --split train --fusion description` | ~15 min | Foreground (after F-1) — text needs no preprocessed audio/video |
| F-7 | Repeat F-6 for `--fusion transcript`, `--fusion fused` × both splits | ~75 min total | Foreground |
| F-8 | (checkpoint) verify all six embedding files exist with expected key counts | <1 min | Foreground |

**Phase A-2: Training (~6 days)**

| Step | Script | Wall-clock | Notes |
|---|---|---|---|
| T-1 | `bin/training/train_temporal_transformer.py --epochs 12` | ~6 days (12 × ~12h/epoch on 5,361 train videos) | **Per refinement 8**: 12 epochs initial, with validation monitoring and the flexibility to stop early if convergence occurs or extend training if the model is still improving. |
| T-2 | `bin/training/export_transformer_embeddings.py` | ~10 min | Only after T-1 finishes and best checkpoint is selected (per refinement 4 from Section 5) |
| T-3 | `bin/training/train_gating_network.py --epochs 15` | ~2 h | Uses frozen transformer-or-meanpool embeddings; per refinement 8 from Section 5, monitor for convergence |

**Phase A-3: Evaluation (~1 hour)**

| Step | Script | Wall-clock | Notes |
|---|---|---|---|
| E-1 | `bin/evaluation/eval_aems_retrieval.py` (canonical: meanpool + fused) | ~10 min | Including bootstrap CIs |
| E-2 | Repeat with `--visual-variant transformer --text-variant fused` | ~10 min | Visual ablation |
| E-3 | Repeat with `--visual-variant meanpool --text-variant description` | ~5 min | Text ablation D |
| E-4 | Repeat with `--visual-variant meanpool --text-variant transcript` | ~5 min | Text ablation T |
| E-5 | Inspect `outputs/aems/summary_table_v1.md` for the canonical results | <1 min | Auto-generated per Section 6.8 |

**Total elapsed wall-clock: ~7–10 days**, dominated by transformer training.

### 7.5 Dependency graph (canonical Phase A execution)

```
F-1 (manifest) ─┬─► F-2 (frames, bg) ──┐
                 ├─► F-3 (audio, bg)  ─┤
                 └─► F-6, F-7 (text) ──┤
                                      ▼
                                F-8 (verify) ──► T-1 (transformer) ──► T-2 (export)
                                                                                  │
                                                                                  ▼
                                                        T-3 (gating) ◄──────────┘
                                                              │
                                                              ▼
                                                  E-1 (canonical eval)
                                                  E-2, E-3, E-4 (ablations)
```

### 7.6 Orchestration script (per refinement)

**Per refinement**: Add a simple orchestration script that executes the approved pipeline stages in the correct dependency order. The individual scripts remain unchanged; the orchestration script improves reproducibility and reduces the chance of running stages out of order.

**New file**: `scripts/run_aems_pipeline.py`

The orchestration script:
- Is a thin wrapper: it just shells out to the individual scripts via `subprocess.run(...)` in the dependency order from §7.4 and §7.5.
- Has a `--stage` flag (`preprocess`, `embed`, `train_transformer`, `export_transformer`, `train_gating`, `eval`, `all`) so a user can run a subset or the full pipeline.
- Has a `--pilot` flag that propagates to the individual scripts (e.g., calls `build_aems_manifest.py --pilot` and uses `aems_pilot/manifest_v1.json`).
- Checks for input artifacts before running each stage (e.g., refuses to run `train_transformer` unless `aems_video_embeddings_v1.pt` exists). Reports which expected inputs are missing.
- Writes a log line per stage with start/end timestamps and exit codes to `outputs/aems/_pipeline_log.txt`.
- **Does not perform automatic cleanup of intermediate checkpoints** (per refinement 6 — that is a separate manual step).

### 7.7 Complete file inventory — NEW files created by Phase A

**Per refinement 5**: Keep the complete file inventory. Having a single authoritative inventory of all new source files, generated artifacts, embeddings, checkpoints, and outputs will greatly improve maintainability and reproducibility.

This is the exhaustive list of files Phase A introduces. Every file in this list is referenced by the canonical `AEMS-plan.md`.

#### 7.7.1 Source code (new)

| # | Path | Section | Purpose |
|---|---|---|---|
| 1 | `bin/data/build_manifest.py` | §2 | Manifest builder with stratified split + `--pilot` flag |
| 2 | `bin/data/extract_frames.py` | §2 | 16-frame ffmpeg extraction from AEMS MP4s |
| 3 | `bin/data/extract_audio.py` | §2 | Three 10s segments (begin/middle/end) moviepy + librosa |
| 4 | `src/data/aems_dataset.py` | §2 | `AEMSDataset` — dumb raw-data loader, dict returns |
| 5 | `bin/embeddings/precompute_video_embeddings.py` | §4 | CLIP meanpool 16-frame → 512-dim per video |
| 6 | `bin/embeddings/precompute_audio_embeddings.py` | §4 | CLAP 3-segment mean → 512-dim per video |
| 7 | `bin/embeddings/precompute_text_embeddings.py` | §3 | Three variants: `--fusion {description,transcript,fused}` × `--split {train,test}` |
| 8 | `bin/training/train_temporal_transformer.py` | §5 | Temporal transformer training (architecture unchanged) |
| 9 | `bin/training/export_transformer_embeddings.py` | §5 | Export embeddings from frozen best checkpoint |
| 10 | `bin/training/train_gating_network.py` | §5 | Gating network training (architecture unchanged) |
| 11 | `bin/evaluation/eval_aems_retrieval.py` | §6 | Unified five-system eval + ablations + bootstrap CIs + summary table |
| 12 | `scripts/run_aems_pipeline.py` | §7.6 | **Orchestration script** (new per §7.6 refinement) |
| 13 | `bin/training/cleanup_checkpoints.py` | §7.10 | **Manual cleanup script** (new per refinement 6 — deletes intermediate per-epoch checkpoints, keeps only best + final) |

#### 7.7.2 Source code (modified)

| # | Path | Section | Change |
|---|---|---|---|
| 14 | `src/config.py` | §2.7, §3.9, §4.4, §5.7 | New AEMS path constants (`AEMS_MANIFEST_PATH`, `AEMS_FRAMES_DIR`, `MAX_CLIP_TEXT_TOKENS`, `CHUNK_TOKEN_BUDGET`, etc.) — purely additive, no existing constants touched |

No other `src/` files modified. `src/encoders/`, `src/models/`, `src/evaluation/`, `src/data/datasets.py`, `src/data/metadata.py`, `src/routing/`, `src/explainability/` — all untouched.

#### 7.7.3 Data artifacts (generated, gitignored)

| # | Path | Section | Approximate size |
|---|---|---|---|
| D-1 | `data/processed/aems/metadata/aems_manifest_v1.json` | §1 | ~50–80 MB |
| D-2 | `data/processed/aems/frames_uniform/<vid>/frame_*.jpg` | §2.4 | ~2–3 GB (6,307 × 16 jpgs) |
| D-3 | `data/processed/aems/audio/<vid>.wav` | §2.5 | ~2 GB |
| D-4 | `data/processed/aems_pilot/` (entire dir) | §7.1 | ~200 MB (smaller subset) |
| D-5 | `data/processed/aems/_frames_failed.txt`, `_audio_failed.txt`, `_text_failed.txt` | §2, §3 | ~few KB per file |

All in `data/processed/aems*/` — already gitignored via parent `.gitignore`'s `/data/` rule.

#### 7.7.4 Embeddings (generated, gitignored)

| # | Path | Section | Approximate size |
|---|---|---|---|
| E-1 | `embeddings/aems_video_embeddings_v1.pt` | §4 | ~13 MB |
| E-2 | `embeddings/aems_audio_embeddings_v1.pt` | §4 | ~13 MB |
| E-3 | `embeddings/aems_text_embeddings_description_train.pt` | §3 | ~11 MB |
| E-4 | `embeddings/aems_text_embeddings_description_test.pt` | §3 | ~2 MB |
| E-5 | `embeddings/aems_text_embeddings_transcript_train.pt` | §3 | ~11 MB |
| E-6 | `embeddings/aems_text_embeddings_transcript_test.pt` | §3 | ~2 MB |
| E-7 | `embeddings/aems_text_embeddings_fused_train.pt` | §3 | ~11 MB |
| E-8 | `embeddings/aems_text_embeddings_fused_test.pt` | §3 | ~2 MB |
| E-9 | `embeddings/aems_video_embeddings_transformer_v1.pt` | §5.4 | ~13 MB |

All gitignored via parent `.gitignore`'s `*.pt` rule.

#### 7.7.5 Model checkpoints (generated, gitignored)

| # | Path | Section | Approximate size |
|---|---|---|---|
| M-1 | `models/aems_temporal_transformer_best_v1.pth` | §5.3 | ~25 MB |
| M-2 | `checkpoints/aems/temporal_transformer_epoch_<N>.pth` (×12) | §5.3 | ~300 MB total → ~75 MB after cleanup |
| M-3 | `models/aems_gating_weights_v1.pth` | §5.5 | <1 MB |

All gitignored via parent `.gitignore`'s `*.pth`, `/checkpoints/`, `/models/` rules.

#### 7.7.6 Evaluation outputs (generated, gitignored)

| # | Path | Section |
|---|---|---|
| O-1 | `outputs/aems/eval_results_meanpool_fused_v1.json` | §6 |
| O-2 | `outputs/aems/eval_results_transformer_fused_v1.json` | §6 |
| O-3 | `outputs/aems/eval_results_meanpool_description_v1.json` | §6 |
| O-4 | `outputs/aems/eval_results_meanpool_transcript_v1.json` | §6 |
| O-5 | `outputs/aems/summary_table_v1.md` | §6.8 |
| O-6 | `outputs/aems/gating_weights_summary.json` | §6.10 |
| O-7 | `outputs/aems/_eval_log.txt` | §6 |

All gitignored via parent `.gitignore`'s `/outputs/` rule.

### 7.8 Total disk footprint of Phase A

| Component | Size |
|---|---|
| Frames (D-2) | ~3 GB |
| Audio .wav (D-3) | ~2 GB |
| Manifest + processing logs (D-1, D-5) | <100 MB |
| Pilot data (D-4) | ~200 MB |
| All embeddings (E-1 through E-9) | ~80 MB |
| Checkpoints (M-1–M-3 during training) | ~300 MB → ~75 MB after cleanup |
| Eval outputs (O-1 – O-7) | <10 MB |
| **Total order-of-magnitude** | **~6 GB additional** (the source 89 GB AEMS dataset is already on disk under `aems/dataset/` and stays) |

Trivial addition relative to the existing ~18 GB MSR-VTT project.

### 7.9 Total source-line footprint of Phase A

| Component | Estimated lines |
|---|---|
| New scripts (12 new files in §7.7.1) | ~1,500 lines |
| `src/data/aems_dataset.py` | ~80 lines |
| `src/config.py` additions | ~30 lines |
| Orchestration (`scripts/run_aems_pipeline.py`) | ~150 lines |
| Cleanup (`bin/training/cleanup_checkpoints.py`) | ~50 lines |
| **Total new code** | **~1,800 lines** (mirrors/mostly copies existing MSR-VTT scripts) |

### 7.10 Checkpoint cleanup script (per refinement 6)

**Per refinement 6**: Keep checkpoint cleanup as a standalone script rather than performing automatic deletion inside the training process. This is safer and allows manual verification before removing intermediate checkpoints.

**New file**: `bin/training/cleanup_checkpoints.py`

**Algorithm:**
```python
# Manual invocation, post-training, post-successful-eval:
# python bin/training/cleanup_checkpoints.py --keep-best --keep-last
#   (default: keep best + last epoch)

best_path = "models/aems_temporal_transformer_best_v1.pth"
last_epoch_path = "checkpoints/aems/temporal_transformer_epoch_12.pth"
intermediate_paths = glob("checkpoints/aems/temporal_transformer_epoch_*.pth")
                       .exclude(best_path)
                       .exclude(last_epoch_path)

print(f"Best:    {best_path}  (keep)")
print(f"Last:    {last_epoch_path}  (keep)")
for p in intermediate_paths:
    print(f"Delete?  {p}")
confirm = input("Proceed with deletion? [y/N] ")
if confirm.lower() != "y":
    sys.exit("Aborted.")
for p in intermediate_paths:
    os.remove(p)
print(f"Reclaimed: {sum sizes}")
```

**Note**: Confirmation prompt is a deliberate human-in-the-loop check. Per refinement 6, this is safer than automatic deletion during training.

### 7.11 What this section deliberately does NOT do

- **No `requirements.txt` changes** — no new packages are needed beyond what is already pinned (`clip`, `laion_clap`, `librosa`, `soundfile`, `moviepy`, `tqdm`, `torch`, `opencv-python`, `PIL`). Confirmed via Section 4 verification.
- **No new documentation files** beyond `AEMS-plan.md` itself (Phase A docs default to this one spec).
- **No Phase B specs** — per locked constraint C-6, Phase B is a separate spec invoked only if Phase A's adaptive gating fails to beat equal fusion.

## 8. Success Criteria & Decision Rules

### 8.1 Primary success criteria (S-1 through S-5)

**Per refinement 1 (Section 8)**: Keep the S-1 through S-5 success criteria. They appropriately focus on validating that the complete leakage-free pipeline functions correctly. S-3 (Equal Fusion R@1 ≥ 5%) is treated as a sanity check rather than a strict scientific pass/fail threshold, since AEMS may prove substantially more challenging than MSR-VTT.

Phase A is **successful** if and only if **all of the following hold on the canonical AEMS evaluation** (meanpool + fused, ~4,730 Q&A queries, ~946 test videos):

| # | Criterion | Measurement | Sanity check only? |
|---|---|---|---|
| **S-1** | **The pipeline runs end-to-end without crashes** from manifest build → final eval. | Inspection of `_pipeline_log.txt` shows every stage's exit code = 0. | No — hard gate |
| **S-2** | **All five systems report non-degenerate metrics**: R@1, R@5, R@10 are all ≥ 0.0001 for visual_only, text_only, audio_only, equal_fusion, adaptive_gating. | Inspection of `outputs/aems/eval_results_meanpool_fused_v1.json`. | No — hard gate |
| **S-3** | **Equal fusion R@1 ≥ 5%** (i.e., ≥ 0.05). | Section 6.3 system 4. | **Yes — sanity check** (per refinement 1). Treated as a warning if violated, not a hard fail. If S-1, S-2, S-4, S-5 all hold but S-3 fails, Phase A is still "successful" with the observation documented. |
| **S-4** | **Per-category R@1 > 0 for at least 12 of 17 categories** under equal fusion. | Section 6.6 stratification. | No — hard gate |
| **S-5** | **Adaptive gating's R@1 confidence interval overlaps OR exceeds equal fusion's R@1 CI**. | Bootstrap CIs (§6.9). | No — hard gate |

**If S-1 through S-5 (with S-3 as sanity check only) all hold → Phase A is successful.** All Section 6 outputs (`summary_table_v1.md`, individual eval JSONs, gating weights summary) are committed to git via a planned Phase A completion commit.

**If any of S-1, S-2, S-4, S-5 fails** → triage per §8.3.
**If S-3 fails but all others pass** → Phase A still successful; record the S-3 failure as an observation in the analysis.

### 8.2 Phase A outcome framing (contextual, not gating)

**Per refinement 2 (Section 8)**: Keep the comparison to the honest MSR-VTT baseline as contextual analysis rather than a success criterion. AEMS is a more challenging corpus, and absolute R@1 values may not be directly comparable.

| Observation | Interpretation |
|---|---|
| AEMS equal-fusion R@1 ≪ 43.03% | AEMS is meaningfully harder; primarily a function of query-distribution difference (Q&A vs captions) and dataset breadth (17 categories). Not a failure of the pipeline. |
| AEMS equal-fusion R@1 ≈ 43.03% | The systems transfer cleanly — the gating architecture investigation (Phase B) is well-motivated by apples-to-apples comparison. |
| AEMS equal-fusion R@1 ≫ 43.03% | The richer text representation (F-mean: description + full transcript) does the heavy lifting. |
| Adaptive gating R@1 ≈ Equal fusion R@1 | The MLP gating collapse reproduces on AEMS. Phase B is triggered (per §8.4). |
| Adaptive gating R@1 > Equal fusion R@1 (within CIs) | Adaptive gating adds marginal value. Phase B qualified-yes. |

### 8.3 Triage rules (when S-1 through S-5 fail)

**Per refinement 3 (Section 8)**: Keep the proposed triage rules. They map naturally to likely implementation failures and provide clear debugging paths.

| Failure | Most likely cause | Triage step |
|---|---|---|
| S-1 fails (any stage crashes) | Path/config bug or malformed MP4 | Inspect stage stderr, `_pipeline_log.txt`. Fix bug in the failing script. Rerun from failed stage (the orchestration script supports `--stage` resumption). |
| S-2 fails (any system R@K is zero/NaN) | Embedding shape mismatch or DB key set disagreement | Inspect `eval_aems_retrieval.py`'s "common video ID set" computation; verify all embedding files have matching key sets. Likely a skipped-extraction issue (some video has frames but no audio). |
| S-3 fails (equal fusion R@1 < 5%) | **Sanity-check failure**: embeddings are non-informative or queries are ill-structured. | Per-modality R@1 reveals the culprit: visual-only ≈ 0 → CLIP frame extraction/encoding bug; text-only ≈ 0 → text-DB fusion bug; audio-only ≈ 0 → expected (was 1.02% on MSR-VTT). By refinement 1, Phase A is not gated on S-3, but the observation is recorded. |
| S-4 fails (>5 categories unretrievable) | Per-category mods: ASR noise (Pronunciation Guides), too-short videos, frame-extract failures clustering by category | Inspect `_frames_failed.txt`, `_audio_failed.txt` for category clustering. Re-process categories with different ffmpeg/moviepy params if errors cluster. For categories that pass preprocessing but fail retrieval: text-branch quality analysis (§3.5 fallback captures which fragments of the transcript are longform). |
| S-5 fails (adaptive ≪ equal, well outside CI) | Gating collapse (the architectural issue from DIAGNOSTIC_REPORT.md) | Phase A has still succeeded if S-1, S-2, S-4 hold; the gating collapse on the new data *confirms* the issue is structural. Proceed to Phase B (per §8.4). |

### 8.4 Phase B trigger rule (per locked constraint C-6)

> "**Only if** the MLP gating again fails to outperform equal fusion do we proceed to Phase B (per-video gating / cross-attention). Phase B is a separate spec."

**Per refinement 4 (Section 8)**: Keep the Phase B trigger based on whether adaptive gating demonstrates a clear, statistically supported improvement over equal fusion. Avoid a fixed improvement threshold (e.g., +3pp) and instead rely on the statistical evaluation to be meaningful.

**Phase B trigger** (one of these conditions must be true for Phase B **not** to trigger):

- Adaptive gating's R@1 CI lower bound **is above** equal fusion's R@1 CI upper bound at the 95% level (i.e., adaptive gating is statistically significantly better).
- *OR* Adaptive gating's R@1 CI lower bound is above a `PhaseB_min_improvement = 0.03` (3pp absolute) above equal fusion CI upper bound (adaptive gating wins meaningfully).

**Otherwise, the MLP gating has not demonstrated per-query adaptivity beyond equal fusion → Phase B is triggered.** Both architectural alternatives sketched in `docs/DIAGNOSTIC_REPORT.md` "Future Work" (per-video gating and cross-attention fusion) are re-investigated under the AEMS benchmark.

The point of Phase A is not to confirm the gating works — it's to establish the leakage-free benchmark and empirically demonstrate whether the existing architectural limitation persists. Either outcome is informative.

### 8.5 AEMS architecture confirms MSR-VTT diagnosis (Phase A contribution even under gating collapse)

If adaptive gating collapses on AEMS (as it did on MSR-VTT), Phase A has still produced:

1. **A leakage-free evaluation corpus** with stratified splits, deterministic seeds, and versioned artifacts — a permanent contribution to the project.
2. **A per-category diagnostic** — which of the 17 categories are retrievable by which modality (visual, text, audio, fusion) is a novel finding.
3. **A replication of the MSR-VTT gating-collapse finding** on a structurally different dataset → confirms the architectural limitation is dataset-independent.
4. **A clean baseline** (equal fusion) that Phase B can compare against. Without this baseline, Phase B's per-video gating improvement claims would be unverifiable.

### 8.6 Stopping rules for training

**Per refinement 5 (Section 8)**: Keep the proposed stopping rules. An initial transformer budget of 12 epochs with early stopping, optional extension, and a hard cap of 18 epochs is a sensible balance between convergence and computational cost.

**Transformer training**:
- Initial budget: 12 epochs.
- **Early-stop rule**: if `val_R@1` does not improve by ≥ 0.005 (0.5pp absolute) over the best-so-far for 3 consecutive epochs → stop training, take the best checkpoint.
- **Extension rule**: if `val_R@1` is still increasing by ≥ 0.005 per epoch at epoch 12 → extend by up to 6 more epochs (to epoch 18).
- **Hard cap**: 18 epochs (no training past this regardless of improvement).
- **NaN abort**: any NaN in train_loss or val_loss → abort immediately (matches existing pattern in `train_temporal_transformer.py` lines 355–357).

**Gating training**:
- Initial budget: 15 epochs.
- **Early-stop rule**: if `train_loss` does not improve by ≥ 0.001 for 5 consecutive epochs → stop, take the lowest-loss checkpoint.
- **Hard cap**: 30 epochs (extension allowed in increments of 5).
- **Collapse detection rule**: if `w_v_mean < 0.05 AND w_a_mean > 0.80 AND w_t_mean < 0.15` persistently (≥ 5 consecutive epochs) → log "collapsing toward audio-only" and let it complete a few more epochs to confirm; do not abort early (the failure mode is itself diagnostic per §8.3).

### 8.7 Reproducibility guarantees (the spec's safety net)

**Per refinement 6 (Section 8)**: Keep the reproducibility guarantees, versioned artifacts, deterministic seeds, and audit logging exactly as proposed.

These guarantees mean anyone running `scripts/run_aems_pipeline.py --stage all --pilot false` on a machine with the same `venv/` packages gets the same numbers within Monte-Carlo noise from non-deterministic CUDA ops.

| Guarantee | Mechanism |
|---|---|
| Deterministic split | `set_seeds(42)` + per-category `random.Random(42 + category_index)` |
| Deterministic training shuffles | `set_seeds(42)` once at script start; per-epoch `random.shuffle(data)` follows the seeded RNG sequence |
| Deterministic bootstrap CIs | `set_seeds(42)` before the bootstrap loop in `eval_aems_retrieval.py` |
| Versioned filenames | All embedding/checkpoint files have `_v1` suffix; if preprocessing changes, the next version is `_v2` |
| Manifest checksum | `aems_manifest_v1.json` includes a `_build_log.txt` with per-category counts and any skipped videos — the manifest is the ground truth for "which videos are in train/test" |
| Pipeline log | `outputs/aems/_pipeline_log.txt` records stage start/end timestamps, exit codes — full audit trail |

### 8.8 Final deliverables of Phase A

If S-1 through S-5 (§8.1) hold:

1. `AEMS-plan.md` — this document (committed).
2. 13 new source files from §7.7.1 (committed).
3. 1 modified file from §7.7.2 (`src/config.py` additions; committed).
4. Data/processed artifacts (D-1 through D-5): NOT committed (gitignored per parent `.gitignore`); regenerated from the source `aems/dataset/` by running `scripts/run_aems_pipeline.py --stage preprocess`.
5. Embeddings (E-1 through E-9): NOT committed (gitignored); regenerated by `--stage embed`.
6. Model checkpoints (M-1 through M-3): NOT committed (gitignored); regenerated by `--stage train_transformer` and `--stage train_gating`.
7. Evaluation outputs (O-1 through O-7): NOT committed (gitignored); regenerated by `--stage eval`.

**Per refinement 7 (Section 8)**: Summarize the final experimental results in the project documentation (`PROJECT_CHECKPOINT.md` and the `AEMS-plan.md` change log) rather than committing generated outputs to Git.

8. **The summary numbers committed to the project memory** by appending them to this plan's Change Log at completion time.
9. A published update to `docs/PROJECT_CHECKPOINT.md` cross-referencing this plan, marking Phase A complete with whatever adaptive-gating outcome was observed.

### 8.9 What Phase A explicitly leaves for Phase B (only on §8.4 trigger)

| Phase B work item | Source for design |
|---|---|
| Per-video pair-conditioned gating (input = [query, video, sim_v, sim_t, sim_a, \|sim_v−sim_t\|, max(sim_v, sim_t, sim_a)]) | `docs/DIAGNOSTIC_REPORT.md` "Future Work — Per-Video Gating" sketch |
| Cross-attention fusion Q=query_embed, K=[video, text, audio], V=[sim_v, sim_t, sim_a], score = softmax(QKᵀ/√d)·V | `docs/DIAGNOSTIC_REPORT.md` "Cross-Attention Fusion (Alternative)" sketch |
| Add temporal clip-level retrieval using `timecoded_text_to_speech` (currently in manifest but unused per the Q1.3) | This plan §1 — field preserved for exactly this |
| Audio-quality investigation (e.g., specifically-tuned CLAP checkpoints, longer audio segments than 30s) | `docs/DIAGNOSTIC_REPORT.md` "Audio Retrieval Near-Zero R@1" finding |

---

## Change Log

| Date | Change |
|---|---|
| 2026-07-20 | **ALL APPROVED** — Sections 1–8 finalized, approved, and consolidated. Specification version v1.0. Ready for self-review and user review gate. |
| 2026-07-20 | Section 7 approved and appended: pilot-then-scale rollout, orchestration script, manual cleanup script, complete file inventory. Section 8 pending. |
| 2026-07-20 | Section 6 approved and appended: consolidated five-system evaluation with category stratification, auto-generated summary table, and bootstrap CIs. Sections 7–8 pending. |
| 2026-07-20 | Initial creation with Sections 1–5 (approved during brainstorming phase). Sections 6–8 pending. |
