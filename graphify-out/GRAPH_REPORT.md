# Graph Report - team23  (2026-09-23)

## Corpus Check
- 88 files · ~112,748 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1060 nodes · 2458 edges · 94 communities (53 shown, 41 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 46 edges (avg confidence: 0.9)
- Token cost: 0 input · 165,726 output

## Community Hubs (Navigation)
- Demo & Behavioural Verification
- CLIP Frame/Text Encoding + InfoNCE Loss
- Gating Ablation Phase 5 Deep Runs
- WavLM Audio Feature Extraction
- Gating Ablation Study Pipeline
- Manifest & Frame/Audio Extraction
- Audio Waveform Dataset & Collation
- ImageBind Data Loading (Audio/Video/Text)
- WavLM->CLIP Audio Adapter Training
- Angular Similarity & Gating Forward Pass
- ImageBind Transformer Backbone
- ImageBind CLIP Tokenizer Utilities
- Text Target Augmentation & BEATs Features
- M4 Adapter Audio Alignment Evaluation
- AEMS CLI (train/eval/export)
- CCA/Whitening Cross-Modal Alignment
- Multi-Seed Crop Evaluation
- Transformer Embedding Export & ImageBind Features
- ImageBind EinOps & Logit Scaling Layers
- ImageBind IMU/Patch Preprocessing
- ImageBind Modality Heads & Preprocessors
- AEMS Plan: Locked Constraints & Manifest Schema
- M4-Plus Local-Scale Loss Evaluation
- Audio Similarity Improvement Experiments
- Extended Method Comparison (M6/A2T)
- ImageBind Video Transform Augmentations
- LoRA Adaptation for Audio Alignment
- ImageBind Attention Block Utilities
- Core Encoder & Fusion Architecture (CLAP/CLIP/Gating/Transformer)
- Audio Encoder Selection: BEATs vs ImageBind vs WavLM
- SSL Adapter Contrastive Training
- Retrieval Metrics Diagnostics (NDCG/MRR)
- Gating Collapse Negative Result & Architecture Audit
- WavLM Encoder Segment Embedding
- M4 vs M6-Hybrid Alignment Decision
- Embedding Anisotropy Diagnostics
- Demo/Eval Shell Scripts
- ImageBind Model Card & Sample Assets
- AEMS Dataset Class
- ImageBind Video-to-Image Patch Modules
- Momentum Memory Bank for Hard Negatives
- Query Ranking Inspection Utilities
- Z-Scored Fusion & Angular Similarity Rationale
- Gating Heuristic Baselines (Phase 0)
- Procrustes Cross-Modal Alignment Test
- Gating Network Variant A
- Gating Network Variant B
- Procrustes Projection Solver
- Alignment & Uniformity Metrics (Wang & Isola)
- Hubness Diagnostics (Local Scaling)
- Embedding Centroid & Distance Metrics
- AEMS Codebase Structure & README
- Audio Choices: Angular Similarity & Semantic Misalignment
- M4 Fusion Impact Evaluation
- Ranking Loss Alignment Tests
- Embedding Loading Utilities
- Learned Similarity Scaling Test
- Train/Val Overlap Check
- BEATs Pretraining Model
- Package Setup
- Angular vs Cosine Similarity Test
- aems Package Marker
- Category-Stratified Recall Evaluation
- ImageBind Code of Conduct
- ImageBind Contributing Guidelines
- aems Package Marker (dup)
- Legacy Ablation Script Reference
- Legacy Audio FT Utils Reference
- Legacy Compare Utils Reference
- Legacy LoRA Torch Reference
- Legacy M4 Adapter Reference

## God Nodes (most connected - your core abstractions)
1. `set_seeds()` - 77 edges
2. `GatingNetwork` - 35 edges
3. `search()` - 27 edges
4. `ImageBindModel` - 24 edges
5. `load_anchor_pairs()` - 22 edges
6. `load_search_index()` - 19 edges
7. `save_db()` - 16 edges
8. `CLAPEncoder` - 15 edges
9. `evaluate_retrieval()` - 15 edges
10. `main()` - 15 edges

## Surprising Connections (you probably didn't know these)
- `Explainability Module (explain_retrieval.py)` --references--> `explain_gating_decision()`  [EXTRACTED]
  docs/EXPLAINABILITY.md → src/explainability/explain_retrieval.py
- `Explainability Module (explain_retrieval.py)` --references--> `explain_modality_contributions()`  [EXTRACTED]
  docs/EXPLAINABILITY.md → src/explainability/explain_retrieval.py
- `Explainability Module (explain_retrieval.py)` --references--> `explain_ranking_difference()`  [EXTRACTED]
  docs/EXPLAINABILITY.md → src/explainability/explain_retrieval.py
- `Bird Image Sample - ImageBind Example` --conceptually_related_to--> `ImageBind Multimodal Embedding`  [INFERRED]
  experiments/audio_alignment/imagebind_code/.assets/bird_image.jpg → experiments/audio_alignment/imagebind_code/README.md
- `Car Image Sample - ImageBind Example` --conceptually_related_to--> `ImageBind Multimodal Embedding`  [INFERRED]
  experiments/audio_alignment/imagebind_code/.assets/car_image.jpg → experiments/audio_alignment/imagebind_code/README.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Audio-CLIP alignment methods evaluated against M4** — docs_audio_linear_projection_insufficient, docs_audio_method_comparison_m4_winner, docs_audio_beats_swap_ssl_single_decision, docs_audio_m6_wav2clip_hybrid_finding, docs_audio_post_m4_tiers_verdict [INFERRED 0.85]
- **Investigations converging on gating-network single-modality collapse** — docs_diagnostic_report_gate_collapses_single_modality, docs_architecture_audit_gating_collapse_negative_result, docs_project_checkpoint_behavioural_verification, docs_diagnostic_report_per_video_gating_sketch [INFERRED 0.85]
- **Audio near-zero recall root-cause diagnosis leading to WavLM fix** — docs_audio_diagnostics_root_cause, docs_known_bugs_bug4_audio_near_zero, docs_audio_beats_swap_wavlm_adoption, docs_architecture_zscore_fusion [INFERRED 0.85]

## Communities (94 total, 41 thin omitted)

### Community 0 - "Demo & Behavioural Verification"
Cohesion: 0.07
Nodes (64): main(), Demonstration pipeline with full explainability output., main(), Behavioural verification: demonstrate gating adapts per query type., main(), Audio-to-video retrieval with explainability. The clip is encoded with WavLM…, main(), Image-to-video retrieval with explainability. The audio branch lives in CLIP… (+56 more)

### Community 1 - "CLIP Frame/Text Encoding + InfoNCE Loss"
Cohesion: 0.05
Nodes (35): get_video_frames(), encode_clip_frames(), encode_clip_text(), get_video_frames(), info_nce_loss(), no_grad, train_epoch(), validate() (+27 more)

### Community 2 - "Gating Ablation Phase 5 Deep Runs"
Cohesion: 0.06
Nodes (42): datetime, Phase 5: Deep run of the best configs (30 epochs), train_and_eval_deep(), angular_similarity(), get_all_configs(), get_method_configs(), load_data(), loss_approx_ndcg() (+34 more)

### Community 3 - "WavLM Audio Feature Extraction"
Cohesion: 0.06
Nodes (35): Precompute raw WavLM-Large audio features (1024-d, 3 fixed 10 s segments).…, build_wavlm_features.py -- Strong self-supervised audio features (WavLM-Large)…, compare_methods.py -- Orchestrator for AEMS audio-alignment method comparison.…, Fit a linear projection mapping CLAP-audio embeddings into CLIP-text space.…, M3: Shared-text-anchor alignment (ImageBind/Girdhar 2023; Grave 2019).…, m4_contrastive_ssl.py -- Adapter on strong self-supervised (WavLM) audio…, check_audio_embeddings(), check_clap_encoder() (+27 more)

### Community 4 - "Gating Ablation Study Pipeline"
Cohesion: 0.09
Nodes (27): eval_system(), main(), Ablation study: remove each modality, compare equal vs adaptive fusion., Evaluation script for Priority 1 optimizations, main(), Final unified evaluation — all systems, identical conditions., csv, evaluate_simple_fusion() (+19 more)

### Community 5 - "Manifest & Frame/Audio Extraction"
Cohesion: 0.12
Nodes (22): argparse, extract_uniform_frames(), get_video_duration(), bootstrap_ci(), compute_recall_at_k(), Clean up intermediate transformer checkpoints, keeping only best + last., clip, collections (+14 more)

### Community 6 - "Audio Waveform Dataset & Collation"
Cohesion: 0.11
Nodes (29): AudioCollator, AudioWaveformDataset, build_audio_db_from_model(), clap_audio_embedding(), contrastive_audio_loss(), find_audio_path(), load_waveform(), Encode a set of videos' waveforms into an aligned audio DB {vid: tensor}. (+21 more)

### Community 7 - "ImageBind Data Loading (Audio/Video/Text)"
Cohesion: 0.08
Nodes (21): crop_boxes(), get_clip_timepoints(), load_and_transform_audio_data(), load_and_transform_text(), load_and_transform_video_data(), NormalizeVideo, Perform crop on the bounding boxes given the offsets. Args: boxes (ndarray or…, Perform uniform spatial sampling on the images and corresponding boxes. Args:… (+13 more)

### Community 8 - "WavLM->CLIP Audio Adapter Training"
Cohesion: 0.12
Nodes (28): Train the WavLM -> CLIP-text audio adapter and export the audio branch.…, val_mrr(), branches(), fixed_baseline(), fuse(), Train the query-conditioned gating network on z-scored branch similarities. The…, Best fixed weights on validation, as the bar the gate has to clear., fit_adapter() (+20 more)

### Community 9 - "Angular Similarity & Gating Forward Pass"
Cohesion: 0.07
Nodes (15): angular_similarity(), main(), Compute angular similarity between query and video embeddings, create_improved_gating_with_audio(), Create a gating network that can handle audio better, build_caption_index(), main(), Build (video_id, text) -> index_in_video mapping matching precompute order. (+7 more)

### Community 10 - "ImageBind Transformer Backbone"
Cohesion: 0.11
Nodes (14): imagebind_huge(), Attention, BlockWithMasking, Mlp, MultiheadAttention, Tensor, Simple Transformer with the following features 1. Supports masked attention 2.…, Inputs - tokens: data of shape N x L x D (or L x N x D depending on the… (+6 more)

### Community 11 - "ImageBind CLIP Tokenizer Utilities"
Cohesion: 0.10
Nodes (20): dtype, cast_if_src_dtype(), basic_clean(), bytes_to_unicode(), get_pairs(), _get_pos_embedding(), interpolate_pos_encoding(), interpolate_pos_encoding_2d() (+12 more)

### Community 12 - "Text Target Augmentation & BEATs Features"
Cohesion: 0.11
Nodes (17): main(), augment_text_targets.py -- Multi-positive CLIP-text targets for TRAIN split…, main(), build_beats_features.py -- BEATs (iter3+ AS2M) audio features…, 3 fixed 10s segments (+pad) mirroring the cached CLAP recipe., segments_from_wave(), main(), segments_from_wave() (+9 more)

### Community 13 - "M4 Adapter Audio Alignment Evaluation"
Cohesion: 0.16
Nodes (19): evaluate_method(), Produce the full metric block for a projected audio gallery. audio_proj [n,d]…, Save aligned audio as {video_id: tensor} for a list of videos., save_db(), AudioAdapter, contrastive_loss(), main(), pairwise_negatives() (+11 more)

### Community 14 - "AEMS CLI (train/eval/export)"
Cohesion: 0.13
Nodes (21): cli(), evaluate(), export(), list_checkpoints(), # TODO:, Evaluate a trained model., # TODO:, Export a model for inference. (+13 more)

### Community 15 - "CCA/Whitening Cross-Modal Alignment"
Cohesion: 0.16
Nodes (18): apply_whiten(), cca_directions(), collect_text_embeddings(), local_scaling(), procrustes_map(), Return projection Wx (audio->shared) that maximizes correlation. Returns (Wx,…, Hubness reduction: divide each embedding by its k-th-neighbor distance., Load CLIP-text (description) for arbitrary video list (test). (+10 more)

### Community 16 - "Multi-Seed Crop Evaluation"
Cohesion: 0.13
Nodes (19): eval_direction(), query_mat [n,d], gallery_mat [n,d], matched rows. Returns metrics dict., encode_crop(), eval_model_on_features(), load_video_targets(), main(), mean_std(), pick() (+11 more)

### Community 17 - "Transformer Embedding Export & ImageBind Features"
Cohesion: 0.15
Nodes (9): main(), build_imagebind_features.py -- ImageBind (huge) audio features…, waveform2melspec(), win_offsets(), audio_sim_for(), normalize(), fusion_impact_aligned_db.py -- End-to-end fusion impact of any aligned audio…, load_metadata() (+1 more)

### Community 18 - "ImageBind EinOps & Logit Scaling Layers"
Cohesion: 0.12
Nodes (8): einops, EinOpsRearrange, LearnableLogitScaling, Module, Text Pooling used in OpenCLIP, SelectElement, SelectEOSAndProject, instantiate_trunk()

### Community 19 - "ImageBind IMU/Patch Preprocessing"
Cohesion: 0.15
Nodes (7): build_causal_attention_mask(), IMUPreprocessor, PatchEmbedGeneric, Module, no_grad, PatchEmbed from Hydra, TextPreprocessor

### Community 20 - "ImageBind Modality Heads & Preprocessors"
Cohesion: 0.19
Nodes (5): Normalize, ImageBindModel, AudioPreprocessor, RGBDTPreprocessor, ThermalPreprocessor

### Community 21 - "AEMS Plan: Locked Constraints & Manifest Schema"
Cohesion: 0.15
Nodes (17): Three-Segment Audio Extraction Strategy (begin/mid/end), AEMS-plan.md Canonical Implementation Specification, Gating architecture kept identical despite simplified sim_t, Leakage-Free Evaluation Protocol (QA questions vs description+transcript), Locked Project Constraints C-1..C-8, AEMS Manifest Schema (aems_manifest_v1.json), Text Embedding Fusion Strategy (D/T/F variants), Comprehensive System Documentation Report (+9 more)

### Community 22 - "M4-Plus Local-Scale Loss Evaluation"
Cohesion: 0.17
Nodes (16): eval_direct(), eval_sim_matrix(), local_scale(), m4p_loss(), main(), mutual_knn_filter(), query_expansion(), Divide each embedding by its k-th-neighbor distance (cosine). (+8 more)

### Community 23 - "Audio Similarity Improvement Experiments"
Cohesion: 0.12
Nodes (12): test_similarity_improvements(), analyze_top_examples(), compute_embeddings(), main(), Test different similarity computation methods, Select audio-heavy queries from test data, Analyze top examples for a few queries, Select subset of videos for testing (+4 more)

### Community 24 - "Extended Method Comparison (M6/A2T)"
Cohesion: 0.17
Nodes (11): eval_pair(), main(), compare_all_methods.py -- Extended method comparison (adds M6 rows, full a2t…, Both-direction retrieval on matched rows. Returns full dict incl. median rank…, eval_pair(), main(), Both-direction retrieval on matched rows. Returns dict., hubness_skew() (+3 more)

### Community 25 - "ImageBind Video Transform Augmentations"
Cohesion: 0.19
Nodes (7): MixUpAugment, Normalize, RandomResizedCrop, Resize, Scale, ShortSideScale, UniformTemporalSubsample

### Community 26 - "LoRA Adaptation for Audio Alignment"
Cohesion: 0.18
Nodes (8): apply_lora_to_linear(), __init__(), inject_lora(), _recurse(), LoRALayer, LoRA (Hu et al. 2021) adapter modules for PyTorch nn.Linear. LoRA.proj for…, Wrap an existing nn.Linear into an nn.Module that keeps the base frozen and…, Insert LoRA adapters inside the HTSAT audio backend. Returns list of (name) of…

### Community 27 - "ImageBind Attention Block Utilities"
Cohesion: 0.17
Nodes (7): Tensor, QuickGELU, Wrapper around nn.Module that prints registered buffers and parameter names., VerboseNNModule, get_sinusoid_encoding_table(), Sinusoid position encoding table, SpatioTemporalPosEmbeddingHelper

### Community 28 - "Core Encoder & Fusion Architecture (CLAP/CLIP/Gating/Transformer)"
Cohesion: 0.27
Nodes (11): CLAP Audio Encoder, CLIP Encoder (Vision & Text), Gating Network (Adaptive Multimodal Fusion), Temporal Transformer (Frame Aggregation), GatingNetwork MLP Architecture, 6GB GPU Memory Discipline Pattern, AGENTS.md Project Conventions, Temporal Transformer Architecture (+3 more)

### Community 29 - "Audio Encoder Selection: BEATs vs ImageBind vs WavLM"
Cohesion: 0.22
Nodes (10): Encoder Race: BEATs (iter3+ AS2M) vs ImageBind (huge), WavLM-Large chosen as BEATs/ImageBind stand-in, ImageBind rejected as audio encoder, Audio Alignment: BEATs Swap -> WavLM-Large Adoption, Extreme Hubness Finding (91.6% queries return one hub), Audio Embedding Diagnostics Post Phase 3 Review 1, Root Cause: CLAP-audio/CLIP-text coordinate mismatch, Orthogonal Procrustes Projection Insufficient for Retrieval (+2 more)

### Community 30 - "SSL Adapter Contrastive Training"
Cohesion: 0.22
Nodes (7): build_adapter(), quick_a2t_r1(), Multi-positive InfoNCE (SupCon style) over in-batch target rows. pa: [B,512]…, MLP adapter mapping WavLM audio (d_in) -> CLIP-text space (d_out)., SSLAdapter, supcon_multi_loss(), train_variant()

### Community 31 - "Retrieval Metrics Diagnostics (NDCG/MRR)"
Cohesion: 0.29
Nodes (9): ap_at_k(), main(), mean_median_rank(), mrr(), ndcg_at_k(), Comprehensive Retrieval Metrics & Evaluation Protocol Audit (AEMS audio-text)…, Mean Reciprocal Rank., Binary relevance NDCG@K. Correct video = relevance 1, else 0. (+1 more)

### Community 32 - "Gating Collapse Negative Result & Architecture Audit"
Cohesion: 0.25
Nodes (9): Ablation Study Results (docs/ABLATIONS.md), Adaptive gating on visual+caption only (best system), Gating Network Ranking-Loss Collapse (Negative Result), Architecture Audit (Module Completion Status), Gating network collapses to single modality under either training regime, Final Verification Report (9 phases complete), Bug 11: CLAP precompute saved previous video's embedding for short clips, Bug 4: Audio retrieval near-zero R@1 (CLAP root cause) (+1 more)

### Community 33 - "WavLM Encoder Segment Embedding"
Cohesion: 0.28
Nodes (5): fixed_segments(), no_grad, (n, samples) float waveforms at 16 kHz -> (n, 1024) L2-normalized., 1-D 16 kHz waveform -> (1024,) L2-normalized clip embedding., WavLMEncoder

### Community 34 - "M4 vs M6-Hybrid Alignment Decision"
Cohesion: 0.36
Nodes (8): M4 vs M6-hybrid Multi-seed/Crop Decision, Verdict: Keep M4 as deployable over M6-hybrid, M6 Wav2CLIP Target-Swap Experiment, M6-hybrid (video+text targets) beats M6-w2c video-only, M4 Adapter+Contrastive selected as winning alignment method, Post-M4 Tier Campaign (LoRA / Full Fine-tune), Full-finetune early failure diagnostics (LambdaLR bug, stale CLAP head, small batch, SpecAugment), Verdict: M4 remains deployable winner over LoRA/full fine-tune

### Community 35 - "Embedding Anisotropy Diagnostics"
Cohesion: 0.25
Nodes (8): average_cosine(), effective_dimensionality(), main(), pca_variance_explained(), Mean pairwise cosine similarity (anisotropy metric, Ethayarajh 2019). For unit…, Effective dimensionality from eigenvalue distribution of covariance. H(eig) =…, ZCA whitening (mean removal + decorrelation + renormalize)., zca_whiten()

### Community 36 - "Demo/Eval Shell Scripts"
Cohesion: 0.29
Nodes (5): PYTHONPATH, run_demo.sh script, PYTHONPATH, run_final_eval.sh script, venv_bin_activate

### Community 37 - "ImageBind Model Card & Sample Assets"
Cohesion: 0.29
Nodes (7): ImageBind Multimodal Embedding, Bird Image Sample - ImageBind Example, Car Image Sample - ImageBind Example, Dog Image Sample - ImageBind Example, ImageBind Model Card, ImageBind: One Embedding Space To Bind Them All, ImageBind Requirements

### Community 38 - "AEMS Dataset Class"
Cohesion: 0.33
Nodes (3): Dataset, AEMSDataset, filter_by_split()

### Community 39 - "ImageBind Video-to-Image Patch Modules"
Cohesion: 0.29
Nodes (3): Im2Video, PadIm2Video, Convert an image into a trivial video.

### Community 40 - "Momentum Memory Bank for Hard Negatives"
Cohesion: 0.29
Nodes (4): MemoryBank, no_grad, Momentum-updated bank of projected audio embeddings for global hard negative…, Hardest global negatives (excluding self) for the given ids.

### Community 41 - "Query Ranking Inspection Utilities"
Cohesion: 0.33
Nodes (3): main(), show_category(), video_theme()

### Community 42 - "Z-Scored Fusion & Angular Similarity Rationale"
Cohesion: 0.33
Nodes (6): Per-Query Z-Scored Branch Fusion, Bug 10: Branch scores fused on incompatible scales, Angular Similarity Function, Angular similarity chosen over cosine for 3x R@1 gain, Priority 1 Optimizations Implementation Summary, Temperature Scaling and Modality-Specific Scaling Config

### Community 45 - "Gating Network Variant A"
Cohesion: 0.40
Nodes (3): GatingNetwork, main(), to_float()

### Community 46 - "Gating Network Variant B"
Cohesion: 0.40
Nodes (3): GatingNetwork, main(), to_tensor()

### Community 47 - "Procrustes Projection Solver"
Cohesion: 0.40
Nodes (5): main(), procrustes(), Solve min_W ||W X - Y||^2 + lam||W||^2 -> W = (XX^T + lam I)^-1 X Y^T. X: [d,…, Orthogonal Procrustes: W = U V^T from SVD of X Y^T (centered inputs)., ridge_fit()

### Community 48 - "Alignment & Uniformity Metrics (Wang & Isola)"
Cohesion: 0.40
Nodes (5): compute_alignment(), compute_uniformity(), main(), Alignment = E_{(x,y)~p_pos} ||x - y||^2 (Wang & Isola eq. 2) For L2-normalized…, Uniformity = log E_{pairs} exp(-t * ||x - y||^2) (Wang & Isola eq. 4) For…

### Community 49 - "Hubness Diagnostics (Local Scaling)"
Cohesion: 0.40
Nodes (5): local_scaling(), main(), Skewness of a distribution. >0 => heavy right tail (hubness signature)., Local scaling (Zelnik-Manor & Perona 2004): normalize each embedding by its…, skewness()

### Community 50 - "Embedding Centroid & Distance Metrics"
Cohesion: 0.40
Nodes (5): centroid(), dist_metrics(), main(), Mean embedding of a set of L2-normalized embeddings (may not be unit norm)., Euclidean + cosine distance between two (possibly non-unit) vectors.

### Community 51 - "AEMS Codebase Structure & README"
Cohesion: 0.50
Nodes (4): AEMS/FineVideo Retrieval Codebase Structure, WavLM->CLIP Adapter Audio Branch, Decision: Adopt WavLM ssl_single as deployable audio branch, AEMS/FineVideo Multimodal Retrieval System (README)

### Community 52 - "Audio Choices: Angular Similarity & Semantic Misalignment"
Cohesion: 0.67
Nodes (4): Audio Choices and Analysis, Angular Similarity (Audio Improvement), Audio-Text Semantic Misalignment Problem, Temperature Scaling (Similarity Adjustment)

### Community 53 - "M4 Fusion Impact Evaluation"
Cohesion: 0.67
Nodes (3): audio_sim_for(), normalize(), fusion_impact_m4.py -- End-to-end fusion impact of the M4 aligned audio DB.…

### Community 54 - "Ranking Loss Alignment Tests"
Cohesion: 0.83
Nodes (3): compute_ranking_loss_fixed(), compute_ranking_loss_original(), test_ranking_alignment()

## Knowledge Gaps
- **24 isolated node(s):** `ClipInfo`, `PYTHONPATH`, `run_final_eval.sh script`, `aems`, `aems` (+19 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 410 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **41 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Explainability Module (explain_retrieval.py)` connect `Demo & Behavioural Verification` to `Gating Collapse Negative Result & Architecture Audit`, `AEMS Plan: Locked Constraints & Manifest Schema`?**
  _High betweenness centrality (0.092) - this node is a cross-community bridge._
- **Why does `set_seeds()` connect `Text Target Augmentation & BEATs Features` to `Gating Ablation Phase 5 Deep Runs`, `WavLM Audio Feature Extraction`, `Gating Ablation Study Pipeline`, `Manifest & Frame/Audio Extraction`, `Audio Waveform Dataset & Collation`, `WavLM->CLIP Audio Adapter Training`, `Angular Similarity & Gating Forward Pass`, `M4 Adapter Audio Alignment Evaluation`, `CCA/Whitening Cross-Modal Alignment`, `Multi-Seed Crop Evaluation`, `Transformer Embedding Export & ImageBind Features`, `M4-Plus Local-Scale Loss Evaluation`, `Extended Method Comparison (M6/A2T)`, `SSL Adapter Contrastive Training`, `Retrieval Metrics Diagnostics (NDCG/MRR)`, `Embedding Anisotropy Diagnostics`, `Gating Heuristic Baselines (Phase 0)`, `Procrustes Projection Solver`, `Alignment & Uniformity Metrics (Wang & Isola)`, `Hubness Diagnostics (Local Scaling)`, `Embedding Centroid & Distance Metrics`, `M4 Fusion Impact Evaluation`?**
  _High betweenness centrality (0.063) - this node is a cross-community bridge._
- **Why does `Final Verification Report (9 phases complete)` connect `Gating Collapse Negative Result & Architecture Audit` to `Demo & Behavioural Verification`, `Core Encoder & Fusion Architecture (CLAP/CLIP/Gating/Transformer)`, `AEMS Plan: Locked Constraints & Manifest Schema`?**
  _High betweenness centrality (0.052) - this node is a cross-community bridge._
- **What connects `ClipInfo`, `PYTHONPATH`, `run_final_eval.sh script` to the rest of the system?**
  _24 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Demo & Behavioural Verification` be split into smaller, more focused modules?**
  _Cohesion score 0.0694579681921454 - nodes in this community are weakly interconnected._
- **Should `CLIP Frame/Text Encoding + InfoNCE Loss` be split into smaller, more focused modules?**
  _Cohesion score 0.05004389815627744 - nodes in this community are weakly interconnected._
- **Should `Gating Ablation Phase 5 Deep Runs` be split into smaller, more focused modules?**
  _Cohesion score 0.06291591046581972 - nodes in this community are weakly interconnected._