# Graph Report - team23  (2026-09-20)

## Corpus Check
- 171 files · ~110,546 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 8 file(s) not represented in the graph (top: (none) 5, .lock 1, .gz 1)

## Summary
- 974 nodes · 2365 edges · 99 communities (55 shown, 44 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 109 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Community 0
- Community 1
- Community 2
- Community 3
- Community 4
- Community 5
- Community 6
- Community 7
- Community 8
- Community 9
- Community 10
- Community 11
- Community 12
- Community 13
- Community 14
- Community 15
- Community 16
- Community 17
- Community 18
- Community 19
- Community 20
- Community 21
- Community 22
- Community 23
- Community 24
- Community 25
- Community 26
- Community 27
- Community 28
- Community 29
- Community 30
- Community 31
- Community 32
- Community 33
- Community 34
- Community 35
- Community 36
- Community 37
- Community 38
- Community 39
- Community 40
- Community 41
- Community 42
- Community 43
- Community 44
- Community 45
- Community 46
- Community 47
- Community 48
- Community 49
- Community 50
- Community 51
- Community 52
- Community 53
- Community 54
- Community 55
- Community 56
- Community 57
- Community 58
- Community 59
- Community 60
- Community 61
- Community 62
- Community 63
- Community 64
- Community 65
- Community 66
- Community 67
- Community 68
- Community 69
- Community 70
- Community 79
- Community 80
- Community 86
- Community 87
- Community 88
- Community 90

## God Nodes (most connected - your core abstractions)
1. `set_seeds()` - 76 edges
2. `load_metadata()` - 65 edges
3. `filter_by_split()` - 51 edges
4. `GatingNetwork` - 46 edges
5. `CLAPEncoder` - 45 edges
6. `evaluate_retrieval()` - 36 edges
7. `ImageBindModel` - 24 edges
8. `main()` - 15 edges
9. `explain_gating_decision()` - 15 edges
10. `load_anchor_pairs()` - 14 edges

## Surprising Connections (you probably didn't know these)
- `Bird Image Sample - ImageBind Example` --conceptually_related_to--> `ImageBind Multimodal Embedding`  [INFERRED]
  experiments/audio_alignment/imagebind_code/.assets/bird_image.jpg → experiments/audio_alignment/imagebind_code/README.md
- `Car Image Sample - ImageBind Example` --conceptually_related_to--> `ImageBind Multimodal Embedding`  [INFERRED]
  experiments/audio_alignment/imagebind_code/.assets/car_image.jpg → experiments/audio_alignment/imagebind_code/README.md
- `Dog Image Sample - ImageBind Example` --conceptually_related_to--> `ImageBind Multimodal Embedding`  [INFERRED]
  experiments/audio_alignment/imagebind_code/.assets/dog_image.jpg → experiments/audio_alignment/imagebind_code/README.md
- `main()` --calls--> `CLAPEncoder`  [EXTRACTED]
  bin/evaluation/ablation_study.py → src/encoders/clap_encode.py
- `main()` --calls--> `GatingNetwork`  [EXTRACTED]
  bin/evaluation/ablation_study.py → src/models/gating_network.py

## Import Cycles
- None detected.

## Communities (99 total, 44 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.05
Nodes (30): get_video_frames(), GatingNetwork, main(), to_tensor(), init_bert_params(), MultiheadAttention, Tensor, Multi-headed attention. See "Attention Is All You Need" for more details. (+22 more)

### Community 1 - "Community 1"
Cohesion: 0.10
Nodes (38): main(), Demonstration pipeline with full explainability output., main(), Behavioural verification: demonstrate gating adapts per query type., encode_audio_query(), main(), Audio-to-video retrieval with explainability., main() (+30 more)

### Community 2 - "Community 2"
Cohesion: 0.10
Nodes (24): compare_all_methods.py -- Extended method comparison (adds M6 rows, full a2t…, M3: Shared-text-anchor alignment (ImageBind/Girdhar 2023; Grave 2019).…, check_ground_truth_alignment(), Step 4: Check if audio embeddings align with expected ground truth, Audio embedding diagnostics - step-by-step validation, evaluate_simple_fusion(), Quick evaluation script to compare different fusion methods, Simple evaluation without re-encoding (+16 more)

### Community 3 - "Community 3"
Cohesion: 0.12
Nodes (19): argparse, Clean up intermediate transformer checkpoints, keeping only best + last., compare_methods.py -- Orchestrator for AEMS audio-alignment method comparison.…, audio_sim_for(), normalize(), fusion_impact_aligned_db.py -- End-to-end fusion impact of any aligned audio…, M4: Adapter + contrastive fine-tuning (AudioCLIP/Guzhov 2022; Wav2CLIP/Wu 2022;…, M6: Wav2CLIP-style audio alignment -- align to CLIP-video, query stays CLIP-… (+11 more)

### Community 4 - "Community 4"
Cohesion: 0.08
Nodes (22): crop_boxes(), get_clip_timepoints(), load_and_transform_audio_data(), load_and_transform_text(), load_and_transform_video_data(), NormalizeVideo, Perform crop on the bounding boxes given the offsets. Args: boxes (ndarray or…, Perform uniform spatial sampling on the images and corresponding boxes. Args:… (+14 more)

### Community 5 - "Community 5"
Cohesion: 0.09
Nodes (24): angular_similarity(), get_all_configs(), get_method_configs(), loss_approx_ndcg(), loss_ce(), loss_info_nce(), loss_listmle(), loss_listnet() (+16 more)

### Community 6 - "Community 6"
Cohesion: 0.10
Nodes (22): main(), augment_text_targets.py -- Multi-positive CLIP-text targets for TRAIN split…, main(), build_beats_features.py -- BEATs (iter3+ AS2M) audio features…, 3 fixed 10s segments (+pad) mirroring the cached CLAP recipe., segments_from_wave(), main(), build_imagebind_features.py -- ImageBind (huge) audio features… (+14 more)

### Community 7 - "Community 7"
Cohesion: 0.13
Nodes (20): angular_similarity(), main(), Evaluation script for Priority 1 optimizations, Compute angular similarity between query and video embeddings, main(), Validate the linear projection on held-out TEST anchors.…, load_embeddings(), Load all required embeddings (+12 more)

### Community 8 - "Community 8"
Cohesion: 0.09
Nodes (26): AEMS Implementation Specification, Architecture System Overview, build_aems_manifest.py, extract_aems_frames.py, precompute_aems_audio_embeddings.py, precompute_aems_text_embeddings.py, precompute_aems_video_embeddings.py, export_aems_transformer_embeddings.py (+18 more)

### Community 9 - "Community 9"
Cohesion: 0.13
Nodes (19): extract_uniform_frames(), get_video_duration(), encode_clip_frames(), encode_clip_text(), get_video_frames(), info_nce_loss(), no_grad, train_epoch() (+11 more)

### Community 10 - "Community 10"
Cohesion: 0.12
Nodes (12): imagebind_huge(), Attention, BlockWithMasking, Mlp, MultiheadAttention, Tensor, Simple Transformer with the following features 1. Supports masked attention 2.…, Inputs - tokens: data of shape N x L x D (or L x N x D depending on the… (+4 more)

### Community 11 - "Community 11"
Cohesion: 0.14
Nodes (21): cli(), evaluate(), export(), list_checkpoints(), # TODO:, Evaluate a trained model., # TODO:, Export a model for inference. (+13 more)

### Community 12 - "Community 12"
Cohesion: 0.13
Nodes (21): Bug 4: Audio Retrieval Near-Zero R@1, Bug 1: Caption Truncation in eval_transformer_baseline.py, Bug 2: Missing Train Caption Embeddings in Gating Retraining, Angular Similarity Metric, CLAP Audio Encoder, CLIP Encoder (OpenAI ViT-B/32), Gating Network - Query-Adaptive Modality Fusion, Modality-Specific Normalization (+13 more)

### Community 13 - "Community 13"
Cohesion: 0.11
Nodes (9): einops, EinOpsRearrange, LearnableLogitScaling, Normalize, Module, Text Pooling used in OpenCLIP, SelectElement, SelectEOSAndProject (+1 more)

### Community 14 - "Community 14"
Cohesion: 0.13
Nodes (16): dtype, cast_if_src_dtype(), _get_pos_embedding(), get_sinusoid_encoding_table(), interpolate_pos_encoding(), interpolate_pos_encoding_2d(), Sinusoid position encoding table, # TODO: make it with torch instead of numpy (+8 more)

### Community 15 - "Community 15"
Cohesion: 0.13
Nodes (16): apply_whiten(), cca_directions(), collect_text_embeddings(), local_scaling(), Return projection Wx (audio->shared) that maximizes correlation. Returns (Wx,…, Hubness reduction: divide each embedding by its k-th-neighbor distance., Load CLIP-text (description) for arbitrary video list (test)., Shared helpers & evaluation harness for AEMS audio alignment methods.… (+8 more)

### Community 16 - "Community 16"
Cohesion: 0.18
Nodes (5): ImageBindModel, AudioPreprocessor, PadIm2Video, RGBDTPreprocessor, ThermalPreprocessor

### Community 17 - "Community 17"
Cohesion: 0.18
Nodes (13): build_adapter(), eval_model(), load_ssl_pairs(), main(), quick_a2t_r1(), Multi-positive InfoNCE (SupCon style) over in-batch target rows. pa: [B,512]…, m4_contrastive_ssl.py -- Adapter on strong self-supervised (WavLM) audio…, MLP adapter mapping WavLM audio (d_in) -> CLIP-text space (d_out). (+5 more)

### Community 18 - "Community 18"
Cohesion: 0.13
Nodes (11): main(), Final unified evaluation — all systems, identical conditions., check_clap_encoder(), Step 2: Test CLAP encoder functionality, Test different CLAP model configurations, test_clap_alternatives(), Test angular similarity instead of cosine similarity, test_angular_similarity() (+3 more)

### Community 19 - "Community 19"
Cohesion: 0.20
Nodes (8): eval_system(), main(), Ablation study: remove each modality, compare equal vs adaptive fusion., eval_system(), Phase 0: Heuristic Baselines for Gating Network Ablation Study. No training…, Phase 5: Deep run of the best configs (30 epochs), train_and_eval_deep(), evaluate_retrieval()

### Community 20 - "Community 20"
Cohesion: 0.14
Nodes (9): bootstrap_ci(), compute_recall_at_k(), angular_similarity(), Compute angular similarity between query and video embeddings Angular…, Shared utilities for gradient-based CLAP audio fine-tuning (Tiers 2/3).…, Bootstrap Confidence Intervals, Category-Stratified Recall Evaluation, laion_clap_training_data (+1 more)

### Community 21 - "Community 21"
Cohesion: 0.20
Nodes (13): encode_crop(), eval_model_on_features(), load_video_targets(), main(), mean_std(), pick(), quick_a2t_r1(), Deterministic rand_trunc 10s-window encoding of test clips. (+5 more)

### Community 22 - "Community 22"
Cohesion: 0.19
Nodes (7): MixUpAugment, Normalize, RandomResizedCrop, Resize, Scale, ShortSideScale, UniformTemporalSubsample

### Community 23 - "Community 23"
Cohesion: 0.23
Nodes (10): Focused runs: constant net tethered toward train-selected optimum…, Multi-seed validation of the best tethered-constant configs., main(), Fine-grained constant-weight sweep over the 2D simplex, reusing once-loaded…, sweep_weights(), Sweep constant weights using TRAIN-set R@1 to select the fusion, then eval on…, gc, scripts_ablation_run_ablation (+2 more)

### Community 24 - "Community 24"
Cohesion: 0.22
Nodes (5): build_causal_attention_mask(), IMUPreprocessor, PatchEmbedGeneric, Module, PatchEmbed from Hydra

### Community 25 - "Community 25"
Cohesion: 0.18
Nodes (9): apply_lora_to_linear(), __init__(), count_trainable(), inject_lora(), _recurse(), LoRALayer, LoRA (Hu et al. 2021) adapter modules for PyTorch nn.Linear. LoRA.proj for…, Wrap an existing nn.Linear into an nn.Module that keeps the base frozen and… (+1 more)

### Community 26 - "Community 26"
Cohesion: 0.18
Nodes (14): contrastive_loss(), main(), pairwise_negatives(), quick_mrr(), quick_r1(), Mine strongest in-batch negatives (excluding self) for each item., InfoNCE with in-batch negatives + hard-negative weighting. text_tgt:…, m4p_loss() (+6 more)

### Community 27 - "Community 27"
Cohesion: 0.18
Nodes (8): basic_clean(), bytes_to_unicode(), get_pairs(), Returns list of utf-8 byte and a corresponding list of unicode strings. The…, Return set of symbol pairs in a word. Word is represented as tuple of symbols…, SimpleTokenizer, whitespace_clean(), object

### Community 28 - "Community 28"
Cohesion: 0.22
Nodes (12): eval_direct(), eval_sim_matrix(), local_scale(), main(), mutual_knn_filter(), query_expansion(), Divide each embedding by its k-th-neighbor distance (cosine)., Mutual k-NN: only keep a text_i->audio_j edge if audio_j is in text_i's top-k… (+4 more)

### Community 29 - "Community 29"
Cohesion: 0.15
Nodes (9): analyze_top_examples(), compute_embeddings(), main(), Test different similarity computation methods, Analyze top examples for a few queries, Select subset of videos for testing, Compute all necessary embeddings for the test, select_video_subset() (+1 more)

### Community 30 - "Community 30"
Cohesion: 0.18
Nodes (6): Tensor, QuickGELU, Wrapper around nn.Module that prints registered buffers and parameter names., VerboseNNModule, no_grad, TextPreprocessor

### Community 31 - "Community 31"
Cohesion: 0.22
Nodes (5): collections, main(), show_category(), video_theme(), Error analysis: where does V+C fusion succeed vs fail? Only needs CLIP (not…

### Community 32 - "Community 32"
Cohesion: 0.22
Nodes (8): AudioCollator, build_audio_db_from_model(), clap_audio_embedding(), Encode a set of videos' waveforms into an aligned audio DB {vid: tensor}., Converts a list of waveforms into CLAP audio input dicts., Run the differentiable audio projection path (mirrors get_audio_embedding)., quick_eval(), Compute held-out test R@1/MRR (audio->text) using real waveform encode.

### Community 33 - "Community 33"
Cohesion: 0.20
Nodes (7): eval_direction(), evaluate_method(), procrustes_map(), query_mat [n,d], gallery_mat [n,d], matched rows. Returns metrics dict., Produce the full metric block for a projected audio gallery. audio_proj [n,d]…, Orthogonal Procrustes on centered inputs. Returns (W, b)., main()

### Community 34 - "Community 34"
Cohesion: 0.29
Nodes (9): average_cosine(), effective_dimensionality(), main(), pca_variance_explained(), Anisotropy Diagnostics for AEMS Embeddings…, Mean pairwise cosine similarity (anisotropy metric, Ethayarajh 2019). For unit…, Effective dimensionality from eigenvalue distribution of covariance. H(eig) =…, ZCA whitening (mean removal + decorrelation + renormalize). (+1 more)

### Community 35 - "Community 35"
Cohesion: 0.29
Nodes (9): ap_at_k(), main(), mean_median_rank(), mrr(), ndcg_at_k(), Comprehensive Retrieval Metrics & Evaluation Protocol Audit (AEMS audio-text)…, Mean Reciprocal Rank., Binary relevance NDCG@K. Correct video = relevance 1, else 0. (+1 more)

### Community 36 - "Community 36"
Cohesion: 0.25
Nodes (7): contrastive_audio_loss(), InfoNCE with identity positives + in-batch negatives. proj_audio, text_tgt:…, collate_deterministic(), main(), quick_eval(), Seed numpy+torch RNG so the random 10s window crop is reproducible., Deterministic held-out test R@1/MRR (audio->text).

### Community 37 - "Community 37"
Cohesion: 0.25
Nodes (9): eval_pair(), main(), Both-direction retrieval on matched rows. Returns full dict incl. median rank…, eval_pair(), main(), Both-direction retrieval on matched rows. Returns dict., hubness_skew(), load_anchor_pairs() (+1 more)

### Community 38 - "Community 38"
Cohesion: 0.25
Nodes (3): csv, main(), Comprehensive diagnostic analysis of AEMS multimodal retrieval system. Covers 5…

### Community 39 - "Community 39"
Cohesion: 0.25
Nodes (5): AudioWaveformDataset, load_waveform(), Mono float32 audio tensor (len = sr*clip_sec)., Pre-cached waveforms keyed by (video_id, audio_path, clip_sec). Eagerly loads…, load_waveform_cached()

### Community 40 - "Community 40"
Cohesion: 0.29
Nodes (5): PYTHONPATH, run_demo.sh script, PYTHONPATH, run_final_eval.sh script, venv_bin_activate

### Community 41 - "Community 41"
Cohesion: 0.29
Nodes (7): ImageBind Multimodal Embedding, Bird Image Sample - ImageBind Example, Car Image Sample - ImageBind Example, Dog Image Sample - ImageBind Example, ImageBind Model Card, ImageBind: One Embedding Space To Bind Them All, ImageBind Requirements

### Community 42 - "Community 42"
Cohesion: 0.38
Nodes (6): main(), procrustes(), Fit a linear projection mapping CLAP-audio embeddings into CLIP-text space.…, Solve min_W ||W X - Y||^2 + lam||W||^2 -> W = (XX^T + lam I)^-1 X Y^T. X: [d,…, Orthogonal Procrustes: W = U V^T from SVD of X Y^T (centered inputs)., ridge_fit()

### Community 43 - "Community 43"
Cohesion: 0.29
Nodes (4): MemoryBank, no_grad, Momentum-updated bank of projected audio embeddings for global hard negative…, Hardest global negatives (excluding self) for the given ids.

### Community 44 - "Community 44"
Cohesion: 0.38
Nodes (6): local_scaling(), main(), Hubness Diagnostics for AEMS Audio-Text Retrieval…, Skewness of a distribution. >0 => heavy right tail (hubness signature)., Local scaling (Zelnik-Manor & Perona 2004): normalize each embedding by its…, skewness()

### Community 45 - "Community 45"
Cohesion: 0.38
Nodes (6): centroid(), dist_metrics(), main(), Modality Gap Diagnostics for AEMS Multi-modal Embeddings…, Mean embedding of a set of L2-normalized embeddings (may not be unit norm)., Euclidean + cosine distance between two (possibly non-unit) vectors.

### Community 46 - "Community 46"
Cohesion: 0.33
Nodes (4): build_caption_index(), main(), Leave-one-out evaluation: prevent exact caption matching between queries and…, Build (video_id, text) -> index_in_video mapping matching precompute order.

### Community 47 - "Community 47"
Cohesion: 0.47
Nodes (3): GatingNetwork, main(), to_float()

### Community 48 - "Community 48"
Cohesion: 0.40
Nodes (5): find_audio_path(), find_paths(), find_paths(), main(), save_db_from_dict()

### Community 49 - "Community 49"
Cohesion: 0.33
Nodes (3): Test cross-modal alignment techniques, test_cross_modal_alignment(), __init__()

### Community 50 - "Community 50"
Cohesion: 0.47
Nodes (4): build_caption_index(), main(), per_query_success(), Quick evaluation of the LOO-retrained gating network. Uses the leave-one-out…

### Community 51 - "Community 51"
Cohesion: 0.47
Nodes (5): fixture, pytest, audio_embeds_path(), metadata_path(), video_embeds_path()

### Community 54 - "Community 54"
Cohesion: 0.67
Nodes (4): Audio Choices and Analysis, Angular Similarity (Audio Improvement), Audio-Text Semantic Misalignment Problem, Temperature Scaling (Similarity Adjustment)

### Community 56 - "Community 56"
Cohesion: 0.67
Nodes (3): audio_sim_for(), normalize(), fusion_impact_m4.py -- End-to-end fusion impact of the M4 aligned audio DB.…

### Community 59 - "Community 59"
Cohesion: 0.83
Nodes (3): compute_ranking_loss_fixed(), compute_ranking_loss_original(), test_ranking_alignment()

## Knowledge Gaps
- **37 isolated node(s):** `aems`, `run_demo.sh script`, `PYTHONPATH`, `run_final_eval.sh script`, `PYTHONPATH` (+32 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 377 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **44 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `set_seeds()` connect `Community 6` to `Community 2`, `Community 3`, `Community 5`, `Community 7`, `Community 9`, `Community 15`, `Community 17`, `Community 19`, `Community 20`, `Community 21`, `Community 23`, `Community 26`, `Community 28`, `Community 33`, `Community 34`, `Community 35`, `Community 36`, `Community 37`, `Community 42`, `Community 44`, `Community 45`, `Community 48`, `Community 55`, `Community 56`?**
  _High betweenness centrality (0.052) - this node is a cross-community bridge._
- **Why does `load_metadata()` connect `Community 7` to `Community 2`, `Community 3`, `Community 5`, `Community 6`, `Community 9`, `Community 15`, `Community 17`, `Community 19`, `Community 20`, `Community 26`, `Community 28`, `Community 33`, `Community 34`, `Community 35`, `Community 37`, `Community 42`, `Community 44`, `Community 45`, `Community 48`, `Community 55`, `Community 56`, `Community 58`?**
  _High betweenness centrality (0.038) - this node is a cross-community bridge._
- **Why does `CLAPEncoder` connect `Community 18` to `Community 1`, `Community 2`, `Community 33`, `Community 5`, `Community 38`, `Community 7`, `Community 46`, `Community 49`, `Community 50`, `Community 19`, `Community 20`, `Community 55`, `Community 58`, `Community 29`?**
  _High betweenness centrality (0.028) - this node is a cross-community bridge._
- **What connects `aems`, `run_demo.sh script`, `PYTHONPATH` to the rest of the system?**
  _37 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.052214452214452214 - nodes in this community are weakly interconnected._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.10087719298245613 - nodes in this community are weakly interconnected._
- **Should `Community 2` be split into smaller, more focused modules?**
  _Cohesion score 0.09523809523809523 - nodes in this community are weakly interconnected._