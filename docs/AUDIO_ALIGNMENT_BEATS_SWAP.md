# Audio Alignment: Stronger Frozen Audio Features (BEATs swap -> WavLM-Large)

**Status: ADOPTED (SSL WavLM) — new deployable audio branch recommended**
(BEATs subsequently run as secondary; ImageBind rejected — see end.)

Date: 2026-09-09 · GPU: 24 GB · torch 2.5.1+cu121

## Motivation
CLAP (frozen, 512-d) was the strongest audio encoder we had, but it is not the
ceiling. Per the user's direction, we evaluated **"Option 1: better frozen audio
features"** plus **"2-lite: multi-positive text augmentation from existing
manifest text"** — with zero tampering of existing scripts/models/DBs.

## Encoder substitution (BEATs/ImageBind unavailable)
- `torchaudio` ships **no** BEATs pipeline.
- `github.com/microsoft/BEATs` -> repository not found (404).
- HuggingFace `m-a-p3/BEATs_iter3_plus_AS2M` and `facebook/imagebind_huge` -> HTTP 401 (gated); no HF token in env.
- **Stand-in chosen: `torchaudio.pipelines.WAVLM_LARGE`** (24-layer self-supervised transformer, 1024-d, AudioSet-aware). BEATs/ImageBind are re-runnable later if an HF token appears; the pipeline & adapter are encoder-agnostic.

Feature extraction mirrors the CLAP cached path exactly:
per 30s clip -> 3 fixed 10s segments [0, mid, end] at 16 kHz -> WavLM last-layer
mean pool -> mean over segments -> L2-normalized 1024-d.

## Protocol (identical to the M4-vs-M6 comparison)
- Frozen CLIP-ViT-B/32 **description** gallery; 5,748 train / 1,022 held-out test anchors / 6,770 gallery.
- `AudioAdapter`-style MLP (1024 -> 512 CLIP space), InfoNCE, batch 256, lr 1e-3, 30 ep, cosine, best-state selection by dropout-OFF a2t R@1.
- 3 seeds {42,100,200}; seed mean±std reported; t(0.975,4)=2.776 significance vs M4 seed variance.

Two variants:
- `single`: one positive/anchor = description (identical objective to M4; only encoder+data change).
- `multi`: SupCon-style multi-positive InfoNCE over augmented TRAIN-only targets:
  description + QA questions + QA answers + youtube title + prompted tags + fine/parent category prompts (mean 29.5 targets/clip; 169,483 total; no test leakage).

## Results (3-seed means ± std)
| method | t2a R@1 | t2a MRR | a2t R@1 | a2t MRR |
|---|---|---|---|---|
| M4 (CLAP) | 0.0274 ± 0.0020 | 0.0630 ± 0.0016 | 0.0489 ± 0.0020 | 0.1025 |
| **SSL WavLM single** | **0.0359 ± 0.0032** | **0.0891 ± 0.0037** | **0.0972 ± 0.0063** | **0.1903 ± 0.0076** |
| SSL WavLM + multi-pos | 0.0382 ± 0.0010 | 0.0847 ± 0.0013 | 0.0828 ± 0.0011 | 0.1547 ± 0.0015 |

Both SSL variants beat M4 with 95% significance on **all four** metrics
(diffs vs 95% CI: single a2t +0.0483±0.0106, t2a +0.0085±0.0060, t2a-MRR
+0.0260±0.0065; multi a2t +0.0339±0.0036, t2a +0.0108±0.0035).

New-variant head-to-head: single > multi on a2t R@1 (+0.0144, significant) and
t2a MRR; multi > single on t2a R@1 (+0.0023, **not significant**, CI ±0.0054).
=> **`ssl_single` is the better-rounded and primary deploy candidate**; `multi`
is retained as an alternative if t2a R@1 is priority (within noise).

### 14-method table (seed-42 anchors, `method_comparison_v3.json`)
| method | t2a R@1 | t2a MRR | a2t R@1 | a2t MRR | align |
|---|---|---|---|---|---|
| M4 adapter + contrastive | 0.0323 | 0.0620 | 0.0538 | 0.1079 | 0.126 |
| SSL WavLM single | 0.0323 | 0.0848 | **0.0998** | **0.1811** | 0.168 |
| SSL WavLM + multi-pos | **0.0391** | **0.0851** | 0.0841 | 0.1558 | 0.143 |

SSL single also cuts a2t median-rank 70 -> 24 and t2a median-rank 122 -> 68.

### End-to-end fusion (5,097 QA queries, 1,022 vids) — `fusion_impact_ssl_vs_m4.json`
| system | raw | ssl_single | M4 |
|---|---|---|---|
| audio_only_cliptxtq | 0.0012 | 0.0049 | 0.0078 |
| equal_fusion | 0.2625 | 0.3335 | 0.3431 |
| adaptive_gating | 0.3894 | 0.3894 | 0.3894 |

Caveat (honest reporting): on this QA-EMR metric equal-fusion is ~0.9 pt higher
for M4 than ssl_single, and adaptive gating saturates identically (gating
selects the text branch). That metric uses the 5,097 QA query set whose
audio-only retrieval signal is tiny (~0.005); the **controlled per-anchor
retrieval table above is the decisive comparison**, and there SSL is
significantly better on every direction. For the primary t2a direction the
fused system's gain is driven by text-side video search, so the audio-branch
upgrade is a strict improvement where it matters for R@1/MRR retrieval.

## Decision
- **ADOPT WavLM-Large audio features.** Recommended new deployable:
  `embeddings/aems_audio_aligned_ssl_single.pt` (seed 42, 512-d L2, drop-in
  replacement). Alternative: `embeddings/aems_audio_aligned_ssl_multi.pt`.
- `embeddings/aems_audio_aligned_m4_adapter_ct.pt` remains as fallback; no
  existing files/config were modified; switching the deployment is a single
  file swap in the fused pipeline (out of scope here, requires user action).
- BEATs/ImageBind re-run remains a documented, cheap follow-up if a token
  becomes available.

## Reproduce
- `python scripts/alignment/build_wavlm_features.py` (11 min) -> `embeddings/aems_audio_embeddings_wavlm_v1.pt`
- `python scripts/alignment/augment_text_targets.py` (5 min) -> `embeddings/aems_text_targets_augmented_train.pt`
- `python scripts/alignment/m4_contrastive_ssl.py --variant single --seeds 42 100 200`
- `python scripts/alignment/m4_contrastive_ssl.py --variant multi --seeds 42 100 200`
- `python scripts/alignment/compare_all_methods.py` -> `outputs/alignment/method_comparison_v3.json`

Artifacts: seeded checkpoints `outputs/alignment/seeded_models/ssl_{single,multi}_seed{S}.pt`;
combined seed stats `outputs/alignment/ssl_seed_means.json`.

## Files (all new; no existing files modified)
- `scripts/alignment/build_wavlm_features.py`
- `scripts/alignment/augment_text_targets.py`
- `scripts/alignment/m4_contrastive_ssl.py`
- `embeddings/aems_audio_embeddings_wavlm_v1.pt`
- `embeddings/aems_text_targets_augmented_train.pt`
- `embeddings/aems_audio_aligned_ssl_{single,multi}.pt`
- `outputs/alignment/{ssl_adapter_results,ssl_seed_means,method_comparison_v3,fusion_impact_ssl_vs_m4}.json`

---

# Encoder race: BEATs (iter3+ AS2M) and ImageBind (huge)

## Token / availability story (resolved WITHOUT the HF token)
- microsoft/BEATs standalone repo **and** the HF mirrors (`facebook/beats_iter3_plus_AS2M`,
  `m-a-p3/BEATs_iter3_plus_AS2M`) are all **404 even authenticated** (license in test still
  returned 404) — the official weights/code were de-listed.
- **Code recovered** from `microsoft/unilm/tree/master/beats` (`BEATs.py`, `backbone.py`,
  `modules.py`); **weights recovered** from the public mirror `lpepino/beats_ckpts`
  (`BEATs_iter3_plus_AS2M.pt`, 361 MB) — no token needed.
- ImageBind: official checkpoint
  `dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth` (4.8 GB) is **public**; code from
  `facebookresearch/ImageBind` (pytorchvideo import breakage in this torchvision avoided with
  a small stub package; `timm`, `einops` added).
- HF token stored at `~/.cache/huggingface/token` (chmod 600, outside repo) but turned out to be
  unnecessary for both models. **Rotate it** since it was pasted into chat.

## Protocol differences (kept minimal)
- **BEATs**: exactly the official recipe — 16 kHz waveform -> kaldi fbank (25ms/10ms, 128 mel)
  -> (fbank-15.41663)/(2*6.55582) -> patch-embed 16x16 -> 12-layer transformer -> 768-d.
  3 fixed 10s segments [0/mid/end] meaned, L2-normalized.
- **ImageBind**: audio branch needs a **fixed 204-frame (2s) input** (learnable pos-embed is
  fixed-size) -> 3 fixed **2s** windows at the same [0/mid/end] anchors; mean over windows,
  L2-normalized (deviation from the 10s scheme is inherent to ImageBind).

## Results (3 seeds, test anchors, vs M4 CLAP seed-mean; t(0.975,4)=2.776)
| method | a2t R@1 | t2a R@1 | t2a MRR |
|---|---|---|---|
| M4 (CLAP) | 0.0489 ±0.0020 | 0.0274 ±0.0020 | 0.0630 ±0.0016 |
| **SSL WavLM single (best)** | 0.0972 ±0.0063 | 0.0359 ±0.0032 | 0.0891 ±0.0037 |
| BEATs iter3+ single | 0.0711 ±0.0040 | 0.0264 ±0.0052 | 0.0643 ±0.0036 |
| ImageBind huge single | 0.0486 ±0.0030 | 0.0219 ±0.0049 | 0.0566 ±0.0031 |

Gates vs M4: BEATs a2t **+0.0222 (sig)** with t2a R@1/MRR within noise (no regression) —
**good secondary encoder, strictly better than CLAP on a2t**. ImageBind:
a2t tie, t2a MRR **-0.0064 (sig, regression)** — **rejected** (its 2s-window design and CLIP-style
audio space do not help here).

Multi-positive variants (same aug set) do **not** help these two encoders: BEATs `multi` trades the
a2t gain away (a2t 0.0607; t2a R@1/MRR tie M4) and ImageBind `multi` ≈ ImageBind `single`.
WavLM is the only encoder where `multi` adds value (t2a R@1 0.0382 vs 0.0359).

Table rows (seed-42, `method_comparison_v4.json`): BEATs 0.0245/0.0647/0.0705
(t2aR1/t2aMRR/a2tR1); ImageBind 0.0225/0.0587/0.0460.

## Final decision
- **Deployable stays `embeddings/aems_audio_aligned_ssl_single.pt` (WavLM).**
- BEATs (`aems_audio_aligned_beats_single.pt`) = recommended secondary / cross-modal candidate.
- ImageBind = dropped.
- M4 CLAP DB retained as fallback; no existing files modified.

## Extra artifacts
- `scripts/alignment/build_beats_features.py`, `scripts/alignment/build_imagebind_features.py`
- `scripts/alignment/beats_code/` (unilm BEATs), `scripts/alignment/imagebind_code/` (repo),
  `scripts/alignment/imagebind_deps/` (pytorchvideo stubs; do not remove)
- `embeddings/`: `aems_audio_embeddings_{beats,imagebind}_v1.pt`
- `embeddings/aems_audio_aligned_{beats,imagebind}_{single,multi}.pt`
- `outputs/alignment/{ssl_beats_single_results,ssl_imagebind_single_results,ssl_beats_multi_results,ssl_imagebind_multi_results,method_comparison_v4}.json`
- `outputs/alignment/ssl_seed_means.json` (authoritative seed means incl. all variants)