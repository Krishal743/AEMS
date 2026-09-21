# Ablation Study Results

Run by: `bin/evaluation/ablation_study.py`
Date: July 2026
Gate weights: `models/gating_weights_meanpool.pth` (w_v=0.40, w_t=0.21, w_a=0.40)

## Results

| System | R@1 | R@5 | R@10 |
|--------|-----|-----|------|
| Visual only | 0.2165 | 0.4173 | 0.5116 |
| Caption only | 0.1159 | 0.2678 | 0.3475 |
| Audio only | 0.0003 | 0.0012 | 0.0026 |
| Visual + Caption (equal) | 0.2501 | 0.4567 | 0.5462 |
| Visual + Audio (equal) | 0.0490 | 0.1349 | 0.1903 |
| Caption + Audio (equal) | 0.0467 | 0.1266 | 0.1814 |
| Equal fusion (all 3) | 0.1711 | 0.3465 | 0.4322 |
| Adaptive gating | 0.1177 | 0.2544 | 0.3322 |
| No audio (equal v+t) | 0.2501 | 0.4567 | 0.5462 |
| Adaptive no audio | **0.2697** | **0.4822** | **0.5745** |

## Key Findings

1. **Audio is useless for retrieval**: R@1=0.0003, essentially random
2. **Equal fusion harms performance**: All-three equal fusion (0.1711) is worse than visual-only (0.2165)
3. **Adaptive gating is worse than equal fusion**: 0.1177 vs 0.1711, because the learned weights are near-uniform and add no benefit
4. **Best system: adaptive gating on visual+caption only**: R@1=0.2697 — removing audio entirely and renormalizing the learned weights improves over visual alone by 5 points
5. **Visual is the strongest single modality**: 0.2165 vs 0.1159 (caption) and 0.0003 (audio)

## Conclusion

The gating network does not improve retrieval over visual-only or visual+caption baselines. Audio negatively impacts all fusion variants. The optimal configuration is adaptive fusion of visual and caption only, which achieves a 25% relative improvement over visual-only baseline (0.2697 vs 0.2165).

See `outputs/ablations/ablation_table.csv` for raw data.
