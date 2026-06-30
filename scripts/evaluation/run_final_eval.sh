#!/usr/bin/env bash
set -e

# ============================================================
# Final Evaluation Orchestration
# Run this AFTER Phase B training (epochs 5-12) completes.
# Each step is independent; you can run them selectively.
# ============================================================

source venv/bin/activate
mkdir -p outputs/eval

echo "============================================================"
echo "FINAL EVALUATION — Post Transformer Training (Phase D)"
echo "============================================================"
echo ""

# ----- Step D1: Verify exported embeddings -----
echo "[D1] Verifying exported embeddings..."
python3 -c "
import torch
emb = torch.load('embeddings/video_embeddings_transformer.pt', weights_only=False)
sample = list(emb.values())[0]
print(f'  Embeddings: {len(emb)} videos, shape: {sample.shape}')
assert sample.shape == (512,), f'Expected (512,), got {sample.shape}'
print('  [OK] Embeddings verified.')
" 2>&1 | tee outputs/eval/d1_verify.txt

# ----- Step D2: Official old baseline -----
echo ""
echo "[D2] Official CLIP baseline (run_clip_baseline.py)..."
python3 scripts/baselines/run_clip_baseline.py 2>&1 | tee outputs/eval/d2_clip_baseline.txt

# ----- Step D3: Transformer visual-only eval -----
echo ""
echo "[D3] Transformer visual-only eval..."
python3 scripts/evaluation/eval_transformer_baseline.py 2>&1 | tee outputs/eval/d3_transformer_eval.txt

# ----- Step D4: Transformer vs old comparison -----
echo ""
echo "[D4] Transformer vs old comparison..."
python3 scripts/evaluation/eval_transformer_baseline.py --compare 2>&1 | tee outputs/eval/d4_comparison.txt

# ----- Step D5: Retrain gating on OLD embeddings -----
echo ""
echo "[D5] Retraining gating on OLD embeddings..."
python3 scripts/training/retrain_gating.py \
    --video-embeds embeddings/video_embeddings.pt \
    --gate-weights models/gating_weights_meanpool.pth \
    --epochs 5 \
    --train-queries 5000 2>&1 | tee outputs/eval/d5_gating_old.txt

# ----- Step D6: Retrain gating on TRANSFORMER embeddings -----
echo ""
echo "[D6] Retraining gating on TRANSFORMER embeddings..."
python3 scripts/training/retrain_gating.py \
    --video-embeds embeddings/video_embeddings_transformer.pt \
    --gate-weights models/gating_weights_transformer.pth \
    --epochs 5 \
    --train-queries 5000 2>&1 | tee outputs/eval/d6_gating_transformer.txt

# ----- Step D7: Full multimodal comparison -----
echo ""
echo "[D7] Full multimodal comparison..."
echo ""

echo "--- Old embeddings + gating ---"
python3 scripts/evaluation/eval_multimodal.py \
    --video-embeds embeddings/video_embeddings.pt \
    --gate-weights models/gating_weights_meanpool.pth \
    --use-gate 2>&1 | tee outputs/eval/d7a_old_gated.txt

echo ""
echo "--- Transformer embeddings + gating ---"
python3 scripts/evaluation/eval_multimodal.py \
    --video-embeds embeddings/video_embeddings_transformer.pt \
    --gate-weights models/gating_weights_transformer.pth \
    --use-gate 2>&1 | tee outputs/eval/d7b_transformer_gated.txt

echo ""
echo "============================================================"
echo "ALL EVALUATIONS COMPLETE"
echo "Results saved to outputs/eval/"
echo "============================================================"
echo ""
echo "Summary of key files:"
echo "  outputs/eval/d2_clip_baseline.txt        — Official old baseline"
echo "  outputs/eval/d3_transformer_eval.txt     — Transformer visual-only"
echo "  outputs/eval/d4_comparison.txt           — Transformer vs old comparison"
echo "  outputs/eval/d5_gating_old.txt           — Gating retrained on old embeddings"
echo "  outputs/eval/d6_gating_transformer.txt   — Gating retrained on transformer"
echo "  outputs/eval/d7a_old_gated.txt           — Old embeds + gating (baseline)"
echo "  outputs/eval/d7b_transformer_gated.txt   — Transformer embeds + gating (final)"
