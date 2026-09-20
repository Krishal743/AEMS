#!/usr/bin/env bash
set -e

source venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "Enter your search query (text):"
read -r QUERY

python3 scripts/demo/demo.py \
    --query "$QUERY" \
    --video-embeds embeddings/video_embeddings.pt \
    --audio-embeds embeddings/audio_embeddings.pt \
    --caption-embeds embeddings/caption_embeddings_test.pt \
    --gate-weights models/gating_weights_meanpool.pth \
    --top-k 5
