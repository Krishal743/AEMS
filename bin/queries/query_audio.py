"""Audio-to-video retrieval with explainability.

The clip is encoded with WavLM and projected into CLIP text space by the audio
adapter, then matched against the audio branch alone. The gate needs a CLIP
query and is therefore not applied.
"""

import argparse
import torch
from src.explainability.explain_retrieval import (
    explain_modality_contributions,
    explain_gating_decision,
    format_explanation,
)
from src.encoders.wavlm_encode import WavLMEncoder
from src.routing.query_router import (
    add_index_args, load_search_index, load_adapter, encode_audio_query, search,
)
from src.config import AEMS_AUDIO_ADAPTER_PATH

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser(description="Audio query retrieval")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--adapter", default=AEMS_AUDIO_ADAPTER_PATH)
    add_index_args(parser)
    args = parser.parse_args()

    index = load_search_index(args.video_embeds, args.audio_embeds, args.caption_embeds,
                              args.chunk_embeds)
    print(f"Candidates: {len(index.video_ids)} videos")

    audio_query = encode_audio_query(WavLMEncoder(device=DEVICE),
                                     load_adapter(args.adapter, DEVICE), args.audio, DEVICE)

    w, *sims = search(index, audio_query=audio_query)
    contributions = explain_modality_contributions(w, sims, index.video_ids, top_k=args.top_k)
    print(format_explanation(contributions, explain_gating_decision(w), top_k=args.top_k))


if __name__ == "__main__":
    main()
