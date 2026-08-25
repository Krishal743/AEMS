import pytest
import os

@pytest.fixture
def metadata_path():
    return "data/processed/metadata/msrvtt_metadata.json"

@pytest.fixture
def video_embeds_path():
    return "embeddings/video_embeddings.pt"

@pytest.fixture
def audio_embeds_path():
    return "embeddings/audio_embeddings.pt"
