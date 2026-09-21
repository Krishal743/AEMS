import pytest
import torch
import torch.nn.functional as F

from src.models.gating_network import GatingNetwork
from src.routing.query_router import (
    compute_modal_similarities,
    load_gate,
    load_search_index,
    search,
)

N, D = 6, 512


def unit(*shape):
    return F.normalize(torch.randn(*shape), dim=-1)


@pytest.fixture
def index():
    torch.manual_seed(0)
    return [f"v{i}" for i in range(N)], unit(N, D), unit(N, D), unit(N, D)


@pytest.fixture
def gate():
    torch.manual_seed(0)
    return GatingNetwork(input_dim=D, hidden_dim=128).eval()


def test_similarities_use_the_matching_space(index):
    _, video_m, caption_m, audio_m = index
    clip_q, clap_q = unit(1, D), unit(1, D)
    sim_v, sim_t, sim_a = compute_modal_similarities(video_m, caption_m, audio_m, clip_q, clap_q)
    assert torch.allclose(sim_v, (clip_q @ video_m.T).squeeze(0))
    assert torch.allclose(sim_t, (clip_q @ caption_m.T).squeeze(0))
    assert torch.allclose(sim_a, (clap_q @ audio_m.T).squeeze(0))


def test_missing_space_leaves_branch_out(index):
    _, video_m, caption_m, audio_m = index
    sim_v, sim_t, sim_a = compute_modal_similarities(video_m, caption_m, audio_m, clip_query=unit(1, D))
    assert sim_v is not None and sim_t is not None and sim_a is None


def test_search_retrieves_exact_match(index, gate):
    _, video_m, caption_m, audio_m = index
    weights, sim_v, sim_t, sim_a = search(index, gate, clip_query=caption_m[2:3], clap_query=audio_m[2:3])
    fused = weights[0] * sim_v + weights[1] * sim_t + weights[2] * sim_a
    assert torch.isclose(weights.sum(), torch.tensor(1.0))
    assert fused.argmax().item() == 2


def test_search_masks_unavailable_branch(index, gate):
    weights, _, _, sim_a = search(index, gate, clip_query=unit(1, D))
    assert weights[2] == 0
    assert torch.isclose(weights.sum(), torch.tensor(1.0))
    assert torch.count_nonzero(sim_a) == 0


def test_audio_only_query_bypasses_gate(index, gate):
    weights, _, _, _ = search(index, gate, clap_query=unit(1, D))
    assert torch.equal(weights, torch.tensor([0.0, 0.0, 1.0]))


def test_search_rejects_empty_query(index, gate):
    with pytest.raises(ValueError):
        search(index, gate)


def test_load_gate_round_trip(tmp_path, gate):
    path = tmp_path / "gate.pth"
    torch.save(gate.state_dict(), path)
    loaded = load_gate(path, "cpu")
    q = unit(1, D)
    with torch.no_grad():
        assert torch.allclose(loaded(q), gate(q))


def test_load_gate_rejects_mismatched_checkpoint(tmp_path, gate):
    state = gate.state_dict()
    state.pop(next(n for n, _ in gate.named_parameters()))
    path = tmp_path / "gate.pth"
    torch.save(state, path)
    with pytest.raises(RuntimeError):
        load_gate(path, "cpu")


def test_load_search_index_keeps_only_complete_videos(tmp_path):
    video = {"a": torch.randn(D), "b": torch.randn(D), "c": torch.randn(D)}
    audio = {"a": torch.randn(D), "c": torch.randn(D)}
    caption = {"a": torch.randn(D), "b": torch.randn(D), "c": torch.randn(D)}
    paths = []
    for name, db in (("v", video), ("a", audio), ("t", caption)):
        paths.append(tmp_path / f"{name}.pt")
        torch.save(db, paths[-1])
    video_ids, video_m, caption_m, audio_m = load_search_index(*paths)
    assert video_ids == ["a", "c"]
    for m in (video_m, caption_m, audio_m):
        assert m.shape == (2, D)
        assert torch.allclose(m.norm(dim=1), torch.ones(2))
