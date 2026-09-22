import pytest
import torch
import torch.nn.functional as F

from src.models.audio_adapter import AudioAdapter, load_audio_adapter
from src.models.gating_network import GatingNetwork
from src.routing.query_router import (
    compute_modal_similarities,
    load_gate,
    load_search_index,
    fixed_weights,
    search,
    zscore,
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
    clip_q, audio_q = unit(1, D), unit(1, D)
    sim_v, sim_t, sim_a = compute_modal_similarities(video_m, caption_m, audio_m, clip_q, audio_q)
    assert torch.allclose(sim_v, (clip_q @ video_m.T).squeeze(0))
    assert torch.allclose(sim_t, (clip_q @ caption_m.T).squeeze(0))
    assert torch.allclose(sim_a, (audio_q @ audio_m.T).squeeze(0))


def test_missing_space_leaves_branch_out(index):
    _, video_m, caption_m, audio_m = index
    sim_v, sim_t, sim_a = compute_modal_similarities(video_m, caption_m, audio_m, clip_query=unit(1, D))
    assert sim_v is not None and sim_t is not None and sim_a is None


def test_search_retrieves_exact_match(index):
    _, video_m, caption_m, audio_m = index
    weights, sim_v, sim_t, sim_a = search(index, clip_query=caption_m[2:3], audio_query=audio_m[2:3])
    fused = weights[0] * sim_v + weights[1] * sim_t + weights[2] * sim_a
    assert torch.isclose(weights.sum(), torch.tensor(1.0))
    assert fused.argmax().item() == 2


def test_gate_mode_takes_weights_from_the_gate(index, gate):
    q = unit(1, D)
    weights, *_ = search(index, clip_query=q, audio_query=q, gate=gate)
    with torch.no_grad():
        expected = gate(q).squeeze(0)
    assert torch.allclose(weights, expected / expected.sum(), atol=1e-6)
    assert not torch.allclose(weights, fixed_weights() / fixed_weights().sum())


def test_search_returns_zscored_similarities(index):
    _, video_m, _, _ = index
    q = unit(1, D)
    _, sim_v, _, _ = search(index, clip_query=q)
    assert torch.allclose(sim_v, zscore((q @ video_m.T).squeeze(0)), atol=1e-6)
    assert abs(sim_v.mean().item()) < 1e-5 and abs(sim_v.std().item() - 1) < 1e-3


def test_fixed_weights_are_used_and_renormalized(index):
    weights, *_ = search(index, clip_query=unit(1, D), audio_query=unit(1, D),
                         weights={"visual": 1.0, "text": 2.0, "audio": 1.0})
    assert torch.allclose(weights, torch.tensor([0.25, 0.5, 0.25]))


def test_search_masks_unavailable_branch(index, gate):
    weights, _, _, sim_a = search(index, clip_query=unit(1, D), gate=gate)
    assert weights[2] == 0
    assert torch.isclose(weights.sum(), torch.tensor(1.0))
    assert torch.count_nonzero(sim_a) == 0


def test_audio_only_query_uses_audio_branch_even_with_zero_weight(index):
    weights, *_ = search(index, audio_query=unit(1, D),
                         weights={"visual": 0.3, "text": 1.0, "audio": 0.0})
    assert torch.equal(weights, torch.tensor([0.0, 0.0, 1.0]))


def test_search_rejects_empty_query(index, gate):
    with pytest.raises(ValueError):
        search(index, gate=gate)


def test_audio_adapter_round_trip(tmp_path):
    torch.manual_seed(0)
    adapter = AudioAdapter(input_dim=1024).eval()
    path = tmp_path / "adapter.pth"
    torch.save(adapter.state_dict(), path)
    loaded = load_audio_adapter(path, "cpu")
    x = torch.randn(3, 1024)
    with torch.no_grad():
        out = loaded(x)
        assert torch.allclose(out, adapter(x))
    assert out.shape == (3, 512)
    assert torch.allclose(out.norm(dim=1), torch.ones(3))


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
