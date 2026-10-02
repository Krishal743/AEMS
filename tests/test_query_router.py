import pytest
import torch
import torch.nn.functional as F

from src.models.audio_adapter import AudioAdapter, load_audio_adapter
from src.models.gating_network import GatingNetwork
from src.routing.query_router import (
    BRANCHES,
    ChunkIndex,
    SearchIndex,
    compute_modal_similarities,
    fixed_weights,
    load_gate,
    load_search_index,
    search,
    zscore,
)

N, D, CHUNKS = 6, 512, 3
NB = len(BRANCHES)


def unit(*shape):
    return F.normalize(torch.randn(*shape), dim=-1)


def chunk_index(per_video):
    """per_video: (n_videos, chunks, D) -> ChunkIndex."""
    n, k, _ = per_video.shape
    return ChunkIndex(per_video.reshape(n * k, D),
                      torch.arange(n).repeat_interleave(k), n)


@pytest.fixture
def chunks():
    torch.manual_seed(1)
    return unit(N, CHUNKS, D)


@pytest.fixture
def index(chunks):
    torch.manual_seed(0)
    return SearchIndex([f"v{i}" for i in range(N)], unit(N, D), unit(N, D),
                       chunk_index(chunks), unit(N, D))


@pytest.fixture
def gate():
    torch.manual_seed(0)
    return GatingNetwork(input_dim=D, hidden_dim=128, num_modalities=NB).eval()


def test_similarities_use_the_matching_branch(index):
    clip_q, audio_q = unit(1, D), unit(1, D)
    sim_v, sim_t, sim_c, sim_a, sim_b, sim_d = compute_modal_similarities(
        index.visual, index.text, index.audio, clip_q, audio_q, index.chunk)
    assert torch.allclose(sim_v, (clip_q @ index.visual.T).squeeze(0))
    assert torch.allclose(sim_t, (clip_q @ index.text.T).squeeze(0))
    assert torch.allclose(sim_a, (audio_q @ index.audio.T).squeeze(0))
    assert sim_c.shape == (N,)
    assert sim_b is None                        # no BM25 index on this fixture
    assert sim_d is None                        # no dense index on this fixture


def test_passage_branch_scores_the_best_chunk(index, chunks):
    """A query equal to one passage must score that video by exactly that match."""
    q = chunks[4, 1:2]
    sim_c = index.chunk.max_sim(q)
    assert sim_c.argmax().item() == 4
    assert torch.isclose(sim_c[4], torch.tensor(1.0), atol=1e-5)
    expected = (q @ chunks.reshape(N * CHUNKS, D).T).squeeze(0).reshape(N, CHUNKS).max(dim=1).values
    assert torch.allclose(sim_c, expected, atol=1e-5)


def test_max_sim_batch_matches_single(index):
    qs = unit(4, D)
    batched = index.chunk.max_sim_batch(qs)
    for i in range(4):
        assert torch.allclose(batched[i], index.chunk.max_sim(qs[i:i + 1]), atol=1e-6)


def test_missing_branch_is_left_out(index):
    sims = compute_modal_similarities(index.visual, index.text, index.audio,
                                      clip_query=unit(1, D), chunk_index=index.chunk)
    assert sims[3] is None                      # no audio query
    assert sims[4] is None                      # no BM25 index
    assert sims[5] is None                      # no dense index
    assert all(s is not None for s in sims[:3])


def test_search_retrieves_exact_match(index):
    weights, *sims = search(index, clip_query=index.text[2:3], audio_query=index.audio[2:3])
    fused = sum(w * s for w, s in zip(weights, sims))
    assert len(sims) == NB
    assert torch.isclose(weights.sum(), torch.tensor(1.0))
    assert fused.argmax().item() == 2


def test_search_returns_zscored_similarities(index):
    q = unit(1, D)
    _, sim_v, *_ = search(index, clip_query=q)
    assert torch.allclose(sim_v, zscore((q @ index.visual.T).squeeze(0)), atol=1e-6)
    assert abs(sim_v.mean().item()) < 1e-5 and abs(sim_v.std().item() - 1) < 1e-3


def test_fixed_weights_are_used_and_renormalized(index):
    weights, *_ = search(index, clip_query=unit(1, D), audio_query=unit(1, D),
                         weights={"visual": 1.0, "text": 2.0, "chunk": 1.0, "audio": 0.0,
                                  "bm25": 5.0, "dense": 5.0})
    # bm25 and dense are unreachable without their indexes, so both are masked out
    assert torch.allclose(weights, torch.tensor([0.25, 0.5, 0.25, 0.0, 0.0, 0.0]))


def test_gate_mode_takes_weights_from_the_gate(index, gate):
    q = unit(1, D)
    weights, *_ = search(index, clip_query=q, audio_query=q, gate=gate)
    with torch.no_grad():
        expected = gate(q).squeeze(0).clone()
    for unreachable in ("bm25", "dense"):      # no index for either on this fixture
        expected[BRANCHES.index(unreachable)] = 0.0
    assert torch.allclose(weights, expected / expected.sum(), atol=1e-6)


def test_search_masks_unavailable_branches(index, gate):
    weights, *sims = search(index, clip_query=unit(1, D), gate=gate)
    sim_a = sims[BRANCHES.index("audio")]
    assert weights[BRANCHES.index("audio")] == 0
    assert torch.isclose(weights.sum(), torch.tensor(1.0))
    assert torch.count_nonzero(sim_a) == 0


def test_audio_only_query_reaches_only_the_audio_branch(index):
    weights, *sims = search(index, audio_query=unit(1, D))
    expected = torch.zeros(NB)
    expected[BRANCHES.index("audio")] = 1.0
    assert torch.equal(weights, expected)
    assert all(torch.count_nonzero(s) == 0
               for i, s in enumerate(sims) if i != BRANCHES.index("audio"))


def test_search_rejects_empty_query(index, gate):
    with pytest.raises(ValueError):
        search(index, gate=gate)


def test_bm25_branch_scores_literal_word_overlap():
    """The lexical branch must find the video whose passage shares a rare term."""
    from src.retrieval.bm25 import BM25PassageIndex
    bm25 = BM25PassageIndex([["a lecture on the bellman equation"],
                             ["renaissance painting"],
                             ["kneading bread dough"]])
    idx = SearchIndex([f"v{i}" for i in range(3)], unit(3, D), unit(3, D),
                      chunk_index(unit(3, CHUNKS, D)), unit(3, D), bm25)
    _, *sims = search(idx, clip_query=unit(1, D), query_text="what is the bellman equation")
    assert sims[BRANCHES.index("bm25")].argmax().item() == 0


def test_bm25_branch_is_masked_without_query_text(index, gate):
    weights, *_ = search(index, clip_query=unit(1, D), gate=gate)
    assert weights[BRANCHES.index("bm25")] == 0


def test_dense_branch_scores_its_own_embedding_space():
    """The dense branch has its own dim and query vector, unlike the CLIP branches."""
    torch.manual_seed(0)
    dense_dim, n_chunks = 768, 2
    rows = F.normalize(torch.randn(3 * n_chunks, dense_dim), dim=1)
    dense = ChunkIndex(rows, torch.arange(3).repeat_interleave(n_chunks), 3)
    idx = SearchIndex([f"v{i}" for i in range(3)], unit(3, D), unit(3, D),
                      chunk_index(unit(3, CHUNKS, D)), unit(3, D), None, dense)
    query = rows[4:5]                                   # exactly video 2's second chunk
    _, *sims = search(idx, clip_query=unit(1, D), dense_query=query)
    assert sims[BRANCHES.index("dense")].argmax().item() == 2


def test_index_without_chunks_still_searches(index):
    no_chunks = index._replace(chunk=None)
    weights, *sims = search(no_chunks, clip_query=unit(1, D))
    assert weights[BRANCHES.index("chunk")] == 0
    assert torch.count_nonzero(sims[BRANCHES.index("chunk")]) == 0


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


def test_load_gate_rejects_wrong_branch_count(tmp_path):
    """A gate trained before the passage branch must fail, not silently misalign."""
    torch.manual_seed(0)
    old = GatingNetwork(input_dim=D, hidden_dim=128, num_modalities=NB - 1)
    path = tmp_path / "old_gate.pth"
    torch.save(old.state_dict(), path)
    with pytest.raises(RuntimeError, match="branches"):
        load_gate(path, "cpu")


def test_load_gate_rejects_mismatched_checkpoint(tmp_path, gate):
    state = gate.state_dict()
    state.pop("fc1.weight")
    path = tmp_path / "gate.pth"
    torch.save(state, path)
    with pytest.raises(RuntimeError):
        load_gate(path, "cpu")


def test_load_search_index_keeps_only_complete_videos(tmp_path):
    video = {"a": torch.randn(D), "b": torch.randn(D), "c": torch.randn(D)}
    audio = {"a": torch.randn(D), "c": torch.randn(D)}
    caption = {"a": torch.randn(D), "b": torch.randn(D), "c": torch.randn(D)}
    chunk = {"a": torch.randn(2, D), "b": torch.randn(3, D), "c": torch.randn(4, D)}
    paths = []
    for name, db in (("v", video), ("a", audio), ("t", caption), ("c", chunk)):
        paths.append(tmp_path / f"{name}.pt")
        torch.save(db, paths[-1])
    index = load_search_index(*paths)
    assert index.video_ids == ["a", "c"]
    for m in (index.visual, index.text, index.audio):
        assert m.shape == (2, D)
        assert torch.allclose(m.norm(dim=1), torch.ones(2))
    assert index.chunk.rows.shape == (6, D)      # 2 chunks for "a" + 4 for "c", "b" dropped
    assert index.chunk.owner.tolist() == [0, 0, 1, 1, 1, 1]
