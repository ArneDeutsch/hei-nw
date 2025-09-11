import torch

from hei_nw.adapter import EpisodicAdapter


def test_noop_when_memory_empty() -> None:
    torch.manual_seed(0)
    adapter = EpisodicAdapter(hidden_size=8, n_heads=2)
    h = torch.randn(2, 3, 8)

    out_none = adapter(h, None)
    assert out_none is h
    assert torch.allclose(out_none, h)
    assert out_none.detach().cpu().numpy().tobytes() == h.detach().cpu().numpy().tobytes()

    empty_mem = torch.randn(2, 0, 8)
    out_empty = adapter(h, empty_mem)
    assert out_empty is h
    assert torch.equal(out_empty, h)


def test_shapes_with_memory_tokens() -> None:
    torch.manual_seed(1)
    hidden = 16
    adapter = EpisodicAdapter(hidden_size=hidden, n_heads=4)
    h = torch.randn(2, 7, hidden)
    memory = torch.randn(2, 5, hidden)
    out = adapter(h, memory)
    assert out.shape == h.shape
    assert out is not h
