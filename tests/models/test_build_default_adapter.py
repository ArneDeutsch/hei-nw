import pytest
import torch

from hei_nw.models.base import build_default_adapter


class DummyModel:
    class Config:
        pass

    config = Config()


def test_missing_config_fields() -> None:
    with pytest.raises(ValueError):
        build_default_adapter(DummyModel())


def test_adapter_uses_model_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """build_default_adapter should place the adapter on the model's device."""

    class ModelWithDevice(DummyModel):
        class Config:
            hidden_size = 8
            num_attention_heads = 2

        config = Config()
        device = "cuda"

    captured: dict[str, object | None] = {"device": None, "dtype": None}

    def fake_to(self, *args: object, **kwargs: object) -> object:  # pragma: no cover - simple shim
        device = kwargs.get("device") if "device" in kwargs else (args[0] if args else None)
        captured["device"] = device
        captured["dtype"] = kwargs.get("dtype")
        return self

    monkeypatch.setattr("hei_nw.adapter.EpisodicAdapter.to", fake_to)
    build_default_adapter(ModelWithDevice())
    assert captured["device"] == ModelWithDevice.device
    assert captured["dtype"] is None


def test_adapter_uses_model_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    """build_default_adapter should place the adapter in the model's dtype."""

    class ModelWithDType(DummyModel):
        class Config:
            hidden_size = 8
            num_attention_heads = 2

        config = Config()
        device = "cuda"
        dtype = torch.bfloat16

    captured: dict[str, object | None] = {"device": None, "dtype": None}

    def fake_to(self, *args: object, **kwargs: object) -> object:  # pragma: no cover - simple shim
        captured["device"] = kwargs.get("device")
        captured["dtype"] = kwargs.get("dtype")
        return self

    monkeypatch.setattr("hei_nw.adapter.EpisodicAdapter.to", fake_to)
    build_default_adapter(ModelWithDType())
    assert captured["device"] == ModelWithDType.device
    assert captured["dtype"] == ModelWithDType.dtype


def test_adapter_exposes_residual_gate() -> None:
    class ModelWithGate(DummyModel):
        class Config:
            hidden_size = 8
            num_attention_heads = 2

        config = Config()
        device = torch.device("cpu")
        dtype = torch.bfloat16

    adapter = build_default_adapter(ModelWithGate(), scale=0.3)
    assert hasattr(adapter, "alpha")
    assert isinstance(adapter.alpha, torch.nn.Parameter)
    assert adapter.alpha.requires_grad is True
    assert adapter.alpha.device.type == "cpu"
    assert adapter.alpha.dtype == torch.bfloat16
    assert torch.isclose(
        adapter.alpha.detach().float(),
        torch.tensor(0.3),
        atol=1e-2,
        rtol=1e-2,
    )
