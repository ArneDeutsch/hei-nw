import pytest

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

    captured: dict[str, str | None] = {"device": None}

    def fake_to(self, device: str | None) -> object:  # pragma: no cover - simple shim
        captured["device"] = device
        return self

    monkeypatch.setattr("hei_nw.adapter.EpisodicAdapter.to", fake_to)
    build_default_adapter(ModelWithDevice())
    assert captured["device"] == ModelWithDevice.device
