import pytest

from hei_nw.models.base import build_default_adapter


class DummyModel:
    class Config:
        pass

    config = Config()


def test_missing_config_fields() -> None:
    with pytest.raises(ValueError):
        build_default_adapter(DummyModel())
