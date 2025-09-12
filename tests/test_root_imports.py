import importlib


def test_root_exports() -> None:
    mod = importlib.import_module("hei_nw")
    expected = {"DGKeyer", "EpisodicStore", "RecallService"}
    assert expected.issubset(set(mod.__all__)), mod.__all__
    for name in expected:
        getattr(mod, name)
