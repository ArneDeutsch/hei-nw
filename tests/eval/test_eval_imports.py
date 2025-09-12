import sys
import types

import pytest


def test_lazy_imports() -> None:
    import hei_nw.eval as ev

    sys.modules.pop("hei_nw.eval.harness", None)
    assert "hei_nw.eval.harness" not in sys.modules
    mod = ev.harness
    assert isinstance(mod, types.ModuleType)
    assert mod.__name__ == "hei_nw.eval.harness"


def test_unknown_attribute_raises() -> None:
    import hei_nw.eval as ev

    with pytest.raises(AttributeError):
        _ = ev.unknown_module
