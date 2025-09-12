from pathlib import Path

from hei_nw.utils.cli import parse_args


def test_parse_args_creates_outdir(tmp_path) -> None:  # type: ignore[no-untyped-def]
    outdir = tmp_path / "out"
    args = parse_args(["--outdir", str(outdir)])
    assert args.outdir == outdir
    assert outdir.exists()


def test_parse_args_uses_default_outdir(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    args = parse_args([])
    assert args.outdir == Path("reports/baseline")
    assert args.outdir.exists()
