import os
from config import compute_output_dir


def test_compute_output_dir_prefers_arg(tmp_path, monkeypatch):
    monkeypatch.setenv("CHECKPOINT_LOCATION", str(tmp_path / "ckpts"))
    out = compute_output_dir("runx", "/custom/out")
    assert out == "/custom/out"


def test_compute_output_dir_env_used_when_arg_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("CHECKPOINT_LOCATION", str(tmp_path / "ckpts"))
    out = compute_output_dir("runx", None)
    assert out == os.path.join(str(tmp_path / "ckpts"), "runx")


def test_compute_output_dir_default_when_no_env(monkeypatch):
    monkeypatch.delenv("CHECKPOINT_LOCATION", raising=False)
    out = compute_output_dir("runx", None)
    assert out.endswith("outputs/runx")



