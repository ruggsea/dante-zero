import os
from pathlib import Path
from save_utils import save_final


class DummyTrainer:
    def __init__(self, should_fail=False):
        self.should_fail = should_fail
        self.saved = None
    def save_model(self, path):
        if self.should_fail:
            raise RuntimeError("save fail")
        os.makedirs(path, exist_ok=True)
        Path(path, "pytorch_model.bin").write_bytes(b"dummy")
        self.saved = path


class DummyModel:
    def __init__(self):
        self.saved = None
    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        Path(path, "model.safetensors").write_bytes(b"dummy")
        self.saved = path


class DummyTokenizer:
    def __init__(self):
        self.saved = None
    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        Path(path, "tokenizer.json").write_text("{}", encoding="utf-8")
        self.saved = path


def test_save_final_happy(tmp_path):
    trainer = DummyTrainer(should_fail=False)
    model = DummyModel()
    tok = DummyTokenizer()
    save_final(str(tmp_path), trainer, model, tok)
    out = tmp_path / "final"
    assert (out / "pytorch_model.bin").exists() or (out / "model.safetensors").exists()
    assert (out / "tokenizer.json").exists()


def test_save_final_trainer_fallback(tmp_path):
    trainer = DummyTrainer(should_fail=True)
    model = DummyModel()
    tok = DummyTokenizer()
    save_final(str(tmp_path), trainer, model, tok)
    out = tmp_path / "final"
    # trainer failed, but model/tokenizer saved
    assert (out / "model.safetensors").exists()
    assert (out / "tokenizer.json").exists()


