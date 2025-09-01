import os
from types import SimpleNamespace
from eval_pipeline import build_and_cache_eval_prompts, compute_endeca_accuracy_for_completion, compute_mean_endeca_accuracy


class DummyChecker:
    def get_all_verses(self):
        # Not used by build_and_cache_eval_prompts (uses utils.create_dataset)
        return []


def test_build_and_cache_eval_prompts(tmp_path, monkeypatch):
    # Monkeypatch utils.create_dataset to return a predictable dataset
    import eval_pipeline as ep
    calls = {"n": 0}

    class FakeDS(dict):
        def __getitem__(self, k):
            if k == "prompt":
                return ["P1", "P2", "P3"]
            return super().__getitem__(k)

    def fake_create_dataset(n, dc):
        calls["n"] += 1
        return FakeDS()

    monkeypatch.setattr(ep, "create_dataset", fake_create_dataset)
    cache_dir = tmp_path / "cache"
    prompts = build_and_cache_eval_prompts(DummyChecker(), str(cache_dir), sample_size=3, seed=123)
    assert prompts == ["P1", "P2", "P3"]
    # Call again: should load from cache without invoking create_dataset
    prompts2 = build_and_cache_eval_prompts(DummyChecker(), str(cache_dir), sample_size=3, seed=123)
    assert prompts2 == prompts
    assert calls["n"] == 1


def test_compute_endeca_accuracy_for_completion(monkeypatch):
    import eval_pipeline as ep

    def fake_is_endecasillabo(line):
        return (line.endswith("X"), "syl")

    monkeypatch.setattr(ep, "is_endecasillabo", fake_is_endecasillabo)
    text = "aX\nb\ncX\n\n"
    score = compute_endeca_accuracy_for_completion(text)
    # Lines: aX (True), b (False), cX (True) => 2/3
    assert abs(score - (2.0 / 3.0)) < 1e-6


def test_compute_mean_endeca_accuracy(monkeypatch):
    import eval_pipeline as ep

    def fake_is_endecasillabo(line):
        return (line == "ok", "syl")

    monkeypatch.setattr(ep, "is_endecasillabo", fake_is_endecasillabo)
    completions = ["ok\nno", "no\nok", "ok\nok"]
    # Per completion accuracies: [1/2, 1/2, 1]
    mean_score = compute_mean_endeca_accuracy(completions)
    assert abs(mean_score - ((0.5 + 0.5 + 1.0) / 3.0)) < 1e-6



