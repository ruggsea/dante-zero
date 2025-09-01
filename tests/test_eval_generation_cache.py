import os
import json
from eval_pipeline import generate_completions_for_prompts


def test_generate_completions_for_prompts_cache(tmp_path):
    prompts = ["P1", "P2", "P3"]

    def fake_gen(pp):
        return [p + " :: C" for p in pp]

    out = generate_completions_for_prompts(
        model_dir=str(tmp_path / "m"),
        prompts=prompts,
        cache_base_dir=str(tmp_path / "cache"),
        run_name="runA",
        checkpoint_name="best",
        generator=fake_gen,
    )
    assert os.path.exists(out)
    with open(out, "r", encoding="utf-8") as f:
        rows = [json.loads(ln) for ln in f]
    assert len(rows) == 3
    assert rows[0]["completion"].endswith(":: C")

    # Second call should reuse existing file
    out2 = generate_completions_for_prompts(
        model_dir=str(tmp_path / "m"),
        prompts=prompts,
        cache_base_dir=str(tmp_path / "cache"),
        run_name="runA",
        checkpoint_name="best",
        generator=fake_gen,
    )
    assert out2 == out



