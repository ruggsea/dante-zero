import os
import pytest

import judge as judge_mod


JUDGE_URL = os.environ.get("JUDGE_URL", "http://127.0.0.1:8008")
MODEL_ID = os.environ.get("JUDGE_MODEL", "Qwen3-235B-A22B-Thinking-2507-AWQ")


def test_judge_prefers_better_verse():
    try:
        _ = judge_mod.list_models(JUDGE_URL)
    except Exception as e:
        pytest.fail(f"Judge not reachable at {JUDGE_URL}: {e}")

    prompt = (
        "Scrivi delle terzine di endecasillabi in stile dantesco\n\n"
        "Nel mezzo del cammin di nostra vita\n"
        "mi ritrovai per una selva oscura\n"
        "che la diritta via era smarrita\n"
    )

    # 'Good' completion: coherent Italian terzina-like lines (not necessarily perfect meter)
    good = (
        "E quella notte, sotto stella pura,\n"
        "cercai la luce oltre la paura\n"
        "finché la mente mia parve sicura\n"
    )

    # 'Bad' completion: repetitive/English/gibberish lines
    bad = (
        "banana banana banana banana\n"
        "this is not italian poetry\n"
        "lorem ipsum dolor sit amet\n"
    )

    choice, _ = judge_mod.judge_pair(JUDGE_URL, MODEL_ID, prompt, good, bad, max_retries=2, retry_backoff_seconds=0.1)
    assert choice == "A"


