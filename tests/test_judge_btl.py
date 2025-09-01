from judge import parse_judge_choice, build_judge_request_body
import judge as judge_mod
from btl import fit_bradley_terry


def test_parse_judge_choice_various():
    assert parse_judge_choice("... FINAL ANSWER: A") == "A"
    assert parse_judge_choice("final: b") == "B"
    assert parse_judge_choice("I choose A") == "A"
    assert parse_judge_choice("I choose B") == "B"
    assert parse_judge_choice("") is None


def test_fit_bradley_terry_simple():
    # Three models 0,1,2. 0 beats 1 twice, 1 beats 2 twice, 0 beats 2 once.
    mids = ["A", "B", "C"]
    comps = [
        (0, 1, 1), (0, 1, 1),  # A > B
        (1, 2, 1), (1, 2, 1),  # B > C
        (0, 2, 1),             # A > C
    ]
    strengths = fit_bradley_terry(mids, comps, l2_reg=1e-4)
    # Ordering expected: A strongest, then B, then C
    assert strengths["A"] > strengths["B"] > strengths["C"]


def test_judge_pair_retries(monkeypatch):
    calls = {"n": 0}

    def fake_http_json(method, url, body=None, timeout=10.0):
        calls["n"] += 1
        if calls["n"] < 2:
            # first call returns unparseable
            return {"choices": [{"message": {"content": "I cannot decide"}}]}
        else:
            return {"choices": [{"message": {"content": "Reasoning...\n\nFINAL ANSWER: B"}}]}

    monkeypatch.setattr(judge_mod, "_http_json", fake_http_json)
    choice, text = judge_mod.judge_pair(
        "http://127.0.0.1:8008", "model", "P", "A", "B", max_retries=3, retry_backoff_seconds=0.0, max_tokens=256
    )
    assert choice == "B"
    assert calls["n"] == 2


def test_build_judge_request_body_params():
    body = build_judge_request_body("m", "P", "A", "B", max_tokens=777, temperature=0.33)
    assert body["model"] == "m"
    assert body["max_tokens"] == 777
    assert abs(body["temperature"] - 0.33) < 1e-6


