#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Judge utilities to compare completions via an OpenAI-compatible endpoint (e.g., vLLM on localhost).

Provides:
- list_models(judge_url)
- judge_pair(judge_url, model_name, prompt, completion_a, completion_b, ...)
- parse_judge_choice(text)
- build_judge_request_body(...)
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Dict, Any, Optional, Tuple
import time
import re


def _http_json(method: str, url: str, body: Optional[Dict[str, Any]] = None, timeout: float = 10.0) -> Dict[str, Any]:
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url=url, method=method)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, data=data, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} when calling {url}: {e.read().decode('utf-8', 'ignore')}")
    except Exception as e:
        raise RuntimeError(f"Error calling {url}: {e}")


def list_models(judge_url: str) -> Dict[str, Any]:
    """Return the OpenAI-compatible models listing."""
    return _http_json("GET", f"{judge_url}/v1/models")


def parse_judge_choice(text: str) -> Optional[str]:
    """
    Parse the judge's final choice from response text.
    Returns 'A' or 'B' or None if not found.
    """
    if not text:
        return None
    # Try JSON extraction first
    try:
        obj = json.loads(text)
        ch = str(obj.get("choice", "")).strip().upper()
        if ch in ("A", "B"):
            return ch
    except Exception:
        pass
    lower = text.strip().lower()

    # 1) Robust regex for "final answer: A/B" in multiple languages/punctuations
    patterns = [
        r"final\s*answer\s*[:\-]?\s*([ab])",
        r"final\s*choice\s*(?:is|=)?\s*([ab])",
        r"final\s*decision\s*(?:is|=)?\s*([ab])",
        r"final\s*[:\-]?\s*([ab])",
        r"finale\s*[:\-]?\s*([ab])",               # 'Finale: A'
        r"scelta\s*finale\s*[:\-]?\s*([ab])",
        r"risposta\s*finale\s*[:\-]?\s*([ab])",
        r"risposta\s*[:\-]?\s*([ab])",
        r"scelta\s*[:\-]?\s*([ab])",
    ]
    for pat in patterns:
        m = re.search(pat, lower)
        if m:
            ch = m.group(1)
            return "A" if ch == "a" else "B"

    # 2) Imperative phrases indicating choice
    phrase_map = [
        (r"\bi\s*choose\s*a\b", "A"),
        (r"\bi\s*choose\s*b\b", "B"),
        (r"\bchoose\s*a\b", "A"),
        (r"\bchoose\s*b\b", "B"),
        (r"\bscelgo\s*a\b", "A"),
        (r"\bscelgo\s*b\b", "B"),
        (r"\bpreferisco\s*a\b", "A"),
        (r"\bpreferisco\s*b\b", "B"),
        (r"\bi\s*prefer\s*a\b", "A"),
        (r"\bi\s*prefer\s*b\b", "B"),
        (r"\bprefer\s*a\b", "A"),
        (r"\bprefer\s*b\b", "B"),
        (r"\bi\s*pick\s*a\b", "A"),
        (r"\bi\s*pick\s*b\b", "B"),
        (r"\bpick\s*a\b", "A"),
        (r"\bpick\s*b\b", "B"),
        (r"\bselect\s*a\b", "A"),
        (r"\bselect\s*b\b", "B"),
        (r"\bvote\s*for\s*a\b", "A"),
        (r"\bvote\s*for\s*b\b", "B"),
        (r"\boption\s*a\s*(?:wins|is\s*better)\b", "A"),
        (r"\boption\s*b\s*(?:wins|is\s*better)\b", "B"),
        (r"\b(a)\s+(?:is|seems|looks)\s+(?:better|preferable)\b", "A"),
        (r"\b(b)\s+(?:is|seems|looks)\s+(?:better|preferable)\b", "B"),
    ]
    for pat, label in phrase_map:
        if re.search(pat, lower):
            return label

    # 3) Look for a last-line standalone A/B (with punctuation)
    lines = [ln.strip() for ln in lower.splitlines() if ln.strip()]
    if lines:
        last = lines[-1]
        if re.fullmatch(r"\(?\s*a\s*[\).!]*", last):
            return "A"
        if re.fullmatch(r"\(?\s*b\s*[\).!]*", last):
            return "B"

    return None


def build_judge_request_body(
    model_name: str,
    prompt: str,
    completion_a: str,
    completion_b: str,
    max_tokens: int = 16384,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """
    Build the chat.completions request body for the judge call.
    Separated for testability.
    """
    system_prompt = (
        "You are a literary critic judging Italian poetry in Dante's terza rima. "
        "Compare two completions A and B for Dante-style quality, coherence, meter adherence, and originality. "
        "Think step by step, then in the final line write: 'FINAL ANSWER: A' or 'FINAL ANSWER: B'."
    )
    user_content = (
        f"Prompt:\n{prompt}\n\n"
        f"Completion A:\n{completion_a}\n\n"
        f"Completion B:\n{completion_b}\n\n"
        "Choose the better completion."
    )
    return {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }


def judge_pair(
    judge_url: str,
    model_name: str,
    prompt: str,
    completion_a: str,
    completion_b: str,
    timeout: float = 60.0,
    max_retries: int = 3,
    retry_backoff_seconds: float = 1.0,
    max_tokens: int = 16384,
    temperature: float = 0.2,
) -> Tuple[Optional[str], str]:
    """
    Use an OpenAI-compatible chat API to judge which completion is better.
    Returns (choice, raw_text). 'choice' is 'A' or 'B' or None if undecided.
    """
    base_body = build_judge_request_body(
        model_name=model_name,
        prompt=prompt,
        completion_a=completion_a,
        completion_b=completion_b,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    last_text = ""
    for attempt in range(max_retries):
        # Escalate constraints after first failure: require "A" or "B" only
        if attempt == 0:
            body = dict(base_body)
        else:
            body = dict(base_body)
            msgs = list(body.get("messages", []))
            msgs.append({
                "role": "user",
                "content": "Restituisci esattamente un JSON senza testo aggiuntivo: {\"choice\":\"A\"} oppure {\"choice\":\"B\"}.",
            })
            body["messages"] = msgs
            body["response_format"] = {"type": "json_object"}
            body["temperature"] = 0.0
            body["max_tokens"] = max_tokens
        try:
            data = _http_json("POST", f"{judge_url}/v1/chat/completions", body=body, timeout=timeout)
            try:
                text = data["choices"][0]["message"]["content"]
            except Exception:
                text = json.dumps(data)
            choice = parse_judge_choice(text)
            if choice in ("A", "B"):
                return choice, text
            last_text = text
        except Exception as _e:
            # ignore and retry
            pass
        # backoff before next attempt
        if attempt < max_retries - 1:
            try:
                time.sleep(retry_backoff_seconds)
            except Exception:
                pass
    # Give best-effort last_text if available
    return parse_judge_choice(last_text), last_text


