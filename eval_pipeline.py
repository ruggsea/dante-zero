#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluation pipeline utilities:
- Build and cache a stable evaluation dataset
- Compute endecasillabo accuracy using the naive is_endecasillabo check
- Generate completions for checkpoints (function stub kept lightweight for tests)
"""

from __future__ import annotations

import os
import json
import random
from typing import List, Dict, Any, Optional, Callable

from utils import create_dataset
from simple_endecasillabo_checker import is_endecasillabo


def build_and_cache_eval_prompts(dc_checker, cache_dir: str, sample_size: int = 100, seed: int = 42) -> List[str]:
    """
    Build an evaluation prompt list using the same style as training and cache it as JSONL.
    Returns the list of prompts. If cache exists, loads from cache instead.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "eval_dataset.jsonl")
    if os.path.exists(cache_path):
        prompts: List[str] = []
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                prompts.append(obj["prompt"])
        return prompts

    # Ensure deterministic sampling for eval set
    random.seed(seed)
    ds = create_dataset(sample_size, dc_checker)
    # 'ds' is a HuggingFace Dataset with field 'prompt'
    prompts = list(ds["prompt"]) if isinstance(ds, dict) else [item for item in ds["prompt"]]

    with open(cache_path, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps({"prompt": p}, ensure_ascii=False) + "\n")

    return prompts


def compute_endeca_accuracy_for_completion(text: str) -> float:
    """
    Compute the fraction of non-empty lines in 'text' that are valid endecasillabi
    according to the naive is_endecasillabo check. Returns 0.0 if no lines.
    """
    if not text:
        return 0.0
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return 0.0
    correct = 0
    total = 0
    for ln in lines:
        ok, _ = is_endecasillabo(ln)
        total += 1
        if ok:
            correct += 1
    return float(correct) / float(total) if total > 0 else 0.0


def compute_mean_endeca_accuracy(completions: List[str]) -> float:
    """
    Compute mean endecasillabo accuracy across many completions.
    """
    if not completions:
        return 0.0
    scores = [compute_endeca_accuracy_for_completion(c or "") for c in completions]
    return sum(scores) / len(scores)


def generate_completions_for_prompts(
    model_dir: str,
    prompts: List[str],
    cache_base_dir: str,
    run_name: str,
    checkpoint_name: str,
    generator: Optional[Callable[[List[str]], List[str]]] = None,
    overwrite: bool = False,
) -> str:
    """
    Generate (or load) completions for a list of prompts using a model checkpoint.
    Writes a JSONL file with {prompt, completion} per line and returns its path.

    For tests, pass a lightweight 'generator' function. If None, raises NotImplementedError
    to avoid accidental heavy model loads in unit tests.
    """
    out_dir = os.path.join(cache_base_dir, "eval_completions", run_name, checkpoint_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "completions.jsonl")

    if os.path.exists(out_path) and not overwrite:
        return out_path

    if generator is None:
        # Avoid heavy dependencies by default during tests
        raise NotImplementedError("Default generator not implemented in tests. Provide a 'generator' callable.")

    completions = generator(prompts)
    if len(completions) != len(prompts):
        raise ValueError("Generator must return one completion per prompt")

    with open(out_path, "w", encoding="utf-8") as f:
        for p, c in zip(prompts, completions):
            f.write(json.dumps({"prompt": p, "completion": c}, ensure_ascii=False) + "\n")
    return out_path


