#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Saving helpers for checkpoints to keep dante_grpo.py light and testable.
"""

from __future__ import annotations

import os
from typing import Any


def save_final(output_dir: str, trainer: Any, model: Any, tokenizer: Any) -> None:
    """
    Save the final model and tokenizer under output_dir/final.
    """
    final_dir = os.path.join(output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    try:
        trainer.save_model(final_dir)
    except Exception:
        try:
            if hasattr(model, "save_pretrained"):
                model.save_pretrained(final_dir)
        except Exception:
            pass
    try:
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(final_dir)
    except Exception:
        pass


