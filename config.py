#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration helpers.
"""

from __future__ import annotations

import os


def compute_output_dir(run_name: str, output_dir_arg: str | None) -> str:
    """
    Compute the output directory using (in order):
    - explicit output_dir_arg if provided
    - CHECKPOINT_LOCATION env var joined with run_name, if present
    - default 'outputs/{run_name}' otherwise
    """
    if output_dir_arg and str(output_dir_arg).strip():
        return output_dir_arg
    base = os.environ.get("CHECKPOINT_LOCATION")
    if base and str(base).strip():
        return os.path.join(base, run_name)
    return f"outputs/{run_name}"



