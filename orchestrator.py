#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training orchestration utilities for running sweeps and GPU selection.

This module does not launch jobs by itself unless explicitly called.
It provides helpers to:
- Build sweep configurations
- Pick the first free GPU among devices 0..3 using a 1GB threshold
- Construct launch commands for dante_grpo.py
"""

from __future__ import annotations

import os
import json
import subprocess
import gc
import ctypes
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Dict
import argparse
import time
from datetime import datetime

from utils import generate_run_name
from config import compute_output_dir
import glob
import socket


VLLM_REPETITION_PENALTY_GRID: List[float] = [1.0, 1.1, 1.2, 1.3]
KL_BETA_GRID: List[float] = [0.01, 0.02, 0.05, 0.1]
ENDECA_MODES: List[str] = ["top_prob", "unique_only"]


@dataclass
class TrainConfig:
    model_name: str
    use_repetition_reward: bool
    endeca_mode: str
    gen_repetition_penalty: float
    kl_beta: float
    num_epochs: int = 10
    batch_size: int = 64
    gradient_accumulation_steps: int = 2
    num_generations: int = 16
    max_prompt_length: int = 256
    max_completion_length: int = 256
    sample_size: int = 1000


def parse_nvidia_smi_memory_used(output: str) -> List[int]:
    """
    Parse the output of:
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
    into a list of integers (MB used per GPU in index order).
    """
    lines = [ln.strip() for ln in output.strip().splitlines() if ln.strip()]
    result: List[int] = []
    for ln in lines:
        try:
            result.append(int(ln))
        except Exception:
            # Ignore unparsable lines
            pass
    return result


def pick_free_gpu(threshold_mb: int = 1024, candidates: Iterable[int] = (0, 1, 2, 3), memory_used_override: Optional[List[int]] = None) -> Optional[int]:
    """
    Return the first GPU id in candidates whose memory.used is <= threshold_mb.
    If none found, return None.

    For testing, pass memory_used_override as a list of ints representing per-GPU used MB.
    """
    used_list: List[int]
    if memory_used_override is not None:
        used_list = list(memory_used_override)
    else:
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            used_list = parse_nvidia_smi_memory_used(proc.stdout)
        except Exception:
            used_list = []

    # Ensure we have values for indices up to max candidate
    max_index = max(candidates) if candidates else -1
    if len(used_list) <= max_index:
        used_list = used_list + [10**9] * (max_index + 1 - len(used_list))

    for gpu_id in candidates:
        if used_list[gpu_id] <= threshold_mb:
            return gpu_id
    return None


def build_sweep_configs(
    model_name: str,
    repetition_penalties: Optional[List[float]] = None,
    kl_betas: Optional[List[float]] = None,
) -> List[TrainConfig]:
    """
    Build the full grid of training configurations with:
      - use_repetition_reward ∈ {True, False}
      - endeca_mode ∈ {top_prob, unique_only}
      - gen_repetition_penalty ∈ provided or default grid
      - kl_beta ∈ provided or default grid
    """
    rp_grid = list(repetition_penalties or VLLM_REPETITION_PENALTY_GRID)
    kl_grid = list(kl_betas or KL_BETA_GRID)
    configs: List[TrainConfig] = []
    for use_rep in (True, False):
        for endeca_mode in ENDECA_MODES:
            for rp in rp_grid:
                for kl in kl_grid:
                    configs.append(
                        TrainConfig(
                            model_name=model_name,
                            use_repetition_reward=use_rep,
                            endeca_mode=endeca_mode,
                            gen_repetition_penalty=rp,
                            kl_beta=kl,
                        )
                    )
    return configs


def format_run_name(base_run_prefix: str, cfg: TrainConfig) -> str:
    rr = "on" if cfg.use_repetition_reward else "off"
    em = "top" if cfg.endeca_mode == "top_prob" else "uniq"
    rp = f"{cfg.gen_repetition_penalty:.2f}"
    kl = f"{cfg.kl_beta:.2f}"
    return f"{base_run_prefix}-rr_{rr}-emode_{em}-rp_{rp}-kl_{kl}"


def build_training_cmd(script_path: str, cfg: TrainConfig, run_name: Optional[str] = None) -> List[str]:
    """
    Construct a python command list to launch a single training run.
    Does not execute the command.
    """
    cmd = [
        "python",
        script_path,
        "--model_name", cfg.model_name,
        "--num_epochs", str(cfg.num_epochs),
        "--batch_size", str(cfg.batch_size),
        "--gradient_accumulation_steps", str(cfg.gradient_accumulation_steps),
        "--num_generations", str(cfg.num_generations),
        "--max_prompt_length", str(cfg.max_prompt_length),
        "--max_completion_length", str(cfg.max_completion_length),
        "--sample_size", str(cfg.sample_size),
        "--endeca_mode", cfg.endeca_mode,
        "--use_repetition_reward", str(1 if cfg.use_repetition_reward else 0),
        "--gen_repetition_penalty", str(cfg.gen_repetition_penalty),
        "--kl_beta", str(cfg.kl_beta),
    ]
    if run_name:
        cmd.extend(["--run_name", run_name])
    return cmd


MANIFESTS_DIR = os.path.join(os.path.dirname(__file__), "manifests")


def run_sweep(
    model_name: str,
    gpus: List[int],
    threshold_mb: int = 1024,
    repetition_penalties: Optional[List[float]] = None,
    kl_betas: Optional[List[float]] = None,
    base_run_prefix: Optional[str] = None,
    script_path: Optional[str] = None,
    poll_seconds: int = 30,
    manifest_path: Optional[str] = None,
) -> None:
    """
    Launch the full sweep, scheduling one job per available GPU (from gpus list).
    """
    # Enforce GPU ids 0..3 only
    allowed_gpus = [g for g in gpus if 0 <= g <= 3]
    if not allowed_gpus:
        raise ValueError("No valid GPU ids in [0,1,2,3] provided")
    if set(allowed_gpus) != set(gpus):
        print(f"[WARN] Filtering GPU list to 0..3. Using: {allowed_gpus}")
    gpus = allowed_gpus

    cfgs = build_sweep_configs(model_name, repetition_penalties, kl_betas)
    if base_run_prefix is None:
        base_run_prefix = generate_run_name(model_name, None)
    if script_path is None:
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "dante_grpo.py"))
    if manifest_path is None:
        os.makedirs(MANIFESTS_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        manifest_path = os.path.join(MANIFESTS_DIR, f"sweep_manifest_{ts}.jsonl")

    # Build queue while skipping configs that already have outputs
    queue: List[TrainConfig] = []
    for cfg in cfgs:
        rn = format_run_name(base_run_prefix, cfg)
        out_dir = compute_output_dir(rn, None)
        best_dir = os.path.join(out_dir, 'best')
        final_dir = os.path.join(out_dir, 'final')
        if (os.path.isdir(best_dir) and os.listdir(best_dir)) or (os.path.isdir(final_dir) and os.listdir(final_dir)):
            print(f"[SKIP existing] {rn} (found outputs)")
            continue
        queue.append(cfg)
    active: Dict[int, subprocess.Popen] = {}

    # ensure manifest file exists
    open(manifest_path, "a").close()

    def launch_one(cfg: TrainConfig, gpu_id: int) -> None:
        run_name = format_run_name(base_run_prefix, cfg)
        cmd = build_training_cmd(script_path, cfg, run_name=run_name)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # Assign unique torch.distributed master port per run to avoid EADDRINUSE
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", 0))
                port = s.getsockname()[1]
        except Exception:
            # Fallback to a deterministic but spaced port
            port = 12000 + int(time.time()) % 4000
        env["MASTER_ADDR"] = env.get("MASTER_ADDR", "127.0.0.1")
        env["MASTER_PORT"] = str(port)
        # Also set accelerate-compatible var to be safe
        env["ACCELERATE_TORCH_DISTRIBUTED_PORT"] = str(port)
        p = subprocess.Popen(cmd, env=env)
        active[gpu_id] = (p, run_name)
        with open(manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "gpu": gpu_id,
                "run_name": run_name,
                "cmd": cmd,
                "cfg": cfg.__dict__,
                "master_port": port,
                "event": "launch",
                "timestamp": datetime.now().isoformat()
            }) + "\n")
        print(f"[LAUNCH] GPU {gpu_id} → {run_name}")

    print(f"[SWEEP] total configs: {len(queue)} on GPUs {gpus} (threshold {threshold_mb}MB)")
    while queue or active:
        # remove finished
        for g, tup in list(active.items()):
            p, rn = tup
            rc = p.poll()
            if rc is not None:
                print(f"[DONE] GPU {g} {rn} exited with code {rc}")
                try:
                    with open(manifest_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "gpu": g,
                            "run_name": rn,
                            "event": "exit",
                            "returncode": rc,
                            "timestamp": datetime.now().isoformat()
                        }) + "\n")
                except Exception:
                    pass
                # Best-effort host RAM cleanup before scheduling next run on this GPU
                _clean_host_ram_best_effort()
                del active[g]

        # schedule new if queue remains
        if queue:
            free_slots = [g for g in gpus if g not in active]
            if free_slots:
                gpick = pick_free_gpu(threshold_mb=threshold_mb, candidates=free_slots)
                if gpick is not None:
                    cfg = queue.pop(0)
                    launch_one(cfg, gpick)

        if queue or active:
            time.sleep(poll_seconds)

    print("[SWEEP] complete.")


def _parse_float_list(csv: str) -> List[float]:
    return [float(x.strip()) for x in csv.split(',') if x.strip()]


def _parse_int_list(csv: str) -> List[int]:
    return [int(x.strip()) for x in csv.split(',') if x.strip()]


def _clean_host_ram_best_effort() -> None:
    """
    Best-effort host RAM reclamation in the orchestrator process after a run ends.
    This does not require root for gc/malloc_trim/sync. drop_caches may fail without root.
    """
    try:
        gc.collect()
    except Exception:
        pass
    try:
        libc = ctypes.CDLL("libc.so.6")
        try:
            libc.malloc_trim(0)
        except Exception:
            pass
    except Exception:
        pass
    # Flush filesystem buffers
    try:
        subprocess.run(["sync"], check=False)
    except Exception:
        pass
    # Try to drop caches (will usually require root; ignore errors)
    try:
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
    except Exception:
        pass

def _latest_manifest_path() -> Optional[str]:
    os.makedirs(MANIFESTS_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(MANIFESTS_DIR, "*.jsonl")), key=os.path.getmtime, reverse=True)
    return files[0] if files else None


def repair_manifest(manifest_path: Optional[str] = None) -> Optional[str]:
    """
    Scan the manifest and checkpoint folders to append status lines for completed/failed runs.
    - If manifest_path is None, picks the most recent manifest in MANIFESTS_DIR
    - For each run_name seen in the manifest, writes an additional status line:
        * {event: "repaired", status: "completed", source: "checkpoint"} if best/ or final/ exist
        * {event: "repaired", status: "failed"} if a prior exit with non-zero returncode and no outputs
    Returns the manifest path repaired (or None if none found).
    """
    if manifest_path is None:
        manifest_path = _latest_manifest_path()
    if not manifest_path or not os.path.exists(manifest_path):
        print("[REPAIR] No manifest to repair")
        return None

    # Collect run_names and last known exit code
    last_exit: Dict[str, Optional[int]] = {}
    seen_runs: set[str] = set()
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                rn = obj.get('run_name')
                if not rn:
                    continue
                seen_runs.add(rn)
                if obj.get('event') == 'exit':
                    last_exit[rn] = int(obj.get('returncode', 0))
    except Exception:
        pass

    # Append repaired statuses
    appended = 0
    with open(manifest_path, 'a', encoding='utf-8') as out:
        for rn in sorted(seen_runs):
            out_dir = compute_output_dir(rn, None)
            best_dir = os.path.join(out_dir, 'best')
            final_dir = os.path.join(out_dir, 'final')
            has_outputs = (os.path.isdir(best_dir) and os.listdir(best_dir)) or (os.path.isdir(final_dir) and os.listdir(final_dir))
            if has_outputs:
                out.write(json.dumps({
                    "run_name": rn,
                    "event": "repaired",
                    "status": "completed",
                    "source": "checkpoint",
                    "timestamp": datetime.now().isoformat()
                }) + "\n")
                appended += 1
            else:
                rc = last_exit.get(rn, None)
                if rc is not None and rc != 0:
                    out.write(json.dumps({
                        "run_name": rn,
                        "event": "repaired",
                        "status": "failed",
                        "returncode": rc,
                        "timestamp": datetime.now().isoformat()
                    }) + "\n")
                    appended += 1
    print(f"[REPAIR] Updated {appended} statuses in {manifest_path}")
    return manifest_path

def main():
    parser = argparse.ArgumentParser(description="Dante-Zero training orchestrator")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sweep = sub.add_parser("sweep", help="Launch full sweep of training runs")
    sweep.add_argument("--model-name", type=str, default="PleIAs/Pleias-350m-Preview")
    sweep.add_argument("--gpus", type=str, default="0,1,2,3", help="Comma-separated GPU ids")
    sweep.add_argument("--threshold-mb", type=int, default=1024)
    sweep.add_argument("--rp-grid", type=str, default="", help="Optional comma-separated repetition penalties")
    sweep.add_argument("--kl-grid", type=str, default="", help="Optional comma-separated KL betas")
    sweep.add_argument("--run-prefix", type=str, default="", help="Optional custom run name prefix")
    sweep.add_argument("--script-path", type=str, default="", help="Path to dante_grpo.py")
    sweep.add_argument("--manifest", type=str, default="", help="Path to manifest JSONL output")
    sweep.add_argument("--poll-seconds", type=int, default=30)

    repair = sub.add_parser("repair", help="Repair latest or specified manifest by scanning checkpoints")
    repair.add_argument("--manifest", type=str, default="", help="Path to manifest JSONL (default: latest in manifests/")

    args = parser.parse_args()
    if args.cmd == "sweep":
        rps = _parse_float_list(args.rp_grid) if args.rp_grid else None
        kls = _parse_float_list(args.kl_grid) if args.kl_grid else None
        gpus = _parse_int_list(args.gpus)
        run_prefix = args.run_prefix or None
        script = args.script_path or None
        manifest = args.manifest or None
        run_sweep(
            model_name=args.model_name,
            gpus=gpus,
            threshold_mb=args.threshold_mb,
            repetition_penalties=rps,
            kl_betas=kls,
            base_run_prefix=run_prefix,
            script_path=script,
            poll_seconds=args.poll_seconds,
            manifest_path=manifest,
        )
    elif args.cmd == "repair":
        mp = args.manifest or None
        repaired = repair_manifest(mp)
        if repaired:
            print(f"[REPAIR] Manifest repaired: {repaired}")


if __name__ == "__main__":
    main()

