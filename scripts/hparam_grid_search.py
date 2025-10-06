#!/usr/bin/env python3
"""Grid-search helper for train_main.py.

The script runs multiple experiments by sweeping over the parameter grid below,
launching `train_main.py` for each combination, and parsing the resulting log to
collect final metrics (Train Loss, Val Loss, Samples/sec, etc.).  Results are
printed as a table and persisted to JSON for later analysis.

Usage:
    python3 scripts/hparam_grid_search.py

Edit `PARAM_GRID` and `BASE_ARGS` to match the hyper-parameters you want to
explore before running.
"""

import argparse
import itertools
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Any

# --------- User editable section ---------

# Root directory that already contains train_main.py
REPO_ROOT = Path(__file__).resolve().parents[1]

# Base arguments applied to every run (do not include params from the grid)
BASE_ARGS: List[str] = [
    "--optimization-level", "enhanced",
    "--epochs", "200",
]

# Hyper-parameter grid; add/remove keys as needed.
# Supported keys for automatic CLI mapping are listed in PARAM_TO_CLI below.
PARAM_GRID: Dict[str, Iterable[Any]] = {
    "model_size": ["tiny", "small"],
    "batch_size": [16, 32],
    "grad_accum": [2, 4],
    "lr": [5e-5, 8e-5],
    "prefetch_to_gpu": [True, False],
}

# Directory for TensorBoard logs and run logs
OUTPUT_ROOT = REPO_ROOT / "experiments"
LOG_DIR = OUTPUT_ROOT / "logs"
TENSORBOARD_DIR = OUTPUT_ROOT / "tensorboard"
RESULTS_JSON = OUTPUT_ROOT / "grid_results.json"

# --------- Implementation ---------

PARAM_TO_CLI = {
    "model_size": "--model-size",
    "batch_size": "--batch-size",
    "grad_accum": "--grad-accum",
    "lr": "--lr",
    "prefetch_to_gpu": ("--prefetch-to-gpu", "--no-prefetch-to-gpu"),
    "num_workers": "--num-workers",
    "tensorboard_log_dir": "--tensorboard-log-dir",
    "enable_gst": "--enable-gst",
}

FLOAT_FORMAT = "{:.6f}"
TRAIN_LOSS_RE = re.compile(r"Train Loss: ([0-9.]+)")
VAL_LOSS_RE = re.compile(r"Val Loss: ([0-9.]+)")
SAMPLES_RE = re.compile(r"Samples/sec: ([0-9.]+)")
RUN_NAME_RE = re.compile(r"Logging to (.+\.log)")


def _cli_for_param(name: str, value: Any) -> List[str]:
    key = name
    if key not in PARAM_TO_CLI:
        raise ValueError(f"Unsupported parameter '{name}'. Extend PARAM_TO_CLI to map it.")

    mapping = PARAM_TO_CLI[key]
    if isinstance(mapping, tuple):  # bool switch with enable/disable flags
        enable_flag, disable_flag = mapping
        if value is None:
            return []
        return [enable_flag] if value else [disable_flag]

    flag = mapping
    return [flag, str(value)] if value is not None else []


def run_experiment(run_idx: int, params: Dict[str, Any]) -> Tuple[str, Dict[str, float]]:
    run_id = f"grid_{run_idx:03d}"
    run_args = BASE_ARGS.copy()
    tensorboard_path = TENSORBOARD_DIR / run_id

    for name, value in params.items():
        run_args.extend(_cli_for_param(name, value))

    # Always set tensorboard dir + log dir per run
    run_args.extend(["--tensorboard-log-dir", str(tensorboard_path)])

    env = os.environ.copy()
    env["MYXTTS_RUN_NAME"] = run_id
    env["MYXTTS_LOG_DIR"] = str(LOG_DIR)

    cmd = [sys.executable, "train_main.py", *run_args]
    print("\n=== Running", run_id, "===")
    print("Command:", " ".join(cmd))

    start = time.time()
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True)
    elapsed = time.time() - start

    if proc.returncode != 0:
        print("Run failed:", proc.stderr, file=sys.stderr)
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr)

    log_path = None
    for line in proc.stdout.splitlines() + proc.stderr.splitlines():
        m = RUN_NAME_RE.search(line)
        if m:
            log_path = Path(m.group(1))
            if not log_path.is_absolute():
                log_path = LOG_DIR / log_path.name
            break

    if not log_path or not log_path.exists():
        raise FileNotFoundError("Could not determine log file for run", run_id)

    metrics = parse_log(log_path)
    metrics["elapsed_seconds"] = elapsed

    return run_id, metrics


def parse_log(log_path: Path) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "Train Loss:" in line:
                m = TRAIN_LOSS_RE.search(line)
                if m:
                    metrics["train_loss"] = float(m.group(1))
            elif "Val Loss:" in line:
                m = VAL_LOSS_RE.search(line)
                if m:
                    metrics["val_loss"] = float(m.group(1))
            elif "Samples/sec:" in line:
                m = SAMPLES_RE.search(line)
                if m:
                    metrics.setdefault("samples_per_sec", []).append(float(m.group(1)))

    if "samples_per_sec" in metrics:
        samples = metrics["samples_per_sec"]
        metrics["samples_per_sec_mean"] = sum(samples) / len(samples)
        metrics["samples_per_sec_max"] = max(samples)
        metrics.pop("samples_per_sec")

    return metrics


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)

    param_names = list(PARAM_GRID.keys())
    combinations = list(itertools.product(*(PARAM_GRID[name] for name in param_names)))

    results = []
    for idx, combo in enumerate(combinations, 1):
        params = dict(zip(param_names, combo))
        run_id, metrics = run_experiment(idx, params)
        record = {
            "run_id": run_id,
            "params": params,
            "metrics": metrics,
        }
        results.append(record)
        print(f"Result {run_id}: val_loss={metrics.get('val_loss')} train_loss={metrics.get('train_loss')} elapsed={metrics.get('elapsed_seconds'):.1f}s")

    results.sort(key=lambda r: r["metrics"].get("val_loss", float("inf")))

    print("\n=== Summary (sorted by val_loss ascending) ===")
    for record in results:
        run_id = record["run_id"]
        params = record["params"]
        metrics = record["metrics"]
        summary = ", ".join(f"{k}={v}" for k, v in params.items())
        val_loss = metrics.get("val_loss")
        train_loss = metrics.get("train_loss")
        sps = metrics.get("samples_per_sec_mean")
        print(f"{run_id}: val={val_loss:.4f} train={train_loss:.4f} sps≈{sps:.2f} :: {summary}")

    with RESULTS_JSON.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {RESULTS_JSON}")


if __name__ == "__main__":
    main()
