
#!/usr/bin/env python3
"""
GPU utilization tuning helper for MyXTTS train_main.py.

This script automatically inspects the ArgumentParser defined in train_main.py,
iterates through CLI parameters, and runs training trials while sampling GPU
utilization. It can sweep individual parameters or sample combinations and
reports the configuration that produced the highest average GPU usage.
"""

import argparse
import ast
import json
import os
import random
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


DEFAULT_BASE_ARGS: List[str] = [
    '--train-data', '../dataset/dataset_train',
    '--val-data', '../dataset/dataset_eval',
    '--checkpoint-dir', './checkpointsmain',
]

DEFAULT_EXCLUDED_PARAMS = {'train_data', 'val_data', 'checkpoint_dir', 'multi_gpu', 'enable_eager_debug'}

CONFLICT_GROUPS = [
    {'enable_gpu_stabilizer', 'disable_gpu_stabilizer'},
]

DISABLE_PREFIXES = ('disable_', 'no_')
DISABLE_DEST_OVERRIDES = {
    'simple_loss': {True},
    'use_pretrained_speaker_encoder': {False},
    'enable_gst': {False},
    'enable_gpu_stabilizer': {False},
}
MAX_DISABLE_FLAGS = 1

@dataclass
class ArgInfo:
    dest: str
    flags: List[str] = field(default_factory=list)
    default: Any = None
    arg_type: Optional[str] = None
    choices: Optional[List[Any]] = None
    store_true_flags: List[str] = field(default_factory=list)
    store_false_flags: List[str] = field(default_factory=list)
    help_text: Optional[str] = None


@dataclass
class RunSpec:
    name: str
    parameters: Dict[str, Any]
    cli: List[str]


ARG_PRIORITY: Dict[str, int] = {
    "batch_size": 100,
    "num_workers": 95,
    "prefetch_buffer_size": 90,
    "prefetch_to_gpu": 90,
    "grad_accum": 85,
    "optimization_level": 80,
    "model_size": 75,
    "enable_gpu_stabilizer": 70,
    "disable_gpu_stabilizer": 65,
    "lr": 60,
    "epochs": 55,
    "evaluation_interval": 40,
    "decoder_strategy": 35,
    "vocoder_type": 30,
    "create_optimized_model": 20,
    "enable_evaluation": 20,
    "multi_gpu": 15,
    "visible_gpus": 10,
}


def safe_literal_eval(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def extract_type_name(node: Optional[ast.AST]) -> Optional[str]:
    if node is None:
        return None
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def parse_train_arguments(script_path: str) -> Dict[str, ArgInfo]:
    with open(script_path, "r", encoding="utf-8") as handle:
        source = handle.read()
    tree = ast.parse(source, filename=script_path)
    arg_infos: Dict[str, ArgInfo] = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue

        target = node.func.value
        if not isinstance(target, ast.Name) or target.id != "parser":
            continue

        if node.func.attr == "add_argument":
            flags: List[str] = []
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    flags.append(arg.value)
            if not flags:
                continue

            dest = None
            default = None
            arg_type = None
            choices = None
            action = None
            help_text = None

            for kw in node.keywords:
                if kw.arg == "dest":
                    dest = safe_literal_eval(kw.value)
                elif kw.arg == "default":
                    default = safe_literal_eval(kw.value)
                elif kw.arg == "type":
                    arg_type = extract_type_name(kw.value)
                elif kw.arg == "choices":
                    choices = safe_literal_eval(kw.value)
                elif kw.arg == "action":
                    action = safe_literal_eval(kw.value)
                elif kw.arg == "help":
                    help_text = safe_literal_eval(kw.value)

            if not dest:
                long_flags = [flag for flag in flags if flag.startswith("--")]
                if long_flags:
                    dest = long_flags[0].lstrip("-").replace("-", "_")
                else:
                    dest = flags[0].lstrip("-")

            info = arg_infos.get(dest)
            if not info:
                info = ArgInfo(dest=dest)
                arg_infos[dest] = info

            info.flags.extend(flags)
            if default is not None:
                info.default = default
            if arg_type:
                info.arg_type = arg_type
            if choices is not None:
                info.choices = list(choices)
            if help_text and not info.help_text:
                info.help_text = help_text

            if action == "store_true":
                info.store_true_flags.extend(flags)
                if info.default is None:
                    info.default = False
            elif action == "store_false":
                info.store_false_flags.extend(flags)
                if info.default is None:
                    info.default = True

        elif node.func.attr == "set_defaults":
            for kw in node.keywords:
                dest = kw.arg
                value = safe_literal_eval(kw.value)
                info = arg_infos.get(dest)
                if not info:
                    info = ArgInfo(dest=dest)
                    arg_infos[dest] = info
                info.default = value

    return arg_infos


def select_flag(flags: Sequence[str]) -> Optional[str]:
    if not flags:
        return None
    for flag in flags:
        if flag.startswith("--"):
            return flag
    return flags[0]


def format_cli_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def derive_candidates(info: ArgInfo, limit: Optional[int], include_strings: bool) -> List[Any]:
    candidates: List[Any] = []

    if info.choices:
        candidates = list(info.choices)
    elif info.store_true_flags or info.store_false_flags:
        if info.store_true_flags:
            candidates.append(True)
        if info.store_false_flags:
            candidates.append(False)
        if not info.store_true_flags and not info.store_false_flags and isinstance(info.default, bool):
            candidates = [True, False]
    else:
        inferred_type = info.arg_type
        default = info.default
        if inferred_type is None and isinstance(default, int):
            inferred_type = "int"
        elif inferred_type is None and isinstance(default, float):
            inferred_type = "float"
        elif inferred_type is None and isinstance(default, bool):
            inferred_type = "bool"

        if inferred_type in ("int", "float") and default is not None:
            base_values: List[float] = []
            if default == 0:
                base_values = [0.0, 1.0]
            else:
                factors = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
                base_values = [float(default) * factor for factor in factors]
            if inferred_type == "int":
                adjusted = {max(1, int(round(val))) for val in base_values}
                candidates = sorted(adjusted)
            else:
                precision = {float(f"{val:.6g}") for val in base_values}
                candidates = sorted(precision)
        elif inferred_type == "bool":
            candidates = [True, False]
        elif include_strings and isinstance(default, str) and default:
            candidates = [default]

    unique: List[Any] = []
    for val in candidates:
        if val not in unique:
            unique.append(val)

    if info.default is not None and len(unique) > 1:
        unique = [val for val in unique if val != info.default]

    if limit and limit > 0 and len(unique) > limit:
        unique = unique[:limit]

    return unique


def build_cli_args(info: Optional[ArgInfo], dest: str, value: Any) -> List[str]:
    if info:
        if isinstance(value, bool):
            if value and info.store_true_flags:
                flag = select_flag(info.store_true_flags)
                return [flag] if flag else []
            if not value and info.store_false_flags:
                flag = select_flag(info.store_false_flags)
                return [flag] if flag else []
        flag = select_flag(info.flags)
        if not flag:
            return []
        if isinstance(value, bool) and (info.store_true_flags or info.store_false_flags):
            return []
        if isinstance(value, bool):
            return [flag, format_cli_value(value)]
        return [flag, format_cli_value(value)]

    flag = dest if dest.startswith("--") else f"--{dest.replace('_', '-')}"
    if isinstance(value, bool):
        return [flag] if value else []
    return [flag, format_cli_value(value)]


def build_default_search_space(
    arg_infos: Dict[str, ArgInfo],
    limit_per_argument: Optional[int],
    include_string_choices: bool,
) -> Dict[str, List[Any]]:
    search_space: Dict[str, List[Any]] = {}
    for dest, info in arg_infos.items():
        candidates = derive_candidates(info, limit_per_argument, include_string_choices)
        if candidates:
            search_space[dest] = candidates
    return search_space


class GPUMonitor(threading.Thread):
    def __init__(self, query_fn, interval: float) -> None:
        super().__init__(daemon=True)
        self._query_fn = query_fn
        self._interval = max(0.1, interval)
        self._stop_event = threading.Event()
        self.samples: List[Tuple[float, float, float]] = []

    def run(self) -> None:
        while not self._stop_event.is_set():
            timestamp = time.time()
            try:
                result = self._query_fn()
            except Exception:
                result = None
            if result:
                util, mem = result
                self.samples.append((timestamp, float(util), float(mem)))
            time.sleep(self._interval)

    def stop(self) -> None:
        self._stop_event.set()


def make_gpu_query(gpu_index: int):
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

        def query() -> Optional[Tuple[float, float]]:
            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total = float(mem_info.total) if mem_info.total else 1.0
                mem_percent = (float(mem_info.used) / total) * 100.0
                return float(utilization.gpu), mem_percent
            except Exception:
                return None

        return query
    except Exception:
        def query() -> Optional[Tuple[float, float]]:
            cmd = [
                "nvidia-smi",
                f"-i={gpu_index}",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ]
            try:
                completed = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                line = completed.stdout.strip().split("\n")[0]
                parts = [chunk.strip() for chunk in line.split(",")]
                util = float(parts[0])
                mem_used = float(parts[1])
                mem_total = float(parts[2]) if len(parts) > 2 else 1.0
                mem_percent = (mem_used / mem_total) * 100.0 if mem_total else 0.0
                return util, mem_percent
            except Exception:
                return None

        return query


def summarize_samples(samples: List[Tuple[float, float, float]], start_time: float, warmup: float, min_samples: int) -> Optional[Dict[str, Any]]:
    if not samples:
        return None
    filtered = [s for s in samples if s[0] - start_time >= warmup]
    if not filtered:
        filtered = samples
    if len(filtered) < max(1, min_samples):
        return None
    utils = [s[1] for s in filtered]
    mems = [s[2] for s in filtered]
    duration = filtered[-1][0] - filtered[0][0] if len(filtered) > 1 else 0.0
    return {
        "avg_gpu_util": sum(utils) / len(utils),
        "peak_gpu_util": max(utils),
        "avg_memory_util": sum(mems) / len(mems),
        "peak_memory_util": max(mems),
        "sample_count": len(filtered),
        "monitor_duration": duration,
    }


def execute_run(
    run_id: int,
    run: RunSpec,
    base_command: List[str],
    log_dir: str,
    query_fn,
    max_seconds: Optional[float],
    warmup: float,
    poll_interval: float,
    min_samples: int,
) -> Dict[str, Any]:
    os.makedirs(log_dir, exist_ok=True)
    command = base_command + run.cli
    quoted_cmd = " ".join(shlex.quote(part) for part in command)
    log_path = os.path.join(log_dir, f"run_{run_id:03d}.log")

    summary = " ".join(f"{k}={v}" for k, v in sorted(run.parameters.items()))
    summary = summary or "baseline"
    print(f"[run {run_id:03d}] {run.name} -> {summary}\n  command: {quoted_cmd}")
    start_time = time.time()
    exit_code = None
    timed_out = False
    samples: List[Tuple[float, float, float]] = []

    try:
        with open(log_path, "w", encoding="utf-8") as log_handle:
            log_handle.write(f"# Command: {quoted_cmd}\n")
            log_handle.flush()
            try:
                process = subprocess.Popen(
                    command,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
            except FileNotFoundError as err:
                print(f"  ! Failed to launch: {err}")
                return {
                    "run_id": run_id,
                    "parameters": run.parameters,
                    "cli": run.cli,
                    "command": command,
                    "log_path": log_path,
                    "exit_code": None,
                    "timed_out": False,
                    "duration": 0.0,
                    "metrics": None,
                    "error": str(err),
                }

            monitor = GPUMonitor(query_fn, interval=poll_interval)
            monitor.start()

            try:
                while True:
                    exit_code = process.poll()
                    if exit_code is not None:
                        break
                    elapsed = time.time() - start_time
                    if max_seconds and elapsed > max_seconds:
                        timed_out = True
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                        break
                    time.sleep(1.0)
            finally:
                monitor.stop()
                monitor.join(timeout=5)
                samples = list(monitor.samples)

    finally:
        end_time = time.time()

    metrics = summarize_samples(samples, start_time, warmup, min_samples)
    effective_filtered = [s for s in samples if s[0] - start_time >= warmup]
    effective_count = len(effective_filtered) if effective_filtered else len(samples)
    if metrics is None and effective_count < max(1, min_samples):
        print(f"  ! Insufficient GPU samples collected ({effective_count}/{min_samples}); skipping metrics")
    duration = end_time - start_time

    return {
        "run_id": run_id,
        "parameters": run.parameters,
        "cli": run.cli,
        "command": command,
        "log_path": log_path,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "duration": duration,
        "metrics": metrics,
        "error": None,
    }


def load_user_search_space(path: str) -> Dict[str, List[Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Search space file must be a JSON object mapping dest to list of values")
    normalized: Dict[str, List[Any]] = {}
    for key, value in data.items():
        if not isinstance(value, list):
            raise ValueError(f"Search space for '{key}' must be a list")
        normalized[key] = value
    return normalized


def assemble_cli(info: Optional[ArgInfo], dest: str, value: Any) -> List[str]:
    cli = build_cli_args(info, dest, value)
    if cli:
        return cli
    if info and isinstance(value, bool):
        flag = select_flag(info.flags)
        if flag:
            return [flag, format_cli_value(value)]
    if isinstance(value, bool):
        flag = dest if dest.startswith("--") else f"--{dest.replace('_', '-')}"
        if value:
            return [flag]
        return [flag, "false"]
    flag = dest if dest.startswith("--") else f"--{dest.replace('_', '-')}"
    return [flag, format_cli_value(value)]


def build_runs(
    search_space: Dict[str, List[Any]],
    arg_infos: Dict[str, ArgInfo],
    include: Optional[List[str]],
    exclude: Optional[List[str]],
    max_arguments: Optional[int],
    strategy: str,
    samples: int,
    change_probability: float,
    conflict_groups: Optional[List[set]],
) -> List[RunSpec]:
    include_set = set(include) if include else None
    exclude_set = set(exclude) if exclude else set()

    destinations = [dest for dest, values in search_space.items() if values]
    if include_set is not None:
        destinations = [dest for dest in destinations if dest in include_set]
    if exclude_set:
        destinations = [dest for dest in destinations if dest not in exclude_set]

    destinations.sort(key=lambda d: (-ARG_PRIORITY.get(d, 0), d))
    if max_arguments is not None:
        destinations = destinations[:max_arguments]

    normalized_conflicts = [set(group) for group in (conflict_groups or []) if group]

    def violates_conflict_rules(params: Dict[str, Any]) -> bool:
        for group in normalized_conflicts:
            present = [dest for dest in group if dest in params and bool(params[dest])]
            if len(present) > 1:
                return True
        return False

    def disable_flag_count(params: Dict[str, Any]) -> int:
        count = 0
        for dest, value in params.items():
            if dest in DISABLE_DEST_OVERRIDES:
                disabled_values = DISABLE_DEST_OVERRIDES[dest]
                if value in disabled_values:
                    count += 1
                continue
            if isinstance(value, bool) and value and any(dest.startswith(prefix) for prefix in DISABLE_PREFIXES):
                count += 1
        return count

    runs: List[RunSpec] = []

    if strategy == "single":
        for dest in destinations:
            info = arg_infos.get(dest)
            values = search_space[dest]
            for value in values:
                params = {dest: value}
                if violates_conflict_rules(params):
                    continue
                if disable_flag_count(params) > MAX_DISABLE_FLAGS:
                    continue
                cli = assemble_cli(info, dest, value)
                if not cli:
                    continue
                runs.append(RunSpec(name=f"{dest}={value}", parameters=params, cli=cli))
        return runs

    selectable = [dest for dest in destinations if search_space[dest]]
    if not selectable:
        return runs

    seen: set[Tuple[Tuple[str, Any], ...]] = set()
    max_attempts = max(samples * 10, 50)
    attempts = 0

    while len(runs) < samples and attempts < max_attempts:
        attempts += 1
        params: Dict[str, Any] = {}
        cli: List[str] = []

        shuffled = selectable[:]
        random.shuffle(shuffled)
        for dest in shuffled:
            values = search_space[dest]
            if not values:
                continue
            if random.random() > change_probability:
                continue
            value = random.choice(values)
            info = arg_infos.get(dest)
            args = assemble_cli(info, dest, value)
            if not args:
                continue
            tentative_params = params | {dest: value}
            if violates_conflict_rules(tentative_params):
                continue
            if disable_flag_count(tentative_params) > MAX_DISABLE_FLAGS:
                continue
            params[dest] = value
            cli.extend(args)

        if not params:
            dest = random.choice(selectable)
            values = search_space[dest]
            info = arg_infos.get(dest)
            value = random.choice(values)
            tentative_params = {dest: value}
            if violates_conflict_rules(tentative_params):
                continue
            if disable_flag_count(tentative_params) > MAX_DISABLE_FLAGS:
                continue
            args = assemble_cli(info, dest, value)
            if not args:
                continue
            params[dest] = value
            cli.extend(args)

        key = tuple(sorted(params.items()))
        if not key or key in seen:
            continue
        seen.add(key)
        runs.append(RunSpec(name=f"combo_{len(runs) + 1:03d}", parameters=params, cli=cli))

    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep train_main.py parameters to maximize GPU utilization")
    parser.add_argument("--train-script", default="train_main.py", help="Path to train_main.py")
    parser.add_argument("--python-executable", default=sys.executable, help="Python executable to use")
    parser.add_argument("--gpu-index", type=int, default=0, help="GPU index to monitor")
    parser.add_argument("--max-seconds", type=float, default=300.0, help="Maximum seconds to allow each run")
    parser.add_argument("--warmup-seconds", type=float, default=10.0, help="Seconds to skip at start when averaging GPU utilization")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="GPU polling interval in seconds")
    parser.add_argument("--limit-per-arg", type=int, default=None, help="Maximum candidate values per argument (auto-derived)")
    parser.add_argument("--max-arguments", type=int, default=None, help="Limit the number of arguments considered in the sweep")
    parser.add_argument("--include", nargs="*", help="Only include these argument dest names in the sweep")
    parser.add_argument("--exclude", nargs="*", help="Exclude these argument dest names from the sweep")
    parser.add_argument("--search-space", help="Path to JSON file overriding the auto-generated search space")
    parser.add_argument("--log-dir", default="gpu_sweep_logs", help="Directory to store per-run logs")
    parser.add_argument("--results-path", default="gpu_sweep_results.json", help="Path to write JSON summary")
    parser.add_argument("--no-baseline", action="store_true", help="Skip running the baseline (no extra arguments)")
    parser.add_argument("--include-string-choices", action="store_true", help="Include string defaults without choices in the auto search space")
    parser.add_argument("--dry-run", action="store_true", help="Only print the planned runs without executing them")
    parser.add_argument("--strategy", choices=["single", "combination"], default="combination", help="Sweep strategy: single parameter at a time or sampled combinations")
    parser.add_argument("--samples", type=int, default=20, help="Number of sampled combinations when using combination strategy")
    parser.add_argument("--change-probability", type=float, default=0.5, help="Probability of toggling each argument when sampling combinations")
    parser.add_argument("--min-samples", type=int, default=20, help="Minimum GPU samples required to score a run")

    args = parser.parse_args()

    arg_infos = parse_train_arguments(args.train_script)
    search_space = build_default_search_space(
        arg_infos,
        limit_per_argument=args.limit_per_arg,
        include_string_choices=args.include_string_choices,
    )

    if args.search_space:
        user_space = load_user_search_space(args.search_space)
        for key, values in user_space.items():
            search_space[key] = values

    runs: List[RunSpec] = []
    if not args.no_baseline:
        runs.append(RunSpec(name="baseline", parameters={}, cli=[]))

    auto_exclude = set(DEFAULT_EXCLUDED_PARAMS)
    if args.include:
        auto_exclude -= set(args.include)
    if args.exclude:
        auto_exclude |= set(args.exclude)
    final_exclude = sorted(auto_exclude)

    runs.extend(
        build_runs(
            search_space=search_space,
            arg_infos=arg_infos,
            include=args.include,
            exclude=final_exclude,
            max_arguments=args.max_arguments,
            strategy=args.strategy,
            samples=args.samples,
            change_probability=max(0.0, min(1.0, args.change_probability)),
            conflict_groups=CONFLICT_GROUPS,
        )
    )

    base_command = [args.python_executable, args.train_script]
    base_command.extend(DEFAULT_BASE_ARGS)

    print(f"Base command: {' '.join(shlex.quote(part) for part in base_command)}")
    print(f"Planned runs: {len(runs)}")
    for idx, run in enumerate(runs):
        if run.parameters:
            summary = " ".join(f"{k}={v}" for k, v in sorted(run.parameters.items()))
        else:
            summary = "baseline"
        print(f"  [{idx:02d}] {run.name} -> {summary}")

    if args.dry_run:
        return

    gpu_query = make_gpu_query(args.gpu_index)
    results: List[Dict[str, Any]] = []
    for idx, run in enumerate(runs):
        result = execute_run(
            run_id=idx,
            run=run,
            base_command=base_command,
            log_dir=args.log_dir,
            query_fn=gpu_query,
            max_seconds=args.max_seconds,
            warmup=args.warmup_seconds,
            poll_interval=args.poll_interval,
            min_samples=args.min_samples,
        )
        results.append(result)

    valid_results: List[Dict[str, Any]] = []
    for res in results:
        metrics = res.get("metrics") or {}
        if res.get("timed_out"):
            continue
        if res.get("exit_code") not in (0,):
            continue
        if res.get("error"):
            continue
        if not metrics or metrics.get("avg_gpu_util") is None:
            continue
        if metrics.get("sample_count", 0) < max(1, args.min_samples):
            continue
        valid_results.append(res)

    best = None
    if valid_results:
        best = max(valid_results, key=lambda item: item["metrics"]["avg_gpu_util"])
        best_metric = best["metrics"]
        best_params = " ".join(f"{k}={v}" for k, v in sorted(best["parameters"].items())) or "baseline"
        best_command = " ".join(shlex.quote(part) for part in best["command"])
        print("\nBest configuration:")
        print(f"  Params: {best_params}")
        print(f"  Avg GPU: {best_metric['avg_gpu_util']:.2f}% | Peak GPU: {best_metric['peak_gpu_util']:.2f}%")
        print(f"  Command: {best_command}")
        print(f"  Log file: {best['log_path']}")
    else:
        print("No successful training runs met the GPU sampling threshold; adjust parameters or increase --max-seconds.")

    summary = {
        "train_script": args.train_script,
        "base_command": base_command,
        "runs": results,
        "best_run": best,
    }
    with open(args.results_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nSummary written to {args.results_path}")


if __name__ == "__main__":
    main()
