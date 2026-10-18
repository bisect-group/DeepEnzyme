import argparse
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import (
    DEFAULT_BASE_DIR,
    DEFAULT_WORK_DIR,
    DEFAULT_SEEDS,
    DEFAULT_SPLIT_GROUPS,
    atomic_csv,
    discover_split_jobs,
    feature_dir,
    normalize_threshold_args,
    result_dir,
)


def _run(cmd, env=None, dry_run=False):
    print(" ".join(map(str, cmd)), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)


def _apply_cpu_thread_env(env, args):
    if args.cpu_threads_per_run is None:
        return env
    threads = str(args.cpu_threads_per_run)
    env["OMP_NUM_THREADS"] = threads
    env["MKL_NUM_THREADS"] = threads
    env["OPENBLAS_NUM_THREADS"] = threads
    env["NUMEXPR_NUM_THREADS"] = threads
    env["VECLIB_MAXIMUM_THREADS"] = threads
    return env


def maybe_prepare(args):
    if args.skip_prepare:
        return
    cmd = [
        sys.executable,
        "emulator_bench/prepare_splits.py",
        "--base_dir",
        str(args.base_dir),
        "--manifests_dir",
        str(args.manifests_dir),
        "--target_col",
        args.source_target_col,
        "--identity_threshold",
        str(args.identity_threshold),
        "--split_groups",
        *args.split_groups,
    ]
    if args.thresholds:
        cmd.extend(["--thresholds", *args.thresholds])
    if args.limit_rows:
        cmd.extend(["--limit_rows", str(args.limit_rows)])
    if args.overwrite_prepare:
        cmd.append("--overwrite")
    _run(cmd, dry_run=args.dry_run)


def maybe_cache(args):
    if args.skip_cache:
        return
    cmd = [
        sys.executable,
        "emulator_bench/cache_features.py",
        "--base_dir",
        str(args.base_dir),
        "--manifests_dir",
        str(args.manifests_dir),
        "--features_dir",
        str(args.features_dir),
        "--cache_dir",
        str(args.cache_dir),
        "--dict_dir",
        str(args.dict_dir),
        "--split_groups",
        *args.split_groups,
    ]
    if args.thresholds:
        cmd.extend(["--thresholds", *args.thresholds])
    if args.limit_rows:
        cmd.extend(["--limit_rows", str(args.limit_rows)])
    if args.overwrite_features:
        cmd.append("--overwrite")
    _run(cmd, dry_run=args.dry_run)


def train_command(args, job, seed):
    train_dir = feature_dir(args.features_dir, job, "train")
    val_dir = feature_dir(args.features_dir, job, "val")
    test_dir = feature_dir(args.features_dir, job, "test")
    out_dir = result_dir(args.results_dir, job, seed)
    cmd = [
        sys.executable,
        "emulator_bench/train_single_target_tvt.py",
        "--train_dir",
        str(train_dir),
        "--val_dir",
        str(val_dir),
        "--test_dir",
        str(test_dir),
        "--out_dir",
        str(out_dir),
        "--dict_dir",
        str(args.dict_dir),
        "--device",
        args.device,
        "--seed",
        str(seed),
    ]
    if args.iteration is not None:
        cmd.extend(["--iteration", str(args.iteration)])
    if args.amp:
        cmd.append("--amp")
    if args.overwrite_runs:
        cmd.append("--overwrite")
    if args.hide_sample_progress:
        cmd.append("--hide_sample_progress")
    cmd.extend(["--progress_interval", str(args.progress_interval)])
    if args.cpu_threads_per_run is not None:
        cmd.extend(["--torch_num_threads", str(args.cpu_threads_per_run)])
    if args.cpu_interop_threads_per_run is not None:
        cmd.extend(["--torch_num_interop_threads", str(args.cpu_interop_threads_per_run)])
    return cmd, out_dir


def _gpu_list(args):
    if args.gpus:
        return [str(gpu) for gpu in args.gpus]
    return [item.strip() for item in str(args.cuda_visible_devices).split(",") if item.strip()]


def run_training_tasks(args, jobs):
    tasks = []
    for job in jobs:
        for seed in args.seeds:
            cmd, out_dir = train_command(args, job, seed)
            complete_marker = out_dir / "completed.json"
            if complete_marker.exists() and not args.overwrite_runs:
                print(f"[skip completed] {out_dir}")
                continue
            tasks.append((job, seed, cmd, out_dir))

    if not tasks:
        return []

    gpus = _gpu_list(args)
    if not gpus:
        gpus = [""]
    if args.runs_per_gpu <= 1 and len(gpus) == 1:
        env = os.environ.copy()
        if gpus[0]:
            env["CUDA_VISIBLE_DEVICES"] = gpus[0]
        env = _apply_cpu_thread_env(env, args)
        print(f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '<not-set>')}")
        if args.cpu_threads_per_run is not None:
            print(f"CPU threads per run: {args.cpu_threads_per_run} | interop={args.cpu_interop_threads_per_run}")
        results = []
        for job, seed, cmd, out_dir in tasks:
            _run(cmd, env=env, dry_run=args.dry_run)
            results.append({"split_group": job["split_group"], "split_name": job["split_name"], "seed": seed, "gpu": gpus[0], "out_dir": str(out_dir), "status": "completed"})
        return results

    work = queue.Queue()
    for task in tasks:
        work.put(task)

    results = []
    result_lock = threading.Lock()

    def worker(gpu_id, slot_index):
        while True:
            try:
                job, seed, cmd, out_dir = work.get_nowait()
            except queue.Empty:
                return
            env = os.environ.copy()
            if gpu_id:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env = _apply_cpu_thread_env(env, args)
            status = "completed"
            try:
                print(
                    f"[launch gpu={gpu_id or '<unset>'} slot={slot_index}] "
                    f"{job['split_group']}/{job['split_name']} seed={seed} -> {out_dir}",
                    flush=True,
                )
                _run(cmd, env=env, dry_run=args.dry_run)
            except Exception as exc:
                status = f"failed: {exc}"
            with result_lock:
                results.append({"split_group": job["split_group"], "split_name": job["split_name"], "seed": seed, "gpu": gpu_id, "slot": slot_index, "out_dir": str(out_dir), "status": status})
            work.task_done()

    threads = []
    for gpu_id in gpus:
        for slot_index in range(args.runs_per_gpu):
            thread = threading.Thread(target=worker, args=(gpu_id, slot_index), daemon=True)
            thread.start()
            threads.append(thread)
    for thread in threads:
        thread.join()

    failed = [row for row in results if str(row["status"]).startswith("failed")]
    if failed and not args.keep_going:
        raise RuntimeError(f"{len(failed)} training task(s) failed.")
    return results


def summarize(results_dir: Path):
    rows = []
    for run_summary in sorted(Path(results_dir).glob("*/*/seed_*/run_summary.json")):
        run_dir = run_summary.parent
        with open(run_summary, "r", encoding="utf-8") as handle:
            summary = json.load(handle)
        row = {
            "split_group": run_dir.parents[1].name,
            "split_name": run_dir.parents[0].name,
            "seed": int(run_dir.name.split("seed_")[-1]),
            "run_dir": str(run_dir),
            **summary,
        }
        for split in ("train", "val", "test"):
            metrics_path = run_dir / f"final_results_{split}.csv"
            if metrics_path.exists():
                metrics = pd.read_csv(metrics_path).iloc[0].to_dict()
                for key, value in metrics.items():
                    row[f"{split}_{key}"] = value
        rows.append(row)
    if not rows:
        return None
    runs = pd.DataFrame(rows)
    atomic_csv(Path(results_dir) / "deepenzyme_summary_runs.csv", runs)
    metric_cols = [c for c in runs.columns if c.startswith("test_")]
    if metric_cols:
        grouped = runs.groupby(["split_group", "split_name"], as_index=False)[metric_cols].agg(["mean", "std"])
        grouped.columns = ["_".join([part for part in col if part]) for col in grouped.columns.to_flat_index()]
        atomic_csv(Path(results_dir) / "deepenzyme_summary_thresholds.csv", grouped.reset_index(drop=True))
    return runs


def main():
    parser = argparse.ArgumentParser(description="Run original-settings DeepEnzyme training across EMULaToR split jobs.")
    parser.add_argument("--base_dir", type=Path, default=DEFAULT_BASE_DIR, help="Task directory, e.g. data/processed/datasets/kinetic_dataset/kcat.")
    parser.add_argument("--work_dir", type=Path, default=DEFAULT_WORK_DIR, help="Output root OUT. Runs go to <work_dir>/<task>/<split_group>/<split_name>/seed_N; the cache goes to <work_dir>/embeddings.")
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--dict_dir", type=Path, default=REPO_ROOT / "Data/Input")
    parser.add_argument("--source_target_col", type=str, default="log10_value")
    parser.add_argument("--identity_threshold", type=float, default=90.0)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cuda_visible_devices", type=str, default="")
    parser.add_argument("--gpus", nargs="+", default=None, help="Physical GPU ids to use for training workers, e.g. --gpus 2 or --gpus 0 1 2.")
    parser.add_argument("--runs_per_gpu", type=int, default=1, help="Concurrent training subprocesses to launch per GPU.")
    parser.add_argument("--cpu_threads_per_run", type=int, default=1, help="CPU BLAS/OpenMP/PyTorch intra-op threads per training subprocess. Use a larger value if CPU is underutilized.")
    parser.add_argument("--cpu_interop_threads_per_run", type=int, default=1, help="PyTorch inter-op CPU threads per training subprocess.")
    parser.add_argument("--iteration", type=int, default=None, help="Smoke-test override of the original 200 epochs.")
    parser.add_argument("--limit_rows", type=int, default=None, help="Limit rows per split for smoke tests.")
    parser.add_argument("--skip_prepare", action="store_true")
    parser.add_argument("--skip_cache", action="store_true")
    parser.add_argument("--overwrite_prepare", action="store_true")
    parser.add_argument("--overwrite_features", action="store_true")
    parser.add_argument("--overwrite_runs", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Opt in to mixed precision. The original code trains in fp32.")
    parser.add_argument("--hide_sample_progress", action="store_true", help="Hide train/val/test sample progress bars in training subprocesses.")
    parser.add_argument("--progress_interval", type=int, default=1000, help="Sample interval for live RMSE progress updates.")
    parser.add_argument("--keep_going", action="store_true", help="Continue other parallel jobs if one training subprocess fails.")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    args.thresholds = normalize_threshold_args(args.thresholds, args.threshold)
    args.base_dir, args.work_dir, args.dict_dir = (Path(os.path.abspath(path)) for path in (args.base_dir, args.work_dir, args.dict_dir))
    args.task_dir = args.work_dir / args.base_dir.name
    args.manifests_dir = args.task_dir / "manifests"
    args.features_dir = args.task_dir / "features"
    args.cache_dir = args.work_dir / "embeddings"
    args.results_dir = args.task_dir

    if args.runs_per_gpu < 1:
        raise ValueError("--runs_per_gpu must be >= 1")
    if args.cpu_threads_per_run is not None and args.cpu_threads_per_run < 1:
        raise ValueError("--cpu_threads_per_run must be >= 1")
    if args.cpu_interop_threads_per_run is not None and args.cpu_interop_threads_per_run < 1:
        raise ValueError("--cpu_interop_threads_per_run must be >= 1")
    if args.progress_interval < 1:
        raise ValueError("--progress_interval must be >= 1")
    gpus = _gpu_list(args)
    print(
        f"Training GPU pool: {gpus} | runs_per_gpu={args.runs_per_gpu} | "
        f"cpu_threads_per_run={args.cpu_threads_per_run} | cpu_interop_threads_per_run={args.cpu_interop_threads_per_run}"
    )

    maybe_prepare(args)
    maybe_cache(args)
    jobs = discover_split_jobs(args.base_dir, args.split_groups, args.thresholds)
    if not jobs:
        raise FileNotFoundError(f"No split jobs found in {args.base_dir}")

    run_training_tasks(args, jobs)
    if not args.dry_run:
        summarize(args.results_dir)


if __name__ == "__main__":
    main()
