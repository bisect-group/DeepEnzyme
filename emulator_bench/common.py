import csv
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EMULATOR_ROOT = REPO_ROOT.parents[1]
DEFAULT_BASE_DIR = EMULATOR_ROOT / "data/processed/datasets/kinetic_dataset/kcat"
DEFAULT_WORK_DIR = EMULATOR_ROOT / "outputs/baselines/DeepEnzyme"
DEFAULT_SEEDS = [666, 777, 888]
DEFAULT_SPLIT_GROUPS = [
    "random_splits_grouped_sequence",
    "random_splits_grouped_smiles",
    "uniprot_time_splits",
    "enzyme_sequence_splits",
    "substrate_splits",
]
KEY_COLUMNS = ["smiles", "sequence", "value", "smiles_hash", "uniprot_date", "log10_value"]


def ensure_parent(path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_json(path: Path, payload: Dict) -> None:
    ensure_parent(path)
    tmp = Path(f"{path}.tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    tmp.replace(path)


def load_json(path: Path, default):
    if not Path(path).exists():
        return default
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    ensure_parent(path)
    tmp = Path(f"{path}.tmp.{os.getpid()}")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_table(path: Path, frame: pd.DataFrame) -> None:
    ensure_parent(path)
    tmp = Path(f"{path}.tmp.{os.getpid()}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame.to_parquet(tmp, index=False)
    elif suffix == ".csv":
        frame.to_csv(tmp, index=False)
    else:
        raise ValueError(f"Unsupported table output format: {path}")
    tmp.replace(path)


def append_csv_row(path: Path, row: Dict) -> None:
    ensure_parent(path)
    exists = Path(path).exists()
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def write_table(path: Path, frame: pd.DataFrame) -> None:
    atomic_table(Path(path), frame)


def require_columns(frame: pd.DataFrame, columns: Iterable[str], path: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {path}")


def normalize_sequence(sequence: str) -> str:
    return "".join(str(sequence).strip().upper().split()).replace("*", "")


def split_safe_name(text: str) -> str:
    return str(text).replace("/", "_").replace(" ", "_")


def threshold_sort_value(name: str) -> float:
    try:
        return float(str(name).split("threshold_")[-1])
    except Exception:
        return math.inf


def normalize_threshold_args(thresholds=None, threshold=None) -> Optional[List[str]]:
    out = []
    if thresholds:
        out.extend(str(item) for item in thresholds if str(item).strip())
    if threshold and str(threshold).strip():
        out.append(str(threshold))
    if not out:
        return None
    seen = set()
    deduped = []
    for item in out:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def _find_split_file(root: Path, split: str) -> Optional[Path]:
    for suffix in (".parquet", ".csv"):
        candidate = root / f"{split}{suffix}"
        if candidate.exists():
            return candidate
    return None


def discover_split_jobs(base_dir: Path, split_groups=None, thresholds=None) -> List[Dict]:
    base_dir = Path(base_dir)
    split_groups = list(split_groups or DEFAULT_SPLIT_GROUPS)
    threshold_filter = set(thresholds) if thresholds else None
    jobs = []
    for split_group in split_groups:
        group_dir = base_dir / split_group
        if not group_dir.exists():
            continue

        direct = {name: _find_split_file(group_dir, name) for name in ("train", "val", "test")}
        if all(direct.values()):
            jobs.append(
                {
                    "split_group": split_group,
                    "split_name": split_group,
                    "root_dir": str(group_dir),
                    "train_path": str(direct["train"]),
                    "val_path": str(direct["val"]),
                    "test_path": str(direct["test"]),
                }
            )
            continue

        children = [p for p in group_dir.iterdir() if p.is_dir()]
        if threshold_filter is not None:
            children = [p for p in children if p.name in threshold_filter]
        children = sorted(children, key=lambda p: (threshold_sort_value(p.name), p.name))
        for child in children:
            split_paths = {name: _find_split_file(child, name) for name in ("train", "val", "test")}
            if not all(split_paths.values()):
                continue
            jobs.append(
                {
                    "split_group": split_group,
                    "split_name": child.name,
                    "root_dir": str(child),
                    "train_path": str(split_paths["train"]),
                    "val_path": str(split_paths["val"]),
                    "test_path": str(split_paths["test"]),
                }
            )
    return jobs


def manifest_path(manifests_dir: Path, job: Dict, split: str) -> Path:
    return Path(manifests_dir) / split_safe_name(job["split_group"]) / split_safe_name(job["split_name"]) / f"{split}.parquet"


def feature_dir(features_dir: Path, job: Dict, split: str) -> Path:
    return Path(features_dir) / split_safe_name(job["split_group"]) / split_safe_name(job["split_name"]) / split


def result_dir(results_dir: Path, job: Dict, seed: int) -> Path:
    return Path(results_dir) / split_safe_name(job["split_group"]) / split_safe_name(job["split_name"]) / f"seed_{seed}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_amp_dtype(device: torch.device):
    if device.type != "cuda":
        return None, "fp32"
    major, _minor = torch.cuda.get_device_capability(device)
    if major >= 8:
        return torch.bfloat16, "bf16"
    return torch.float16, "fp16"


def torch_load(path: Path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def metric_dict(y_true, y_pred) -> Dict[str, float]:
    from scipy.stats import pearsonr
    from sklearn.metrics import mean_squared_error, r2_score

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mae = float(np.mean(np.abs(y_pred - y_true))) if len(y_true) else float("nan")
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred))) if len(y_true) else float("nan")
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan")
    if len(y_true) >= 2 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        pcc = float(pearsonr(y_true, y_pred)[0])
    else:
        pcc = float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "PCC": pcc}

