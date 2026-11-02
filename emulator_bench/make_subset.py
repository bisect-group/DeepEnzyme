"""Fixed random train subset of a split's features. Val and test are linked unchanged. Usage: make_subset.py SPLIT_FEATURES_DIR OUT_DIR N"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emulator_bench.feature_utils import atomic_save_object_array  # noqa: E402

src, out, n = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
(out / "train").mkdir(parents=True, exist_ok=True)
for name in ("val", "test"):
    if not (out / name).exists():
        (out / name).symlink_to((src / name).resolve())
pick = np.sort(np.random.RandomState(0).choice(len(np.load(src / "train/logkcat.npy", allow_pickle=True)), n, replace=False))
for name in ("fingerprint", "smileadjacencies", "sequences", "proteinadjacencies", "logkcat"):
    column = np.load(src / "train" / f"{name}.npy", allow_pickle=True)
    atomic_save_object_array(out / "train" / f"{name}.npy", [column[i] for i in pick])
summary = json.loads((src / "train/feature_summary.json").read_text())
summary["subset"] = {"n": n, "rng_seed": 0}
(out / "train/feature_summary.json").write_text(json.dumps(summary))
