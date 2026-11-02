import json
import pickle
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sparse
import torch
from Bio.SeqUtils import seq3

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from emulator_bench.feature_utils import atomic_save_object_array  # noqa: E402

DICT_DIR = ROOT / "Data" / "Input"
N_FINGERPRINT, N_WORD = 30, 50
DIMS = {"dim": 16, "layer_output": 3, "hidden_dim1": 16, "hidden_dim2": 16, "nhead": 4, "hid_size": 16, "layers_trans": 2}
_orig_device = {"device": "cpu"}


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a CUDA device (run with --gpu)")
    config.addinivalue_line("markers", "noise: production-precision test with a noise-control bound")


@pytest.fixture(scope="session")
def original():
    """The authors' Code.Model and Code.Example modules. Their hardcoded .cuda() calls move to the test device."""
    mp = pytest.MonkeyPatch()
    mp.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self.to(_orig_device["device"]))
    mp.syspath_prepend(str(ROOT / "Code" / "Example"))
    import Code.Model.DeepEnzyme as model
    import Code.Model.GCN as gcn
    import Code.Model.train as train
    import example

    yield Namespace(model=model, gcn=gcn, train=train, example=example, device=_orig_device)
    mp.undo()


@pytest.fixture
def cpu_original(original):
    original.device["device"] = "cpu"
    return original


def write_pdb(path: Path, coords, first_residue=1, chain="A", sequence=None) -> None:
    """A PDB file with N, CA and C atoms for every residue (ALA unless a sequence is given). Coordinates are those of the CA atoms."""
    lines = []
    for i, (x, y, z) in enumerate(coords):
        residue = seq3(sequence[i]).upper() if sequence else "ALA"
        for name in ("N", "CA", "C"):
            lines.append(f"ATOM  {len(lines) + 1:5d}  {name:<3s} {residue} {chain}{first_residue + i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {name[0]}")
    path.write_text("\n".join(lines) + "\nTER\nEND\n")


def random_coords(n, seed, spread=12.0):
    return np.random.RandomState(seed).uniform(-spread, spread, size=(n, 3)).round(3)


def synthetic_rows(n, seed, lengths=(6, 14)):
    """Rows of (fingerprints, substrate adjacency, words, protein adjacency, log2 label) in the dtypes of the real features."""
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n):
        atoms, length = rng.randint(3, 8), rng.randint(*lengths)
        adjacency = (rng.rand(atoms, atoms) < 0.4).astype(np.float32)
        adjacency = np.maximum(adjacency, adjacency.T)
        np.fill_diagonal(adjacency, 0)
        contacts = (rng.rand(length, length) < 0.3).astype(np.int32)
        contacts = np.maximum(contacts, contacts.T)
        if i % 2:
            padded = np.zeros((length + 3, length + 3))
            padded[:length, :length] = contacts
            np.fill_diagonal(padded, 1)
            contacts = padded
        rows.append(
            (
                rng.randint(0, N_FINGERPRINT, atoms).astype(np.int64),
                adjacency,
                rng.randint(0, N_WORD, contacts.shape[0]).astype(np.int64),
                sparse.csr_matrix(contacts),
                np.asarray([rng.uniform(-3, 8)], dtype=np.float32),
            )
        )
    return rows


def write_feature_dir(path: Path, rows) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    for name, column in zip(("fingerprint", "smileadjacencies", "sequences", "proteinadjacencies", "logkcat"), zip(*rows)):
        atomic_save_object_array(path / f"{name}.npy", list(column))
    summary = {"rows_failed": 0, "failed_by_stage": {}, "structure": {"n_rows": len(rows)}}
    (path / "feature_summary.json").write_text(json.dumps(summary))
    return path


def write_tiny_dicts(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    for name, size in (("fingerprint", N_FINGERPRINT), ("sequence", N_WORD)):
        with open(path / f"{name}_dict_0612.pickle", "wb") as handle:
            pickle.dump({i: i for i in range(size)}, handle)
    return path


@pytest.fixture
def toy(tmp_path):
    """Feature folders and a tiny dictionary folder for the training script."""
    data = {"train": synthetic_rows(6, 1), "val": synthetic_rows(3, 2), "test": synthetic_rows(3, 3)}
    dirs = {name: write_feature_dir(tmp_path / "features" / name, rows) for name, rows in data.items()}
    return Namespace(rows=data, dirs=dirs, dict_dir=write_tiny_dicts(tmp_path / "dicts"), root=tmp_path)


def train_args(toy, out_dir, iteration, seed=666, **overrides):
    values = {
        "train_dir": str(toy.dirs["train"]),
        "val_dir": str(toy.dirs["val"]),
        "test_dir": str(toy.dirs["test"]),
        "out_dir": str(out_dir),
        "dict_dir": str(toy.dict_dir),
        "device": "cpu",
        "seed": seed,
        "amp": False,
        "fp16": False,
        "tf32": False,
        "fused_adamw": False,
        "sparse_adjacency": False,
        "overwrite": False,
        "log_json": False,
        "hide_sample_progress": True,
        "progress_interval": 1000,
        "torch_num_threads": None,
        "torch_num_interop_threads": None,
        "lr": 0.001,
        "iteration": iteration,
        "weight_decay": 0.01,
        "dropout": 0.3,
        **DIMS,
    }
    values.update(overrides)
    return Namespace(**values)
