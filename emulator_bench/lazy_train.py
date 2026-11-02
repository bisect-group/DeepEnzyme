"""Single-run DeepEnzyme trainer with lazy AdamW for the two word embedding tables (lazy_adamw.py).

Same data, model, loss, schedule, metrics and output files as `train_single_target_tvt.py`. Only the optimizer differs: the
two tables of 137,455 x 64 weights are updated only in the rows that a sample reads, and every row catches up the steps it
skipped before it is read again. The other parameters use torch AdamW (fused on CUDA). The checkpoint holds the fully
caught-up state in torch AdamW format, so a dense run can resume from it and this trainer can resume from a dense checkpoint.
"""

import argparse
import sys
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch import nn, optim

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from emulator_bench import train_single_target_tvt as single  # noqa: E402
from emulator_bench.common import atomic_csv, atomic_json, resolve_device, set_seed, torch_load  # noqa: E402
from emulator_bench.lazy_adamw import LazyEmbedding, LazyRows, Schedule  # noqa: E402

LAZY = ("embed_wordGCN.weight", "embed_wordTrans.weight")
HEADERS = ["Epoch", "Time(sec)"] + [f"{m}_{s}" for s in ("train", "val") for m in ("Loss", "MAE", "RMSE", "R2", "PCC")] + [f"{m}_test" for m in ("MAE", "RMSE", "R2", "PCC")] + ["Lr"]


def with_unique_words(dataset):
    """Adds the unique word ids of each sample and the inverse index to the sample tuple."""
    return [(*data[:3], data[3], *torch.unique(data[2], return_inverse=True), data[4]) for data in dataset]


def normalized_sparse(adjacency, device):
    """D^-1/2 (A + I) D^-1/2 of a contact map as a sparse CSR tensor. The entries are those of the dense `normalize_adj`:
    float32 arithmetic for an integer map, float64 for a float map, cast to float32 at the end."""
    matrix = sp.csr_matrix(adjacency)
    dtype = np.float64 if matrix.dtype == np.float64 else np.float32
    matrix = matrix.astype(dtype) + sp.identity(matrix.shape[0], dtype=dtype, format="csr")
    scale = np.asarray(matrix.sum(1)).ravel() ** -0.5
    rows = np.repeat(np.arange(matrix.shape[0]), np.diff(matrix.indptr))
    values = (matrix.data * scale[rows] * scale[matrix.indices]).astype(np.float32)
    index = torch.from_numpy(np.concatenate((matrix.indptr, matrix.indices)).astype(np.int64)).to(device, non_blocking=True)
    n = matrix.shape[0]
    return torch.sparse_csr_tensor(index[: n + 1], index[n + 1 :], torch.from_numpy(values).to(device, non_blocking=True), size=matrix.shape)


class Stepper:
    """Stands in for model and optimizer in `run_pass`: catches the rows of the sample up, binds them, and updates lazily."""

    def __init__(self, model, regular, lazy, sparse_adjacency=False):
        self.model, self.regular, self.lazy, self.steps, self.sparse_adjacency = model, regular, lazy, 0, sparse_adjacency
        if sparse_adjacency:
            model.protein_gcn = lambda adjacency, words, dropout: torch.sparse.mm(adjacency, model.gcn.layer4.linear(model.embed_wordGCN(words)))
        self.embeddings = [model.embed_wordGCN, model.embed_wordTrans]

    def __call__(self, inputs, layer_output, dropout):
        *inputs, self.rows, inverse = inputs
        # A training forward at step t reads the rows as of step t - 1; an evaluation forward reads them as of the last step.
        self.lazy.advance(self.rows, self.steps)
        for embedding in self.embeddings:
            embedding.bind(self.rows, inverse)
        if self.sparse_adjacency:
            inputs[3] = normalized_sparse(inputs[3], inputs[2].device)
        return self.model(tuple(inputs), layer_output, dropout)

    def zero_grad(self, set_to_none=True):
        self.regular.zero_grad(set_to_none=True)

    def step(self):
        self.steps += 1
        self.lazy.step(self.rows, {name: e.rows_grad for name, e in zip(LAZY, self.embeddings)}, self.steps)
        self.regular.step()


def run(stepper, dataset, labels, loss_fn, args, device, desc, train):
    if train:
        stepper.model.train(True)
        return single.run_pass(stepper, dataset, labels, loss_fn, args, device, None, desc, stepper)
    stepper.model.train(False)
    with torch.no_grad():
        return single.run_pass(stepper, dataset, labels, loss_fn, args, device, None, desc)


def split_optimizer_state(model, state):
    """The torch AdamW state_dict of the whole model split into the part of the regular parameters and the lazy tables."""
    names = [n for n, _ in model.named_parameters()]
    regular = [n for n in names if n not in LAZY]
    position = {n: i for i, n in enumerate(regular)}
    restricted = {
        "state": {position[n]: state["state"][names.index(n)] for n in regular if names.index(n) in state["state"]},
        "param_groups": [{**state["param_groups"][0], "params": list(range(len(regular)))}],
    }
    return restricted, {n: state["state"][names.index(n)] for n in LAZY}


def merged_optimizer_state(model, regular, lazy, steps):
    """The fully caught-up state of the lazy run in the format of a dense AdamW over `model.parameters()`."""
    lazy.finalize(steps)
    params = dict(model.named_parameters())
    names = list(params)
    entries = lazy.state(steps)
    state = {}
    for i, name in enumerate(names):
        if name in LAZY:
            state[i] = entries[name]
        elif params[name] in regular.state and regular.state[params[name]]:
            state[i] = {k: v.detach().cpu().clone() for k, v in regular.state[params[name]].items()}
    group = {k: v for k, v in regular.param_groups[0].items() if k != "params"}
    return {"state": state, "param_groups": [{**group, "params": list(range(len(names)))}]}


def save_checkpoint(path, epoch, model, regular, lazy, scheduler, records, args, steps):
    payload = {
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": merged_optimizer_state(model, regular, lazy, steps),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": None,
        "records": records,
        "rng_state": single._rng_state(),
        "args": vars(args),
    }
    tmp = Path(f"{path}.tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def main(args):
    if args.torch_num_threads is not None:
        torch.set_num_threads(args.torch_num_threads)
    if args.torch_num_interop_threads is not None:
        torch.set_num_interop_threads(args.torch_num_interop_threads)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "completed.json").exists() and not args.overwrite:
        print(f"Run already completed: {out_dir}")
        return

    set_seed(args.seed)
    device = resolve_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"Device: {device} | precision: fp32 + lazy AdamW")
    model, n_fingerprint, n_word = single.build_model(args, Path(args.dict_dir), device)
    model.embed_wordGCN, model.embed_wordTrans = LazyEmbedding(model.embed_wordGCN), LazyEmbedding(model.embed_wordTrans)
    params = dict(model.named_parameters())
    regular = optim.AdamW([p for n, p in params.items() if n not in LAZY], lr=args.lr, weight_decay=args.weight_decay, fused=device.type == "cuda" or None)
    scheduler = optim.lr_scheduler.StepLR(regular, step_size=3, gamma=0.9)
    loss_fn = nn.MSELoss().to(device)

    print("Loading split feature arrays...")
    datasets = {name: with_unique_words(single.load_split_dataset(Path(getattr(args, f"{name}_dir")), device)) for name in ("train", "val", "test")}
    datasets["train"] = single.shuffled_once(datasets["train"], args.seed)
    labels = {name: single.label_array(dataset) for name, dataset in datasets.items()}
    lazy = LazyRows({n: params[n] for n in LAZY}, Schedule(args.lr, args.weight_decay, args.iteration, len(datasets["train"]), device), args.weight_decay)
    stepper = Stepper(model, regular, lazy, args.sparse_adjacency)

    last_checkpoint = out_dir / "checkpoint_last.pt"
    start_epoch, records = 0, []
    if last_checkpoint.exists() and not args.overwrite:
        checkpoint = torch_load(last_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        restricted, tables = split_optimizer_state(model, checkpoint["optimizer_state"])
        regular.load_state_dict(restricted)
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch, records = int(checkpoint["epoch"]), list(checkpoint["records"])
        stepper.steps = start_epoch * len(datasets["train"])
        lazy.load(tables, stepper.steps)
        single._set_rng_state(checkpoint.get("rng_state"))
        print(f"Resumed from epoch {start_epoch}")

    started = timeit.default_timer()
    for epoch in range(start_epoch + 1, args.iteration + 1):
        train = run(stepper, datasets["train"], labels["train"], loss_fn, args, device, "train", True)[0]
        val = run(stepper, datasets["val"], labels["val"], loss_fn, args, device, "val", False)[0]
        test = run(stepper, datasets["test"], labels["test"], loss_fn, args, device, "test", False)[0]
        if epoch // 10 > 0 and epoch % 10 > 0:
            scheduler.step()
        row = {"Epoch": epoch, "Time(sec)": round(timeit.default_timer() - started, 4)}
        row.update({f"{k}_train": train[k] for k in ("Loss", "MAE", "RMSE", "R2", "PCC")})
        row.update({f"{k}_val": val[k] for k in ("Loss", "MAE", "RMSE", "R2", "PCC")})
        row.update({f"{k}_test": test[k] for k in ("MAE", "RMSE", "R2", "PCC")})
        row["Lr"] = regular.param_groups[0]["lr"]
        records.append(row)
        atomic_csv(out_dir / "logfile.csv", pd.DataFrame(records, columns=HEADERS))
        save_checkpoint(last_checkpoint, epoch, model, regular, lazy, scheduler, records, args, stepper.steps)
        print(f"epoch {epoch}/{args.iteration} | train RMSE {train['RMSE']:.4g} | val RMSE {val['RMSE']:.4g} | test RMSE {test['RMSE']:.4g} | lr {row['Lr']:.3g}", flush=True)

    for name, dataset in datasets.items():
        metrics, y_true, y_pred = run(stepper, dataset, labels[name], loss_fn, args, device, f"final-{name}", False)
        atomic_csv(out_dir / f"final_results_{name}.csv", pd.DataFrame([metrics]))
        atomic_csv(out_dir / f"pred_label_{name}.csv", pd.DataFrame({"pred": y_pred, "label": y_true}))
    summaries = {name: single.feature_summary(getattr(args, f"{name}_dir")) for name in datasets}
    atomic_json(
        out_dir / "run_summary.json",
        {
            "seed": int(args.seed),
            **{f"{name}_size": len(datasets[name]) for name in datasets},
            "final_epoch": int(args.iteration),
            "final_val_rmse": float(records[-1]["RMSE_val"]),
            "rows_dropped": {name: summaries[name]["rows_failed"] for name in datasets},
            "failed_by_stage": {name: summaries[name]["failed_by_stage"] for name in datasets},
            "structure": {name: summaries[name]["structure"] for name in datasets},
            "precision": "fp32",
            "n_fingerprint": int(n_fingerprint),
            "n_word": int(n_word),
        },
    )
    atomic_json(out_dir / "completed.json", {"completed": True, "final_epoch": int(args.iteration)})


def parse_args():
    parser = argparse.ArgumentParser(description="DeepEnzyme single run with lazy AdamW for the word embedding tables.")
    for name in ("train_dir", "val_dir", "test_dir", "out_dir"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--dict_dir", default=str(single.REPO_ROOT / "Data/Input"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", default=666, type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sparse_adjacency", action="store_true", help="Protein graph convolution with a sparse normalised contact map (same entries as the dense one).")
    parser.add_argument("--torch_num_threads", type=int, default=None)
    parser.add_argument("--torch_num_interop_threads", type=int, default=None)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--iteration", default=200, type=int)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--dropout", default=0.3, type=float)
    for name, default in (("dim", 64), ("layer_output", 3), ("hidden_dim1", 64), ("hidden_dim2", 64), ("nhead", 4), ("hid_size", 64), ("layers_trans", 3)):
        parser.add_argument(f"--{name}", default=default, type=int)
    # run_pass reads these progress settings
    parser.set_defaults(hide_sample_progress=True, progress_interval=10**9)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
