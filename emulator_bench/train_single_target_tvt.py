import argparse
import math
import json
import os
import sys
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
try:
    from src.utils.rich_progress import progress, write
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from src.utils.rich_progress import progress, write

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import atomic_csv, atomic_json, metric_dict, resolve_amp_dtype, resolve_device, set_seed, torch_load
from emulator_bench.feature_utils import load_pickle
from emulator_bench.modeling_fast import DeepEnzymeBench


# The model trains on log2(kcat) as in the original code. Metrics and predictions are in log10 so they compare across baselines.
# "Loss" stays the MSE of the log2 training target.
LOG10_OF_2 = math.log10(2.0)


def load_tensor(file_name, dtype, device):
    np_dtype = np.int64 if dtype is torch.LongTensor else np.float32
    torch_dtype = torch.long if dtype is torch.LongTensor else torch.float32
    return [
        torch.as_tensor(np.asarray(d, dtype=np_dtype), dtype=torch_dtype, device=device)
        for d in np.load(file_name + ".npy", allow_pickle=True)
    ]


def load_split_dataset(split_dir: Path, device):
    split_dir = Path(split_dir)
    fingerprint = load_tensor(str(split_dir / "fingerprint"), torch.LongTensor, device)
    smileadjacencies = load_tensor(str(split_dir / "smileadjacencies"), torch.FloatTensor, device)
    sequences = load_tensor(str(split_dir / "sequences"), torch.LongTensor, device)
    proteinadjacencies = np.load(split_dir / "proteinadjacencies.npy", allow_pickle=True)
    labels = load_tensor(str(split_dir / "logkcat"), torch.FloatTensor, device)
    return list(zip(fingerprint, smileadjacencies, sequences, proteinadjacencies, labels))


def shuffled_once(dataset, seed):
    """The original shuffles the data once (np.random.seed, np.random.shuffle) and trains in that order in every epoch."""
    return [dataset[i] for i in np.random.RandomState(seed).permutation(len(dataset))]


def label_array(dataset):
    """Labels of a dataset as one float64 array, read from the device once."""
    return torch.cat([data[-1] for data in dataset]).cpu().double().numpy()


def feature_summary(split_dir: Path) -> dict:
    return json.loads((Path(split_dir) / "feature_summary.json").read_text())


def build_model(args, dict_dir: Path, device):
    fingerprint_dict = load_pickle(dict_dir / "fingerprint_dict_0612.pickle")
    word_dict = load_pickle(dict_dir / "sequence_dict_0612.pickle")
    model = DeepEnzymeBench(
        len(fingerprint_dict),
        args.dim,
        len(word_dict),
        args.layer_output,
        args.hidden_dim1,
        args.hidden_dim2,
        args.dropout,
        args.nhead,
        args.hid_size,
        args.layers_trans,
    ).to(device)
    return model, len(fingerprint_dict), len(word_dict)


def configure_precision(args, device):
    """Autocast dtype, precision name and GradScaler of a run. The default is fp32 without TF32, as the original code."""
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    if args.fp16:
        amp_dtype, mode = torch.float16, "fp16"
    else:
        amp_dtype, mode = resolve_amp_dtype(device) if args.amp else (None, "fp32")
    if args.tf32:
        mode += "+tf32"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16) if device.type == "cuda" else None
    return amp_dtype, mode, scaler


def make_optimizer(model, args):
    return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=args.fused_adamw or None)


def _autocast_context(device, amp_dtype):
    return torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None)


def run_pass(model, dataset, labels, loss_fn, args, device, amp_dtype, desc, optimizer=None, scaler=None):
    """One pass over the dataset with one sample per step; it trains when an optimizer is given.

    Loss and predictions stay on the device until the pass ends. Returns metrics, labels and predictions in log10.
    """
    iterator = progress(dataset, desc=desc, leave=False, unit="sample", disable=args.hide_sample_progress, dynamic_ncols=True)
    losses, preds = [], []
    squared_error = torch.zeros((), device=device)
    for idx, data in enumerate(iterator, start=1):
        inputs, label = data[:-1], data[-1]
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, amp_dtype):
            output = model(inputs, args.layer_output, args.dropout)
            loss = loss_fn(output.float(), label.float())
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        elif optimizer is not None:
            loss.backward()
            optimizer.step()
        losses.append(loss.detach())
        preds.append(output.detach().float().reshape(()))
        squared_error += (preds[-1] - label.reshape(())) ** 2
        if not args.hide_sample_progress and (idx == 1 or idx % args.progress_interval == 0 or idx == len(dataset)):
            iterator.set_postfix(rmse=f"{(squared_error.item() / idx) ** 0.5:.4g}")
    y_true = labels * LOG10_OF_2
    y_pred = torch.stack(preds).cpu().double().numpy() * LOG10_OF_2
    metrics = metric_dict(y_true, y_pred)
    # A sequential sum of the per-sample float values, like the original loop.
    metrics["Loss"] = sum(torch.stack(losses).cpu().tolist()) / max(1, len(dataset))
    return metrics, y_true, y_pred


def train_epoch(model, dataset, labels, optimizer, loss_fn, args, device, amp_dtype, scaler):
    model.train(True)
    return run_pass(model, dataset, labels, loss_fn, args, device, amp_dtype, "train", optimizer, scaler)[0]


@torch.no_grad()
def evaluate(model, dataset, labels, loss_fn, args, device, amp_dtype, desc):
    model.train(False)
    return run_pass(model, dataset, labels, loss_fn, args, device, amp_dtype, desc)


def _rng_state():
    state = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state):
    if not state:
        return
    torch.set_rng_state(state["torch"].cpu())
    np.random.set_state(state["numpy"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def save_checkpoint(path, epoch, model, optimizer, scheduler, scaler, records, args):
    payload = {
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "records": records,
        "rng_state": _rng_state(),
        "args": vars(args),
    }
    tmp = Path(f"{path}.tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    checkpoint = torch_load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    if scaler is not None and checkpoint.get("scaler_state"):
        scaler.load_state_dict(checkpoint["scaler_state"])
    _set_rng_state(checkpoint.get("rng_state"))
    return int(checkpoint["epoch"]), list(checkpoint.get("records", []))


def write_final_outputs(model, datasets, labels, loss_fn, args, device, amp_dtype, out_dir: Path):
    for split_name, dataset in datasets.items():
        metrics, y_true, y_pred = evaluate(model, dataset, labels[split_name], loss_fn, args, device, amp_dtype, desc=f"final-{split_name}")
        atomic_csv(out_dir / f"final_results_{split_name}.csv", pd.DataFrame([metrics]))
        atomic_csv(out_dir / f"pred_label_{split_name}.csv", pd.DataFrame({"pred": y_pred, "label": y_true}))


def main(args):
    if args.progress_interval < 1:
        raise ValueError("--progress_interval must be >= 1")
    if args.torch_num_threads is not None:
        torch.set_num_threads(args.torch_num_threads)
    if args.torch_num_interop_threads is not None:
        torch.set_num_interop_threads(args.torch_num_interop_threads)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    complete_marker = out_dir / "completed.json"
    if complete_marker.exists() and not args.overwrite:
        print(f"Run already completed: {out_dir}")
        return

    set_seed(args.seed)
    device = resolve_device(args.device)
    amp_dtype, precision_mode, scaler = configure_precision(args, device)
    print(f"Device: {device} | precision: {precision_mode} | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<not-set>')}")

    model, n_fingerprint, n_word = build_model(args, Path(args.dict_dir), device)
    optimizer = make_optimizer(model, args)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)
    loss_fn = nn.MSELoss().to(device)

    print("Loading split feature arrays...")
    datasets = {
        "train": load_split_dataset(Path(args.train_dir), device),
        "val": load_split_dataset(Path(args.val_dir), device),
        "test": load_split_dataset(Path(args.test_dir), device),
    }
    datasets["train"] = shuffled_once(datasets["train"], args.seed)
    labels = {name: label_array(dataset) for name, dataset in datasets.items()}

    last_checkpoint = out_dir / "checkpoint_last.pt"
    start_epoch, records = 0, []
    if last_checkpoint.exists() and not args.overwrite:
        start_epoch, records = load_checkpoint(last_checkpoint, model, optimizer, scheduler, scaler, device)
        print(f"Resumed from epoch {start_epoch}")

    headers = [
        "Epoch",
        "Time(sec)",
        "Loss_train",
        "MAE_train",
        "RMSE_train",
        "R2_train",
        "PCC_train",
        "Loss_val",
        "MAE_val",
        "RMSE_val",
        "R2_val",
        "PCC_val",
        "MAE_test",
        "RMSE_test",
        "R2_test",
        "PCC_test",
        "Lr",
    ]
    started = timeit.default_timer()
    epoch_iter = progress(
        range(start_epoch + 1, args.iteration + 1),
        desc="epochs",
        unit="epoch",
        initial=start_epoch,
        total=args.iteration,
        dynamic_ncols=True,
    )
    for epoch in epoch_iter:
        epoch_iter.set_postfix(phase="train")
        train_metrics = train_epoch(model, datasets["train"], labels["train"], optimizer, loss_fn, args, device, amp_dtype, scaler)
        epoch_iter.set_postfix(train_rmse=f"{train_metrics['RMSE']:.4g}")
        epoch_iter.set_postfix(phase="val")
        val_metrics, _val_y, _val_pred = evaluate(model, datasets["val"], labels["val"], loss_fn, args, device, amp_dtype, desc="val")
        epoch_iter.set_postfix(val_rmse=f"{val_metrics['RMSE']:.4g}")
        epoch_iter.set_postfix(phase="test")
        test_metrics, _test_y, _test_pred = evaluate(model, datasets["test"], labels["test"], loss_fn, args, device, amp_dtype, desc="test")

        if epoch // 10 > 0 and epoch % 10 > 0:
            scheduler.step()

        row = {
            "Epoch": epoch,
            "Time(sec)": round(timeit.default_timer() - started, 4),
            "Loss_train": train_metrics["Loss"],
            "MAE_train": train_metrics["MAE"],
            "RMSE_train": train_metrics["RMSE"],
            "R2_train": train_metrics["R2"],
            "PCC_train": train_metrics["PCC"],
            "Loss_val": val_metrics["Loss"],
            "MAE_val": val_metrics["MAE"],
            "RMSE_val": val_metrics["RMSE"],
            "R2_val": val_metrics["R2"],
            "PCC_val": val_metrics["PCC"],
            "MAE_test": test_metrics["MAE"],
            "RMSE_test": test_metrics["RMSE"],
            "R2_test": test_metrics["R2"],
            "PCC_test": test_metrics["PCC"],
            "Lr": optimizer.param_groups[0]["lr"],
        }
        records.append(row)
        atomic_csv(out_dir / "logfile.csv", pd.DataFrame(records, columns=headers))
        save_checkpoint(last_checkpoint, epoch, model, optimizer, scheduler, scaler, records, args)
        epoch_iter.set_postfix(test_rmse=f"{test_metrics['RMSE']:.4g}", lr=f"{optimizer.param_groups[0]['lr']:.3g}")
        if args.log_json:
            write(json.dumps(row))

    write_final_outputs(model, datasets, labels, loss_fn, args, device, amp_dtype, out_dir)
    summaries = {name: feature_summary(getattr(args, f"{name}_dir")) for name in datasets}
    atomic_json(
        out_dir / "run_summary.json",
        {
            "seed": int(args.seed),
            "train_size": len(datasets["train"]),
            "val_size": len(datasets["val"]),
            "test_size": len(datasets["test"]),
            "final_epoch": int(args.iteration),
            "final_val_rmse": float(records[-1]["RMSE_val"]),
            "rows_dropped": {name: summaries[name]["rows_failed"] for name in datasets},
            "failed_by_stage": {name: summaries[name]["failed_by_stage"] for name in datasets},
            "structure": {name: summaries[name]["structure"] for name in datasets},
            "precision": precision_mode,
            "n_fingerprint": int(n_fingerprint),
            "n_word": int(n_word),
        },
    )
    atomic_json(out_dir / "completed.json", {"completed": True, "final_epoch": int(args.iteration)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resumable DeepEnzyme train/val/test training with original defaults. The final model is the last epoch, as in the original code.")
    parser.add_argument("--train_dir", required=True, type=str)
    parser.add_argument("--val_dir", required=True, type=str)
    parser.add_argument("--test_dir", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--dict_dir", default=str(REPO_ROOT / "Data/Input"), type=str)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument("--seed", default=666, type=int)
    parser.add_argument("--amp", action="store_true", help="Opt in to mixed precision. The original code trains in fp32.")
    parser.add_argument("--fp16", action="store_true", help="fp16 autocast with GradScaler. It replaces the bf16 of --amp.")
    parser.add_argument("--tf32", action="store_true", help="TF32 matrix multiplications in fp32 code.")
    parser.add_argument("--fused_adamw", action="store_true", help="Fused AdamW kernel (CUDA only).")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log_json", action="store_true", help="Print one JSON metrics row per epoch to stdout.")
    parser.add_argument("--show_sample_progress", action="store_true", help="Deprecated compatibility flag; sample progress is shown by default.")
    parser.add_argument("--hide_sample_progress", action="store_true", help="Hide train/val/test per-sample progress bars.")
    parser.add_argument("--progress_interval", type=int, default=1000, help="Sample interval for live RMSE progress updates.")
    parser.add_argument("--torch_num_threads", type=int, default=None, help="Limit PyTorch intra-op CPU threads for this training process.")
    parser.add_argument("--torch_num_interop_threads", type=int, default=None, help="Limit PyTorch inter-op CPU threads for this training process.")

    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--iteration", default=200, type=int)
    parser.add_argument("--weight_decay", default=0.01, type=float, help="torch.optim.AdamW default, which the original code uses.")
    parser.add_argument("--dropout", default=0.3, type=float)
    parser.add_argument("--dim", default=64, type=int)
    parser.add_argument("--layer_output", default=3, type=int)
    parser.add_argument("--hidden_dim1", default=64, type=int)
    parser.add_argument("--hidden_dim2", default=64, type=int)
    parser.add_argument("--nhead", default=4, type=int)
    parser.add_argument("--hid_size", default=64, type=int)
    parser.add_argument("--layers_trans", default=3, type=int)
    main(parser.parse_args())
