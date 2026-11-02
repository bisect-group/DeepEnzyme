"""Trains R DeepEnzyme runs in one process. Each run keeps the maths of `train_single_target_tvt.py`.

Every step, run r takes the next sample of its own seeded order from its own split and makes its own batch-1 AdamW update.
The R samples are padded to a common length bucket (batched_model.py) and the step is one CUDA graph per bucket.
Runs with fewer training rows wait (no update) until every run has finished the epoch. Each run evaluates val and test at
the end of its epoch, writes its own checkpoint, log, predictions and completed.json in the layout of the single-run trainer.
Dropout masks come from one RNG stream for the whole batch, not from the stream of a single run.
"""

import argparse
import contextlib
import math
import queue
import threading
import sys
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import optim

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from emulator_bench import train_single_target_tvt as single  # noqa: E402
from emulator_bench.batched_model import forward, trained_names  # noqa: E402
from emulator_bench.common import atomic_csv, atomic_json, metric_dict, set_seed, torch_load  # noqa: E402

L_LADDER = [128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]
A_LADDER = [64, 128, 256, 512, 1024]
CAPACITY = 160  # contacts per residue in the static adjacency buffer; the capacity doubles when a sample needs more
BETAS, EPS = (0.9, 0.999), 1e-8
STAT = ("Loss", "MAE", "RMSE", "R2", "PCC")
HEADERS = ["Epoch", "Time(sec)"] + [f"{m}_train" for m in STAT] + [f"{m}_val" for m in STAT] + [f"{m}_test" for m in STAT[1:]] + ["Lr"]


def bucket(n, ladder):
    return next((size for size in ladder if size >= n), -(-n // 1024) * 1024)


def load_host_split(split_dir):
    """One split as host arrays: per-sample lists and the log2 labels."""
    split_dir = Path(split_dir)

    def load(name):
        return list(np.load(split_dir / f"{name}.npy", allow_pickle=True))

    labels = np.concatenate([np.asarray(d, dtype=np.float32).reshape(-1) for d in load("logkcat")])
    return {"fingerprints": load("fingerprint"), "substrate_adjacency": load("smileadjacencies"), "words": load("sequences"), "protein_adjacency": load("proteinadjacencies"), "labels": labels}


def shape_key(splits, picks):
    """Padded shape of a group's step: (atoms, residues, contact capacity level)."""
    atoms = max(len(s["fingerprints"][i]) for s, i in zip(splits, picks))
    residues = max(len(s["words"][i]) for s, i in zip(splits, picks))
    contacts = sum(s["protein_adjacency"][i].nnz for s, i in zip(splits, picks))
    Ab, Lb, level = bucket(atoms, A_LADDER), bucket(residues, L_LADDER), 0
    while contacts > CAPACITY * Lb * len(splits) << level:
        level += 1
    return Ab, Lb, level


def buffer_shapes(runs, key):
    Ab, Lb, level = key
    cap = CAPACITY * Lb * runs << level
    return {
        "fingerprints": ((runs, Ab), torch.int64),
        "words": ((runs, Lb), torch.int64),
        "substrate_adjacency": ((runs, Ab, Ab), torch.float32),
        "lengths": ((runs, 2), torch.int64),
        "labels": ((runs,), torch.float32),
        "adjacency_index": ((cap,), torch.int64),
        "adjacency_value": ((cap,), torch.float32),
    }


def fill_batch(splits, picks, key, out):
    """Writes the samples `picks` (one index per run of the group) into the arrays `out` of shape `buffer_shapes`."""
    _, Lb, _ = key
    runs = len(splits)
    for name in ("fingerprints", "words", "substrate_adjacency"):
        out[name][:] = 0
    scratch = runs * Lb * Lb  # padding entries of the contact list write 0 to this extra cell
    out["adjacency_index"][:] = scratch
    out["adjacency_value"][:] = 0
    at = 0
    for r, (s, i) in enumerate(zip(splits, picks)):
        fingerprint, atoms, words, contacts = s["fingerprints"][i], s["substrate_adjacency"][i], s["words"][i], s["protein_adjacency"][i]
        a, n = len(fingerprint), len(words)
        out["fingerprints"][r, :a], out["words"][r, :n], out["substrate_adjacency"][r, :a, :a] = fingerprint, words, atoms
        out["lengths"][r], out["labels"][r] = (a, n), s["labels"][i]
        contacts = contacts.tocsr()
        end = at + contacts.nnz
        out["adjacency_index"][at:end] = r * Lb * Lb + np.repeat(np.arange(n, dtype=np.int64), np.diff(contacts.indptr)) * Lb + contacts.indices
        out["adjacency_value"][at:end] = contacts.data
        at = end


class Bucket:
    """Static input buffers of one padded shape and the CUDA graphs that read them."""

    def __init__(self, runs, key, device):
        self.inputs = {name: torch.zeros(shape, dtype=dtype, device=device) for name, (shape, dtype) in buffer_shapes(runs, key).items()}
        self.inputs["found_inf"] = torch.zeros(runs, device=device)
        self.pred = torch.zeros(runs, device=device)
        self.train_graph = self.eval_graph = None


class RunGroup:
    """Stacked parameters, AdamW state and steps of G runs that share a padded batch."""

    def __init__(self, models, args, device, lr, stream=None, pool=None):
        self.args, self.device, self.runs, self.stream, self.pool, self.lr = args, device, len(models), stream, pool, lr
        self.names = trained_names(args.layer_output, args.layers_trans)
        self.hp = {"layers_trans": args.layers_trans, "nhead": args.nhead, "dropout": args.dropout, "layer_output": args.layer_output}
        self.frozen = [{k: v.detach().cpu().clone() for k, v in m.state_dict().items()} for m in models]
        states = [m.state_dict() for m in models]
        self.P = {n: torch.stack([s[n].detach() for s in states]).to(device).requires_grad_(True) for n in self.names}
        self.position0 = models[0].protein_transformer.PositionalEncoding.pe[0].detach().to(device)
        self.exp_avg = {n: torch.zeros_like(p) for n, p in self.P.items()}
        self.exp_avg_sq = {n: torch.zeros_like(p) for n, p in self.P.items()}
        self.steps = {n: [torch.zeros((), device=device) for _ in range(self.runs)] for n in self.names}
        self.buckets = {}

    def update(self, found_inf, active):
        """AdamW with decoupled weight decay for each run. A run with found_inf = 1 keeps its parameters, moments and step."""
        args = self.args
        for r in range(self.runs):
            params, grads = [self.P[n].data[r] for n in self.names], [self.P[n].grad[r] for n in self.names]
            avgs, sqs = [self.exp_avg[n][r] for n in self.names], [self.exp_avg_sq[n][r] for n in self.names]
            steps = [self.steps[n][r] for n in self.names]
            if self.device.type == "cuda":
                torch._foreach_add_(steps, 1)
                torch._fused_adamw_(params, grads, avgs, sqs, [], steps, amsgrad=False, lr=self.lr, beta1=BETAS[0], beta2=BETAS[1], weight_decay=args.weight_decay, eps=EPS, maximize=False, grad_scale=None, found_inf=found_inf[r])
                torch._foreach_sub_(steps, [found_inf[r]] * len(steps))
            elif active[r]:
                lr = float(self.lr)
                for p, g, m, v, s in zip(params, grads, avgs, sqs, steps):
                    s += 1
                    p.mul_(1 - lr * args.weight_decay)
                    m.lerp_(g, 1 - BETAS[0])
                    v.mul_(BETAS[1]).addcmul_(g, g, value=1 - BETAS[1])
                    step = s.item()
                    denom = (v.sqrt() / math.sqrt(1 - BETAS[1] ** step)).add_(EPS)
                    p.addcdiv_(m, denom, value=-lr / (1 - BETAS[0] ** step))

    def step(self, inputs, pred_out, training, active=None):
        """Forward, and when training backward and update. Writes the predictions of the step to `pred_out`."""
        for p in self.P.values():
            p.grad = None
        runs, Ab, Lb = self.runs, inputs["fingerprints"].shape[1], inputs["words"].shape[1]
        lengths = inputs["lengths"]
        batch = {
            "fingerprints": inputs["fingerprints"],
            "words": inputs["words"],
            "substrate_adjacency": inputs["substrate_adjacency"],
            "atom_mask": (torch.arange(Ab, device=self.device)[None] < lengths[:, :1]).float(),
            "residue_mask": (torch.arange(Lb, device=self.device)[None] < lengths[:, 1:]).float(),
        }
        flat = torch.zeros(runs * Lb * Lb + 1, device=self.device)
        flat[inputs["adjacency_index"]] = inputs["adjacency_value"]
        batch["protein_adjacency"] = flat[:-1].view(runs, Lb, Lb)
        with torch.set_grad_enabled(training):
            pred = forward(self.P, batch, self.position0, self.hp, training)
            if training:
                ((pred - inputs["labels"]) ** 2).sum().backward()
                self.update(inputs["found_inf"], active)
        pred_out.copy_(pred.detach())

    def _capture(self, bucket, training):
        """CUDA graph of one step. The warm-up steps use found_inf = 1, so they change nothing."""
        inputs = bucket.inputs
        saved = inputs["found_inf"].clone()
        inputs["found_inf"].fill_(1)
        for _ in range(2):
            self.step(inputs, bucket.pred, training)
        torch.cuda.current_stream().synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool, stream=self.stream, capture_error_mode="thread_local"):
            self.step(inputs, bucket.pred, training)
        inputs["found_inf"].copy_(saved)
        return graph

    def run(self, key, host, active, training):
        """One step of the group on a host batch (arrays of `buffer_shapes`). Returns the predictions [G] on the device."""
        if key not in self.buckets:
            self.buckets[key] = Bucket(self.runs, key, self.device)
        bucket = self.buckets[key]
        for name, array in host.items():
            bucket.inputs[name].copy_(array, non_blocking=True)
        if training:
            bucket.inputs["found_inf"].copy_(torch.from_numpy(~np.asarray(active)).float(), non_blocking=True)
        if self.device.type != "cuda":
            self.step(bucket.inputs, bucket.pred, training, active)
            return bucket.pred
        attr = "train_graph" if training else "eval_graph"
        if getattr(bucket, attr) is None:
            torch.cuda.synchronize()
            setattr(bucket, attr, self._capture(bucket, training))
        getattr(bucket, attr).replay()
        return bucket.pred

    def model_state(self, r):
        state = {k: v.clone() for k, v in self.frozen[r].items()}
        for n in self.names:
            state[n] = self.P[n].detach()[r].cpu()
        return state

    def load_run(self, r, model_state, optimizer_state, names_in_order):
        for n in self.names:
            self.P[n].data[r].copy_(model_state[n])
        for index, n in zip(sorted(optimizer_state["state"]), names_in_order):
            entry = optimizer_state["state"][index]
            self.exp_avg[n][r].copy_(entry["exp_avg"])
            self.exp_avg_sq[n][r].copy_(entry["exp_avg_sq"])
            self.steps[n][r].fill_(float(entry["step"]))


class Job:
    """One run: its split paths, seed and output folder."""

    def __init__(self, train_dir, val_dir, test_dir, out_dir, seed):
        self.dirs = {"train": Path(train_dir), "val": Path(val_dir), "test": Path(test_dir)}
        self.out_dir, self.seed = Path(out_dir), int(seed)


class Runner:
    """All runs of one process: groups, CUDA streams and pinned host buffers."""

    def __init__(self, models, args, device, streams=8):
        self.device, self.args = device, args
        self.lr = torch.tensor(args.lr, device=device)
        cuda = device.type == "cuda"
        self.streams = [torch.cuda.Stream(device) for _ in range(streams)] if cuda else [None]
        pools = [torch.cuda.graph_pool_handle() for _ in self.streams] if cuda else [None]
        size = args.group_size
        self.slices = [slice(i, min(i + size, len(models))) for i in range(0, len(models), size)]
        self.groups = [RunGroup(models[sl], args, device, self.lr, self.streams[g % len(self.streams)], pools[g % len(self.streams)]) for g, sl in enumerate(self.slices)]
        self.runs = len(models)
        self.pinned = {}

    def locate(self, r):
        g = next(g for g, sl in enumerate(self.slices) if sl.start <= r < sl.stop)
        return self.groups[g], r - self.slices[g].start

    def host_buffers(self, slot, g, key, runs):
        """Pinned arrays for the batch of group g in a ring slot."""
        if (slot, g, key) not in self.pinned:
            tensors = {n: torch.zeros(shape, dtype=dtype) for n, (shape, dtype) in buffer_shapes(runs, key).items()}
            self.pinned[slot, g, key] = {n: (t.pin_memory() if self.device.type == "cuda" else t) for n, t in tensors.items()}
        return self.pinned[slot, g, key]

    def batches(self, splits, orders, sizes, steps, slots):
        """Producer: for every step the filled host buffers of the groups that are active in the step."""
        for i in range(steps):
            slot = slots.get()
            work = []
            for g, sl in enumerate(self.slices):
                active = [i < n for n in sizes[sl]]
                if not any(active):
                    continue
                s_g = splits[sl]
                picks = [order[min(i, len(order) - 1)] for order in orders[sl]]
                key = shape_key(s_g, picks)
                bufs = self.host_buffers(slot, g, key, len(s_g))
                fill_batch(s_g, picks, key, {n: t.numpy() for n, t in bufs.items()})
                work.append((g, key, bufs, active))
            yield slot, work

    def pass_over(self, splits, orders, training):
        """One pass over every run's order. Returns the per-run float32 predictions."""
        sizes = [len(o) for o in orders]
        steps = max(sizes)
        ring = 4
        slots = queue.Queue()
        for s in range(ring):
            slots.put(s)
        out = queue.Queue(ring)
        done = object()

        def work():
            try:
                for item in self.batches(splits, orders, sizes, steps, slots):
                    out.put(item)
            finally:
                out.put(done)

        threading.Thread(target=work, daemon=True).start()
        preds = torch.zeros((steps, self.runs), device=self.device)
        for stream in self.streams:
            if stream is not None:
                stream.wait_stream(torch.cuda.current_stream())
        pending = []
        while (item := out.get()) is not done:
            slot, work_items = item
            i = len(pending)
            events = []
            for g, key, bufs, active in work_items:
                group, sl = self.groups[g], self.slices[g]
                stream = group.stream
                with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
                    pred = group.run(key, bufs, active, training)
                    preds[i, sl].copy_(pred)
                    if stream is not None:
                        event = torch.cuda.Event()
                        event.record(stream)
                        events.append(event)
            pending.append((slot, events))
            if len(pending) > 2:
                old_slot, old_events = pending[len(pending) - 3]
                for event in old_events:
                    event.synchronize()
                slots.put(old_slot)
                pending[len(pending) - 3] = (None, [])
        for stream in self.streams:
            if stream is not None:
                stream.synchronize()
        preds = preds.cpu().numpy()
        return [preds[:n, r] for r, n in enumerate(sizes)]


def metrics_of(preds, labels):
    """Metrics in the units of the single-run trainer: loss in log2, the others in log10."""
    squared = (preds - labels) ** 2
    out = metric_dict(labels.astype(np.float64) * single.LOG10_OF_2, preds.astype(np.float64) * single.LOG10_OF_2)
    out["Loss"] = sum(squared.tolist()) / max(1, len(labels))
    return out


def build_runs(jobs, args, device):
    models = []
    for job in jobs:
        set_seed(job.seed)
        models.append(single.build_model(args, Path(args.dict_dir), device)[0].cpu())
    return models


def save_run_checkpoint(group, r, job, epoch, records, scheduler_state, args, model):
    """Per-run checkpoint with the keys of the single-run trainer; the optimizer state is in torch AdamW format."""
    model.load_state_dict(group.model_state(r))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    named = dict(model.named_parameters())
    for n in group.names:
        optimizer.state[named[n]] = {"step": group.steps[n][r].cpu().clone(), "exp_avg": group.exp_avg[n][r].cpu().clone(), "exp_avg_sq": group.exp_avg_sq[n][r].cpu().clone()}
    payload = {
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler_state,
        "scaler_state": None,
        "records": records,
        "rng_state": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
        "args": vars(args),
        "order_names": [n for n, p in model.named_parameters() if optimizer.state.get(p)],
    }
    tmp = Path(f"{job.out_dir / 'checkpoint_last.pt'}.tmp")
    torch.save(payload, tmp)
    tmp.replace(job.out_dir / "checkpoint_last.pt")


def main(args, jobs):
    """Train every job that has no completed.json. Jobs with an unequal resume epoch raise."""
    jobs = [j for j in jobs if args.overwrite or not (j.out_dir / "completed.json").exists()]
    if not jobs:
        print("All runs completed.")
        return
    device = single.resolve_device(args.device)
    for j in jobs:
        j.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | {len(jobs)} runs | precision: fp32")
    torch.backends.cuda.matmul.allow_tf32 = False
    models = build_runs(jobs, args, device)
    state = Runner(models, args, device)
    host = {path: load_host_split(path) for path in {str(j.dirs[name]) for j in jobs for name in j.dirs}}
    data = {name: [host[str(j.dirs[name])] for j in jobs] for name in ("train", "val", "test")}
    orders = {"train": [np.random.RandomState(j.seed).permutation(len(d["labels"])) for j, d in zip(jobs, data["train"])]}
    for name in ("val", "test"):
        orders[name] = [np.arange(len(d["labels"])) for d in data[name]]

    start_epoch, records, lr, scheduler_steps = 0, [[] for _ in jobs], args.lr, 0
    checkpoints = [j.out_dir / "checkpoint_last.pt" for j in jobs]
    if not args.overwrite and all(c.exists() for c in checkpoints):
        loaded = [torch_load(c, map_location="cpu") for c in checkpoints]
        if len({c["epoch"] for c in loaded}) != 1:
            raise ValueError("The runs are at different epochs; resume them separately.")
        for r, c in enumerate(loaded):
            group, local = state.locate(r)
            group.load_run(local, c["model_state"], c["optimizer_state"], c["order_names"])
            records[r] = list(c["records"])
        start_epoch, scheduler_steps = int(loaded[0]["epoch"]), loaded[0]["scheduler_state"]["last_epoch"]
        lr = loaded[0]["optimizer_state"]["param_groups"][0]["lr"]
        rng = loaded[0]["rng_state"]
        torch.set_rng_state(rng["torch"].cpu())
        if rng["cuda"] is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
        print(f"Resumed from epoch {start_epoch}")
    else:
        set_seed(jobs[0].seed)
    state.lr.fill_(lr)
    model_holder = models[0]

    started = timeit.default_timer()
    for epoch in range(start_epoch + 1, args.iteration + 1):
        train_preds = state.pass_over(data["train"], orders["train"], True)
        val_preds = state.pass_over(data["val"], orders["val"], False)
        test_preds = state.pass_over(data["test"], orders["test"], False)
        if epoch // 10 > 0 and epoch % 10 > 0:
            scheduler_steps += 1
            if scheduler_steps % 3 == 0:
                lr = lr * 0.9
        for r, job in enumerate(jobs):
            labels = {name: data[name][r]["labels"][orders[name][r]] for name in orders}
            tm, vm, sm = metrics_of(train_preds[r], labels["train"]), metrics_of(val_preds[r], labels["val"]), metrics_of(test_preds[r], labels["test"])
            row = {"Epoch": epoch, "Time(sec)": round(timeit.default_timer() - started, 4)}
            row.update({f"{k}_train": tm[k] for k in ("Loss", "MAE", "RMSE", "R2", "PCC")})
            row.update({f"{k}_val": vm[k] for k in ("Loss", "MAE", "RMSE", "R2", "PCC")})
            row.update({f"{k}_test": sm[k] for k in ("MAE", "RMSE", "R2", "PCC")})
            row["Lr"] = lr
            records[r].append(row)
            atomic_csv(job.out_dir / "logfile.csv", pd.DataFrame(records[r], columns=HEADERS))
        state.lr.fill_(lr)
        scheduler_state = {"step_size": 3, "gamma": 0.9, "base_lrs": [args.lr], "last_epoch": scheduler_steps, "_step_count": scheduler_steps + 1, "_last_lr": [lr], "_get_lr_called_within_step": False, "_is_initial": False, "verbose": "deprecated"}
        for r, job in enumerate(jobs):
            save_run_checkpoint(*state.locate(r), job, epoch, records[r], scheduler_state, args, model_holder)
        print(f"epoch {epoch}: {timeit.default_timer() - started:.0f} s | val RMSE {[round(rec[-1]['RMSE_val'], 3) for rec in records]}", flush=True)

    final = {"train": state.pass_over(data["train"], orders["train"], False), "val": state.pass_over(data["val"], orders["val"], False), "test": state.pass_over(data["test"], orders["test"], False)}
    for r, job in enumerate(jobs):
        for name in ("train", "val", "test"):
            labels = data[name][r]["labels"][orders[name][r]]
            atomic_csv(job.out_dir / f"final_results_{name}.csv", pd.DataFrame([metrics_of(final[name][r], labels)]))
            atomic_csv(job.out_dir / f"pred_label_{name}.csv", pd.DataFrame({"pred": final[name][r].astype(np.float64) * single.LOG10_OF_2, "label": labels.astype(np.float64) * single.LOG10_OF_2}))
        summaries = {name: single.feature_summary(job.dirs[name]) for name in job.dirs}
        atomic_json(job.out_dir / "run_summary.json", {
            "seed": job.seed,
            **{f"{name}_size": len(data[name][r]["labels"]) for name in ("train", "val", "test")},
            "final_epoch": int(args.iteration),
            "final_val_rmse": float(records[r][-1]["RMSE_val"]),
            "rows_dropped": {name: summaries[name]["rows_failed"] for name in summaries},
            "failed_by_stage": {name: summaries[name]["failed_by_stage"] for name in summaries},
            "structure": {name: summaries[name]["structure"] for name in summaries},
            "precision": "fp32",
            "n_fingerprint": int(state.groups[0].P["embed_fingerprint.weight"].shape[1]),
            "n_word": int(state.groups[0].P["embed_wordGCN.weight"].shape[1]),
        })
        atomic_json(job.out_dir / "completed.json", {"completed": True, "final_epoch": int(args.iteration)})


def parse_args():
    parser = argparse.ArgumentParser(description="Train several DeepEnzyme runs in one process.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dict_dir", default=str(single.REPO_ROOT / "Data/Input"))
    parser.add_argument("--job", nargs=5, action="append", required=True, metavar=("TRAIN_DIR", "VAL_DIR", "TEST_DIR", "OUT_DIR", "SEED"), help="One run. Repeat for every run.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--group_size", default=1, type=int, help="Runs that share one padded batch and one CUDA graph. 1 gives each run its own graph and stream.")
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--iteration", default=200, type=int)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--dropout", default=0.3, type=float)
    for name, default in (("dim", 64), ("layer_output", 3), ("hidden_dim1", 64), ("hidden_dim2", 64), ("nhead", 4), ("hid_size", 64), ("layers_trans", 3)):
        parser.add_argument(f"--{name}", default=default, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args, [Job(*job) for job in args.job])
