"""DeepEnzyme with real minibatches of B samples and shared weights, to measure the deviation from the original batch-1 protocol.

Everything else follows `train_single_target_tvt.py`: AdamW, StepLR(3, 0.9) stepped as in the original loop, dropout, one seeded shuffle
reused in every epoch, per-epoch val and test evaluation, `logfile.csv` format, last-epoch model. Batches are consecutive chunks of that order.
The loss is the mean squared error over the batch.

Padding is inert. The two GCNs run on the concatenated nodes of the batch with a block-diagonal sparse adjacency (no padded nodes). The
protein transformer pads residues and masks the padded keys. Every pooling is a mean over the real positions of its own sample.
Layers have no batch statistics, so a batch can be split into micro-batches of similar length (`--max_tokens`) with summed gradients.
That only bounds memory and padding; the update is that of the whole batch.
"""

import argparse
import sys
import timeit
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch import nn, optim

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from emulator_bench import train_single_target_tvt as single  # noqa: E402
from emulator_bench.common import atomic_csv, atomic_json, metric_dict, set_seed, torch_load  # noqa: E402

LOG10_OF_2 = single.LOG10_OF_2
STAT = ("Loss", "MAE", "RMSE", "R2", "PCC")
HEADERS = ["Epoch", "Time(sec)"] + [f"{m}_train" for m in STAT] + [f"{m}_val" for m in STAT] + [f"{m}_test" for m in STAT[1:]] + ["Lr"]


class Split:
    """One split as host lists. Substrate and protein graphs are CSR matrices; labels are log2(kcat)."""

    def __init__(self, fingerprints, substrate, words, protein, labels):
        self.fingerprints, self.substrate, self.words, self.protein, self.labels = fingerprints, substrate, words, protein, labels
        self.lengths = np.array([len(w) for w in words])

    def __len__(self):
        return len(self.words)

    def take(self, order):
        return Split(*([column[i] for i in order] for column in (self.fingerprints, self.substrate, self.words, self.protein)), self.labels[order])


def load_split(split_dir) -> Split:
    split_dir = Path(split_dir)

    def load(name):
        return list(np.load(split_dir / f"{name}.npy", allow_pickle=True))

    labels = np.concatenate([np.asarray(d, dtype=np.float32).reshape(-1) for d in load("logkcat")])
    return Split(
        [np.asarray(d, dtype=np.int64) for d in load("fingerprint")],
        [sparse.csr_matrix(np.asarray(d, dtype=np.float32)) for d in load("smileadjacencies")],
        [np.asarray(d, dtype=np.int64) for d in load("sequences")],
        [d.tocsr() for d in load("proteinadjacencies")],
        labels,
    )


def concat_csr(matrices, device):
    """Block-diagonal CSR matrix of the samples as (crow, col, val, row) tensors."""
    sizes = np.array([m.shape[0] for m in matrices])
    node_offset = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    nnz_offset = np.concatenate([[0], np.cumsum([m.nnz for m in matrices])])
    crow = np.concatenate([m.indptr[:-1] + o for m, o in zip(matrices, nnz_offset)] + [nnz_offset[-1:]])
    col = np.concatenate([m.indices + o for m, o in zip(matrices, node_offset)])
    val = np.concatenate([m.data for m in matrices]).astype(np.float32)
    return tuple(torch.from_numpy(a).to(device) for a in (crow.astype(np.int64), col.astype(np.int64), val))


def collate(split, idx, device):
    """Device tensors of the samples `idx`: flat atoms and residues, padded words, and both block-diagonal graphs."""
    atoms = np.array([len(split.fingerprints[i]) for i in idx])
    lengths = split.lengths[idx]
    words = np.zeros((len(idx), lengths.max()), dtype=np.int64)
    for r, i in enumerate(idx):
        words[r, : lengths[r]] = split.words[i]
    to = lambda a: torch.from_numpy(a).to(device)  # noqa: E731
    return {
        "fingerprints": to(np.concatenate([split.fingerprints[i] for i in idx])),
        "atoms": to(atoms),
        "substrate": concat_csr([split.substrate[i] for i in idx], device),
        "words": to(words),
        "lengths": to(lengths),
        "flat_words": to(np.concatenate([split.words[i] for i in idx])),
        "protein": concat_csr([split.protein[i] for i in idx], device),
        "labels": to(split.labels[idx]),
    }


def graph_conv(x, graph):
    """The original GCN layer after its linear map: D^-1/2 (A + I) D^-1/2 x on the block-diagonal graph."""
    crow, col, val = graph
    row = torch.repeat_interleave(torch.arange(len(crow) - 1, device=x.device), crow[1:] - crow[:-1])
    scale = (torch.zeros(len(crow) - 1, device=x.device).index_add_(0, row, val) + 1).pow(-0.5)
    adjacency = torch.sparse_csr_tensor(crow, col, val * scale[row] * scale[col], size=(len(crow) - 1,) * 2)
    return torch.sparse.mm(adjacency, x) + (scale * scale)[:, None] * x


def segment_mean(x, sizes):
    ids = torch.repeat_interleave(torch.arange(len(sizes), device=x.device), sizes)
    return torch.zeros(len(sizes), x.shape[1], device=x.device).index_add_(0, ids, x) / sizes[:, None]


def forward(model, mb, layer_output, dropout):
    """Predictions [B] in log2 units. Same maths as `DeepEnzymeBench.forward` per sample."""
    gcn_linear = model.gcn.layer4.linear
    substrate = segment_mean(graph_conv(gcn_linear(model.embed_fingerprint(mb["fingerprints"])), mb["substrate"]), mb["atoms"])

    transformer = model.protein_transformer
    pad = torch.arange(mb["words"].shape[1], device=mb["words"].device)[None] >= mb["lengths"][:, None]
    x = model.embed_wordTrans(mb["words"]) + transformer.PositionalEncoding.pe[0]  # the original adds row 0 to every residue
    x = transformer.transformer_encoder(x.transpose(0, 1), src_key_padding_mask=pad).transpose(0, 1)
    x = transformer.linear(x) * (~pad)[:, :, None]
    sequence = x.sum(1) / mb["lengths"][:, None]

    protein = segment_mean(graph_conv(gcn_linear(model.embed_wordGCN(mb["flat_words"])), mb["protein"]), mb["lengths"])

    x = torch.cat((substrate, protein, sequence), 1)
    for j in range(layer_output):
        x = nn.functional.dropout(nn.functional.relu(x), dropout, training=model.training)
        x = model.W_out[j](x)
    x = nn.functional.dropout(nn.functional.relu(x), dropout, training=model.training)
    return model.W_interaction(x).squeeze(1)


def micro_batches(lengths, positions, max_tokens):
    """Splits a batch into groups of similar length whose padded size stays within `max_tokens` residues."""
    order = positions[np.argsort(-lengths[positions], kind="stable")]
    groups, start = [], 0
    while start < len(order):
        end = start + 1
        while end < len(order) and (end - start + 1) * lengths[order[start]] <= max_tokens:
            end += 1
        groups.append(order[start:end])
        start = end
    return groups


def prefetched(split, jobs, device, max_tokens):
    """Yields (micro_groups, device batches) for each job, preparing the next job in a thread while the GPU works."""
    def prepare(positions):
        groups = micro_batches(split.lengths, positions, max_tokens)
        return groups, [collate(split, g, device) for g in groups]

    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(prepare, jobs[0])
        for k in range(len(jobs)):
            result = future.result()
            if k + 1 < len(jobs):
                future = pool.submit(prepare, jobs[k + 1])
            yield result


def train_epoch(model, split, optimizer, args, device):
    """One epoch over consecutive batches of `split`. Returns the log10 metrics, the log2 per-sample training predictions and the per-step losses."""
    model.train(True)
    n, B = len(split), args.batch_size
    jobs = [np.arange(s, min(s + B, n)) for s in range(0, n, B)]
    preds, step_losses = torch.zeros(n, device=device), []
    for positions, (groups, batches) in zip(jobs, prefetched(split, jobs, device, args.max_tokens)):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for g, mb in zip(groups, batches):
            output = forward(model, mb, args.layer_output, args.dropout)
            loss = ((output - mb["labels"]) ** 2).sum() / len(positions)
            loss.backward()
            preds[torch.from_numpy(g).to(device)] = output.detach()
            step_loss += loss.detach()
        optimizer.step()
        step_losses.append(step_loss)
    preds = preds.cpu().double().numpy()
    return summarize(split.labels, preds), preds, torch.stack(step_losses).cpu().numpy()


@torch.no_grad()
def evaluate(model, split, args, device):
    """Eval-mode pass over one split in length-sorted groups. Returns log10 metrics and the log2 predictions in the order of the split."""
    model.train(False)
    n = len(split)
    order = np.argsort(split.lengths, kind="stable")
    jobs = [order[s : s + args.eval_batch_size] for s in range(0, n, args.eval_batch_size)]
    preds = torch.zeros(n, device=device)
    for _, (groups, batches) in zip(jobs, prefetched(split, jobs, device, args.max_tokens)):
        for g, mb in zip(groups, batches):
            preds[torch.from_numpy(g).to(device)] = forward(model, mb, args.layer_output, args.dropout)
    preds = preds.cpu().double().numpy()
    return summarize(split.labels, preds), preds


def summarize(labels, preds):
    """Metrics in log10 as the original wrapper. Loss is the mean squared error of the log2 target."""
    labels = labels.astype(np.float64)
    metrics = metric_dict(labels * LOG10_OF_2, preds * LOG10_OF_2)
    metrics["Loss"] = float(np.mean((preds - labels) ** 2))
    return metrics


def write_final_outputs(model, splits, args, device, out_dir):
    for name, split in splits.items():
        metrics, preds = evaluate(model, split, args, device)
        atomic_csv(out_dir / f"final_results_{name}.csv", pd.DataFrame([metrics]))
        atomic_csv(out_dir / f"pred_label_{name}.csv", pd.DataFrame({"pred": preds * LOG10_OF_2, "label": split.labels * LOG10_OF_2}))


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "completed.json").exists() and not args.overwrite:
        print(f"Run already completed: {out_dir}")
        return

    set_seed(args.seed)
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, _, _ = single.build_model(args, Path(args.dict_dir), device)
    optimizer = single.make_optimizer(model, args)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)

    splits = {name: load_split(getattr(args, f"{name}_dir")) for name in ("train", "val", "test")}
    splits["train"] = splits["train"].take(np.random.RandomState(args.seed).permutation(len(splits["train"])))

    last_checkpoint = out_dir / "checkpoint_last.pt"
    start_epoch, records = 0, []
    if last_checkpoint.exists() and not args.overwrite:
        start_epoch, records = single.load_checkpoint(last_checkpoint, model, optimizer, scheduler, None, device)
        print(f"Resumed from epoch {start_epoch}", flush=True)

    started = timeit.default_timer()
    for epoch in range(start_epoch + 1, args.iteration + 1):
        train_metrics, _, _ = train_epoch(model, splits["train"], optimizer, args, device)
        if epoch % args.eval_every == 0 or epoch == args.iteration:
            val_metrics, _ = evaluate(model, splits["val"], args, device)
            test_metrics, _ = evaluate(model, splits["test"], args, device)
        else:  # evaluation does not change training; skipped epochs log NaN
            val_metrics = test_metrics = dict.fromkeys(STAT, float("nan"))
        if epoch // 10 > 0 and epoch % 10 > 0:
            scheduler.step()
        row = {"Epoch": epoch, "Time(sec)": round(timeit.default_timer() - started, 4)}
        row.update({f"{m}_{name}": metrics[m] for name, metrics in (("train", train_metrics), ("val", val_metrics), ("test", test_metrics)) for m in STAT if (name, m) != ("test", "Loss")})
        row["Lr"] = optimizer.param_groups[0]["lr"]
        records.append(row)
        atomic_csv(out_dir / "logfile.csv", pd.DataFrame(records, columns=HEADERS))
        single.save_checkpoint(last_checkpoint, epoch, model, optimizer, scheduler, None, records, args)
        print(f"epoch {epoch} time {row['Time(sec)']:.0f}s train_rmse {row['RMSE_train']:.4f} val_rmse {row['RMSE_val']:.4f} test_rmse {row['RMSE_test']:.4f}", flush=True)

    write_final_outputs(model, splits, args, device, out_dir)
    atomic_json(out_dir / "run_summary.json", {"seed": args.seed, "batch_size": args.batch_size, "lr": args.lr, "final_epoch": args.iteration, "final_val_rmse": float(records[-1]["RMSE_val"]), "precision": "fp32"})
    atomic_json(out_dir / "completed.json", {"completed": True, "final_epoch": int(args.iteration)})


def make_parser():
    parser = argparse.ArgumentParser(description="DeepEnzyme with minibatches and shared weights; otherwise the original protocol.")
    for name in ("train_dir", "val_dir", "test_dir", "out_dir"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--dict_dir", default=str(ROOT / "Data/Input"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", default=666, type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fused_adamw", action="store_true")
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--eval_batch_size", default=256, type=int)
    parser.add_argument("--eval_every", default=1, type=int, help="Evaluate val and test every this many epochs and at the end.")
    parser.add_argument("--max_tokens", default=32768, type=int, help="Padded residues per micro-batch; bounds memory, does not change the update.")
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--iteration", default=200, type=int)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--dropout", default=0.3, type=float)
    parser.add_argument("--dim", default=64, type=int)
    parser.add_argument("--layer_output", default=3, type=int)
    parser.add_argument("--hidden_dim1", default=64, type=int)
    parser.add_argument("--hidden_dim2", default=64, type=int)
    parser.add_argument("--nhead", default=4, type=int)
    parser.add_argument("--hid_size", default=64, type=int)
    parser.add_argument("--layers_trans", default=3, type=int)
    return parser


if __name__ == "__main__":
    main(make_parser().parse_args())
