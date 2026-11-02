"""DeepEnzyme minibatch trainer on packed tokens: the same update as `minibatch_trainer.py` without padded residues.

Residues of a batch are concatenated and the protein transformer attends within each sample through nested-tensor (jagged)
attention. Everything else is as in `minibatch_trainer.py`: block-diagonal GCNs, per-sample mean pooling, one seeded shuffle
reused in every epoch, per-epoch evaluation, last-epoch model and the same output files and log10 metrics.

A batch is one micro-batch unless it holds more than `--max_tokens` residues. Then it is cut into consecutive chunks whose
gradients are summed, so the update stays that of the whole batch and only the memory is bounded.
`--bf16` is an opt-in autocast. The GCN sparse products and the segment means stay in fp32.
"""

import json
import sys
import timeit
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn, optim

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from emulator_bench import minibatch_trainer as mt  # noqa: E402
from emulator_bench import train_single_target_tvt as single  # noqa: E402
from emulator_bench.common import atomic_csv, atomic_json, set_seed  # noqa: E402

LOG10_OF_2 = mt.LOG10_OF_2
STAT, HEADERS = mt.STAT, mt.HEADERS
PREFETCH = 3  # jobs prepared ahead of the GPU


class Packed:
    """Host copy of a `Split` prepared for batching: float32 graph values and per-sample sizes."""

    def __init__(self, split):
        self.labels, self.lengths, self.words, self.fingerprints = split.labels, split.lengths, split.words, split.fingerprints
        self.atoms = np.array([len(f) for f in split.fingerprints])
        self.substrate, self.protein = (
            ([(m.indptr[:-1], m.indices, m.data.astype(np.float32)) for m in matrices], np.array([m.shape[0] for m in matrices]), np.array([m.nnz for m in matrices]))
            for matrices in (split.substrate, split.protein)
        )

    def __len__(self):
        return len(self.labels)


def block_diagonal(graph, idx, device):
    """Block-diagonal CSR matrix of the samples `idx` as (crow, col, val), as `minibatch_trainer.concat_csr`.
    The per-sample arrays go to the device as they are; the block offsets are added there."""
    parts, all_sizes, all_nnz = graph
    sizes, nnz = all_sizes[idx], all_nnz[idx]
    indptr, indices, data = (torch.from_numpy(np.concatenate(a)).to(device) for a in zip(*(parts[i] for i in idx)))
    shift = lambda counts: torch.from_numpy(np.cumsum(counts) - counts).to(device)  # noqa: E731
    crow = torch.cat([indptr.long() + shift(nnz).repeat_interleave(torch.from_numpy(sizes).to(device), output_size=int(sizes.sum())), torch.tensor([nnz.sum()], device=device)])
    col = indices.long() + shift(sizes).repeat_interleave(torch.from_numpy(nnz).to(device), output_size=int(nnz.sum()))
    return crow, col, data


def collate(split, idx, device):
    """Device tensors of the samples `idx` with packed residues and `offsets` of the samples in them."""
    lengths = split.lengths[idx]
    to = lambda a: torch.from_numpy(a).to(device)  # noqa: E731
    return {
        "fingerprints": to(np.concatenate([split.fingerprints[i] for i in idx])),
        "atoms": to(split.atoms[idx]),
        "substrate": block_diagonal(split.substrate, idx, device),
        "lengths": to(lengths),
        "offsets": to(np.concatenate([[0], np.cumsum(lengths)])),
        "flat_words": to(np.concatenate([split.words[i] for i in idx])),
        "protein": block_diagonal(split.protein, idx, device),
        "labels": to(split.labels[idx]),
    }


def graph_conv(x, graph):
    """`minibatch_trainer.graph_conv` in fp32 whatever the autocast state."""
    with torch.autocast(x.device.type, enabled=False):
        return mt.graph_conv(x.float(), graph)


def segment_mean(x, sizes):
    return mt.segment_mean(x.float(), sizes)


def packed_layer(layer, x, offsets):
    """One post-norm encoder layer on packed tokens [T, d]; attention is limited to each sample by the jagged offsets."""
    attention, p, training = layer.self_attn, layer.dropout.p, layer.training
    heads = attention.num_heads
    qkv = nn.functional.linear(x, attention.in_proj_weight, attention.in_proj_bias).view(len(x), 3, heads, -1)
    q, k, v = (torch.nested.nested_tensor_from_jagged(qkv[:, i].contiguous(), offsets).transpose(1, 2) for i in range(3))
    out = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=p if training else 0.0).transpose(1, 2).values().reshape(len(x), -1)
    x = layer.norm1(x + nn.functional.dropout(attention.out_proj(out), p, training))
    hidden = layer.linear2(nn.functional.dropout(nn.functional.relu(layer.linear1(x)), p, training))
    return layer.norm2(x + nn.functional.dropout(hidden, p, training))


def forward(model, mb, layer_output, dropout):
    """Predictions [B] in log2 units. Same maths as `minibatch_trainer.forward` without padding."""
    gcn_linear = model.gcn.layer4.linear
    substrate = segment_mean(graph_conv(gcn_linear(model.embed_fingerprint(mb["fingerprints"])), mb["substrate"]), mb["atoms"])

    transformer = model.protein_transformer
    x = model.embed_wordTrans(mb["flat_words"]) + transformer.PositionalEncoding.pe[0]  # the original adds row 0 to every residue
    for layer in transformer.transformer_encoder.layers:
        x = packed_layer(layer, x, mb["offsets"])
    sequence = segment_mean(transformer.linear(x), mb["lengths"])

    protein = segment_mean(graph_conv(gcn_linear(model.embed_wordGCN(mb["flat_words"])), mb["protein"]), mb["lengths"])

    x = torch.cat((substrate, protein, sequence), 1)
    for j in range(layer_output):
        x = model.W_out[j](nn.functional.dropout(nn.functional.relu(x), dropout, training=model.training))
    return model.W_interaction(nn.functional.dropout(nn.functional.relu(x), dropout, training=model.training)).squeeze(1).float()


def token_chunks(lengths, positions, max_tokens):
    """Consecutive chunks of `positions` of at most `max_tokens` residues each (a single longer sample is its own chunk)."""
    cumulative = np.cumsum(lengths[positions])
    chunks, start = [], 0
    while start < len(positions):
        base = cumulative[start - 1] if start else 0
        end = max(int(np.searchsorted(cumulative, base + max_tokens, side="right")), start + 1)
        chunks.append(positions[start:end])
        start = end
    return chunks


def prefetched(split, jobs, device, max_tokens):
    """Yields (chunks, device batches) for each job, preparing the next jobs in threads while the GPU works."""
    def prepare(positions):
        chunks = token_chunks(split.lengths, positions, max_tokens)
        return chunks, [collate(split, c, device) for c in chunks]

    with ThreadPoolExecutor(PREFETCH) as pool:
        queue = deque(pool.submit(prepare, job) for job in jobs[:PREFETCH])
        for k in range(len(jobs)):
            result = queue.popleft().result()
            if k + PREFETCH < len(jobs):
                queue.append(pool.submit(prepare, jobs[k + PREFETCH]))
            yield result


def autocast(args, device):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.bf16)


def train_epoch(model, split, optimizer, args, device):
    """One epoch over consecutive batches of `split`. Returns the log10 metrics, the log2 per-sample training predictions and the per-step losses."""
    model.train(True)
    n, B = len(split), args.batch_size
    jobs = [np.arange(s, min(s + B, n)) for s in range(0, n, B)]
    preds, step_losses = torch.zeros(n, device=device), []
    for positions, (chunks, batches) in zip(jobs, prefetched(split, jobs, device, args.max_tokens)):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for c, mb in zip(chunks, batches):
            with autocast(args, device):
                output = forward(model, mb, args.layer_output, args.dropout)
            loss = ((output - mb["labels"]) ** 2).sum() / len(positions)
            loss.backward()
            preds[torch.from_numpy(c).to(device)] = output.detach()
            step_loss += loss.detach()
        optimizer.step()
        step_losses.append(step_loss)
    preds = preds.cpu().double().numpy()
    return mt.summarize(split.labels, preds), preds, torch.stack(step_losses).cpu().numpy()


@torch.no_grad()
def evaluate(model, split, args, device):
    """Eval-mode pass over one split. Returns log10 metrics and the log2 predictions in the order of the split."""
    model.train(False)
    n = len(split)
    jobs = [np.arange(s, min(s + args.eval_batch_size, n)) for s in range(0, n, args.eval_batch_size)]
    preds = torch.zeros(n, device=device)
    for _, (chunks, batches) in zip(jobs, prefetched(split, jobs, device, args.max_tokens)):
        for c, mb in zip(chunks, batches):
            with autocast(args, device):
                preds[torch.from_numpy(c).to(device)] = forward(model, mb, args.layer_output, args.dropout)
    preds = preds.cpu().double().numpy()
    return mt.summarize(split.labels, preds), preds


def write_final_outputs(model, splits, args, device, out_dir):
    for name, split in splits.items():
        metrics, preds = evaluate(model, split, args, device)
        atomic_csv(out_dir / f"final_results_{name}.csv", pd.DataFrame([metrics]))
        atomic_csv(out_dir / f"pred_label_{name}.csv", pd.DataFrame({"pred": preds * LOG10_OF_2, "label": split.labels * LOG10_OF_2}))


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    completed = out_dir / "completed.json"
    if completed.exists() and not args.overwrite:
        final_epoch = json.loads(completed.read_text())["final_epoch"]
        if final_epoch != args.iteration:
            raise RuntimeError(f"{out_dir} completed {final_epoch} epochs but --iteration is {args.iteration}; use --overwrite or another --out_dir")
        print(f"Run already completed: {out_dir}")
        return

    set_seed(args.seed)
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, _, _ = single.build_model(args, Path(args.dict_dir), device)
    optimizer = single.make_optimizer(model, args)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)

    splits = {name: mt.load_split(getattr(args, f"{name}_dir")) for name in ("train", "val", "test")}
    splits["train"] = splits["train"].take(np.random.RandomState(args.seed).permutation(len(splits["train"])))
    splits = {name: Packed(split) for name, split in splits.items()}

    last_checkpoint = out_dir / "checkpoint_last.pt"
    start_epoch, records = 0, []
    if last_checkpoint.exists() and not args.overwrite:
        start_epoch, records = single.load_checkpoint(last_checkpoint, model, optimizer, scheduler, None, device)
        print(f"Resumed from epoch {start_epoch}", flush=True)

    peak_gib = None
    started = timeit.default_timer()
    for epoch in range(start_epoch + 1, args.iteration + 1):
        if device.type == "cuda" and peak_gib is None:
            torch.cuda.reset_peak_memory_stats(device)
        train_started = timeit.default_timer()
        train_metrics, _, _ = train_epoch(model, splits["train"], optimizer, args, device)
        train_seconds = timeit.default_timer() - train_started
        if device.type == "cuda" and peak_gib is None:
            peak_gib = torch.cuda.max_memory_allocated(device) / 2**30
            print(f"peak memory of the first epoch: {peak_gib:.2f} GiB (max_tokens {args.max_tokens})", flush=True)
        eval_started = timeit.default_timer()
        if epoch % args.eval_every == 0 or epoch == args.iteration:
            val_metrics, _ = evaluate(model, splits["val"], args, device)
            test_metrics, _ = evaluate(model, splits["test"], args, device)
        else:  # evaluation does not change training; skipped epochs log NaN
            val_metrics = test_metrics = dict.fromkeys(STAT, float("nan"))
        eval_seconds = timeit.default_timer() - eval_started
        if epoch // 10 > 0 and epoch % 10 > 0:
            scheduler.step()
        row = {"Epoch": epoch, "Time(sec)": round(timeit.default_timer() - started, 4)}
        row.update({f"{m}_{name}": metrics[m] for name, metrics in (("train", train_metrics), ("val", val_metrics), ("test", test_metrics)) for m in STAT if (name, m) != ("test", "Loss")})
        row["Lr"] = optimizer.param_groups[0]["lr"]
        records.append(row)
        atomic_csv(out_dir / "logfile.csv", pd.DataFrame(records, columns=HEADERS))
        single.save_checkpoint(last_checkpoint, epoch, model, optimizer, scheduler, None, records, args)
        print(f"epoch {epoch} time {row['Time(sec)']:.0f}s (train {train_seconds:.1f}s eval {eval_seconds:.1f}s) train_rmse {row['RMSE_train']:.4f} val_rmse {row['RMSE_val']:.4f} test_rmse {row['RMSE_test']:.4f}", flush=True)

    write_final_outputs(model, splits, args, device, out_dir)
    atomic_json(
        out_dir / "run_summary.json",
        {"seed": args.seed, "batch_size": args.batch_size, "lr": args.lr, "final_epoch": args.iteration, "final_val_rmse": float(records[-1]["RMSE_val"]),
         "precision": "bf16" if args.bf16 else "fp32", "first_epoch_peak_gib": peak_gib},
    )
    atomic_json(completed, {"completed": True, "final_epoch": int(args.iteration)})


def make_parser():
    parser = mt.make_parser()
    parser.description = "DeepEnzyme with packed-token minibatches and shared weights; otherwise the original protocol."
    parser.set_defaults(max_tokens=400000)
    parser._option_string_actions["--max_tokens"].help = "Residues per micro-batch; a larger batch is cut with summed gradients. Bounds memory, does not change the update."
    parser.add_argument("--bf16", action="store_true", help="bf16 autocast; GCN sparse products and segment means stay fp32.")
    return parser


if __name__ == "__main__":
    main(make_parser().parse_args())
