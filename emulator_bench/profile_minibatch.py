"""Profiling of minibatch_trainer at one batch size. Usage: profile_minibatch.py OUT_DIR [B]. Writes only into OUT_DIR."""
import json, sys, time, types
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emulator_bench import minibatch_trainer as mt  # noqa: E402
from emulator_bench import train_single_target_tvt as single  # noqa: E402

OUT = Path(sys.argv[1]); B = int(sys.argv[2]) if len(sys.argv) > 2 else 512
F = ROOT.parent.parent / "outputs/baselines/DeepEnzyme/kcat/features/random_splits_grouped_sequence/all"
dev = torch.device("cuda:0")
args = mt.make_parser().parse_args(["--train_dir", "x", "--val_dir", "x", "--test_dir", "x", "--out_dir", "x", "--batch_size", str(B), "--fused_adamw"])
torch.backends.cuda.matmul.allow_tf32 = False
torch.manual_seed(0)
model, _, _ = single.build_model(args, Path(args.dict_dir), dev)
opt = single.make_optimizer(model, args)
t = time.time()
splits = {n: mt.load_split(F / n) for n in ("train", "val", "test")}
splits["train"] = splits["train"].take(np.random.RandomState(666).permutation(len(splits["train"])))
print("load", time.time() - t, flush=True)
train = splits["train"]
T = defaultdict(float)


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


def timed(name, fn):
    a = sync(); r = fn(); T[name] += sync() - a; return r


def tforward(model, mb):
    gl = model.gcn.layer4.linear
    e = timed("emb_fp+lin", lambda: gl(model.embed_fingerprint(mb["fingerprints"])))
    sub = timed("sub_gcn+pool", lambda: mt.segment_mean(mt.graph_conv(e, mb["substrate"]), mb["atoms"]))
    tr = model.protein_transformer
    def trans():
        pad = torch.arange(mb["words"].shape[1], device=dev)[None] >= mb["lengths"][:, None]
        x = model.embed_wordTrans(mb["words"]) + tr.PositionalEncoding.pe[0]
        x = tr.transformer_encoder(x.transpose(0, 1), src_key_padding_mask=pad).transpose(0, 1)
        x = tr.linear(x) * (~pad)[:, :, None]
        return x.sum(1) / mb["lengths"][:, None]
    seq = timed("transformer", trans)
    pe = timed("emb_word+lin", lambda: gl(model.embed_wordGCN(mb["flat_words"])))
    prot = timed("prot_gcn+pool", lambda: mt.segment_mean(mt.graph_conv(pe, mb["protein"]), mb["lengths"]))
    def head():
        x = torch.cat((sub, prot, seq), 1)
        for j in range(3):
            x = model.W_out[j](nn.functional.dropout(nn.functional.relu(x), 0.3, True))
        return model.W_interaction(nn.functional.dropout(nn.functional.relu(x), 0.3, True)).squeeze(1)
    return timed("head", head)


def step(jobs_k, timed_mode=True, prepared=None):
    pos = jobs[jobs_k]
    if prepared is None:
        groups = timed("cpu_groups", lambda: mt.micro_batches(train.lengths, pos, args.max_tokens))
        batches = timed("collate(cpu+h2d)", lambda: [mt.collate(train, g, dev) for g in groups])
    else:
        groups, batches = prepared
    opt.zero_grad(set_to_none=True)
    for g, mb in zip(groups, batches):
        out = tforward(model, mb)
        loss = ((out - mb["labels"]) ** 2).sum() / len(pos)
        timed("backward", loss.backward)
    timed("optimizer", opt.step)
    return groups


jobs = [np.arange(s, min(s + B, len(train)), dtype=np.int64) for s in range(0, len(train), B)]
model.train(True)
for k in range(3):
    step(k)
T.clear(); torch.cuda.reset_peak_memory_stats()
N = 30; stats = []
a = sync()
for k in range(3, 3 + N):
    g = step(k)
    stats.append(len(g))
wall = sync() - a
print(f"B={B} steps={N} wall/step={wall/N*1000:.1f} ms  micro/step={np.mean(stats):.1f} peak={torch.cuda.max_memory_allocated()/2**30:.1f}GiB")
tot = sum(T.values())
res = {k: v / N * 1000 for k, v in T.items()}
for k, v in sorted(res.items(), key=lambda x: -x[1]):
    print(f"{k:20s} {v:8.1f} ms  {100*v/ (tot/N*1000):5.1f}%")
print("sum of phases", tot / N * 1000, "ms; wall/step incl. sync overhead", wall / N * 1000)
json.dump({"B": B, "wall_ms": wall / N * 1000, "phases_ms": res}, open(OUT / f"phases_B{B}.json", "w"), indent=1)

# eval time
model.train(False)
for n in ("val", "test"):
    a = sync(); mt.evaluate(model, splits[n], args, dev); print(f"eval {n} n={len(splits[n])} {sync()-a:.1f}s", flush=True)
# eval phases without prefetch: only collate
a = time.perf_counter()
o = np.argsort(splits["val"].lengths, kind="stable")
for s in range(0, len(o), 256):
    for g in mt.micro_batches(splits["val"].lengths, o[s:s+256], args.max_tokens):
        mt.collate(splits["val"], g, dev)
print("val collate only (sync):", sync() - a)
