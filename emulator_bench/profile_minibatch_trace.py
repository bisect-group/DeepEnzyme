"""torch.profiler trace, GPU idle share and padding waste of minibatch_trainer. Usage: profile_minibatch_trace.py OUT_DIR [B]."""
import json, sys
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

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
train = mt.load_split(F / "train")
train = train.take(np.random.RandomState(666).permutation(len(train)))
jobs = [np.arange(s, min(s + B, len(train))) for s in range(0, len(train), B)]

# padding waste over the whole epoch
real = padded = attn_real = attn_pad = 0; nmb = []; flops_lin = 0
for pos in jobs:
    gs = mt.micro_batches(train.lengths, pos, args.max_tokens); nmb.append(len(gs))
    for g in gs:
        L = train.lengths[g]; m = L.max()
        real += L.sum(); padded += len(g) * m
        attn_real += (L.astype(np.float64) ** 2).sum(); attn_pad += len(g) * float(m) ** 2
L = train.lengths
w = dict(real_residues=int(real), padded_residues=int(padded), token_waste_pct=100 * (1 - real / padded),
         attn_waste_pct=100 * (1 - attn_real / attn_pad), micro_per_step=float(np.mean(nmb)),
         len_mean=float(L.mean()), len_median=float(np.median(L)), len_p99=float(np.percentile(L, 99)), len_max=int(L.max()),
         rows_over_1500=int((L > 1500).sum()))
# max_tokens-free ideal: tokens in a step if each micro is a single sorted group (same as now) vs per-sample
print(json.dumps(w, indent=1))
json.dump(w, open(OUT / "padding.json", "w"), indent=1)

model.train(True)
def step(k):
    pos = jobs[k]
    groups = mt.micro_batches(train.lengths, pos, args.max_tokens)
    batches = [mt.collate(train, g, dev) for g in groups]
    opt.zero_grad(set_to_none=True)
    for g, mb in zip(groups, batches):
        out = mt.forward(model, mb, 3, 0.3)
        (((out - mb["labels"]) ** 2).sum() / len(pos)).backward()
    opt.step()
for k in range(3): step(k)
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
    for k in range(3, 13): step(k)
    torch.cuda.synchronize()
prof.export_chrome_trace(str(OUT / f"trace_B{B}.json"))
ev = prof.key_averages()
tot = sum(e.device_time_total for e in ev if e.device_type == torch.autograd.DeviceType.CUDA)
rows = sorted([e for e in ev if e.device_type == torch.autograd.DeviceType.CUDA], key=lambda e: -e.device_time_total)
print(f"total CUDA kernel time {tot/1e3/10:.1f} ms/step")
with open(OUT / "top_kernels.txt", "w") as f:
    for e in rows[:40]:
        s = f"{100*e.device_time_total/tot:5.1f}% {e.device_time_total/1e3/10:8.2f} ms/step n/step={e.count/10:7.1f}  {e.key[:110]}"
        f.write(s + "\n")
        if rows.index(e) < 25: print(s)
# categories
cats = {"sparse": ("spmm", "csr", "sparse", "cusparse", "SpMM"), "attention": ("attention", "fmha", "softmax", "flash", "mha", "efficient", "bmm", "gemm_bmm"),
        "embedding": ("embedding", "index_put", "indexing", "sort", "radix", "cub", "scatter", "index_add"), "adamw": ("adam", "multi_tensor", "foreach"),
        "gemm": ("gemm", "sgemm", "cutlass", "cublas", "ampere", "sm80", "sm86"), "elementwise": ("elementwise", "vectorized", "reduce")}
# dump all kernel names for manual categorisation
json.dump([(e.key, e.device_time_total / 1e3 / 10, e.count / 10) for e in rows], open(OUT / "all_kernels.json", "w"))

# GPU idle from trace: union of kernel intervals vs span
tr = json.load(open(OUT / f"trace_B{B}.json"))["traceEvents"]
k = sorted((e["ts"], e["ts"] + e["dur"]) for e in tr if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e)
span0, span1 = k[0][0], max(b for _, b in k)
busy = 0; cur0, cur1 = k[0]; gaps = []
for a, b in k[1:]:
    if a > cur1:
        busy += cur1 - cur0; gaps.append(a - cur1); cur0, cur1 = a, b
    else:
        cur1 = max(cur1, b)
busy += cur1 - cur0
g = np.array(gaps)
print(f"span {(span1-span0)/1e3/10:.1f} ms/step, GPU busy {100*busy/(span1-span0):.1f}%, idle {100*(1-busy/(span1-span0)):.1f}%; gaps>1ms: n={int((g>1000).sum())} total {g[g>1000].sum()/1e3/10:.1f} ms/step; gaps>10ms total {g[g>10000].sum()/1e3/10:.1f} ms/step")
json.dump({"span_ms_step": (span1 - span0) / 1e3 / 10, "busy_pct": 100 * busy / (span1 - span0), "gap_gt1ms_ms_step": float(g[g > 1000].sum() / 1e3 / 10)}, open(OUT / "idle.json", "w"))
