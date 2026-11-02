"""GPU-only step time of exact and precision-changing variants. Usage: profile_minibatch_variants.py [B]. Update maths unchanged except as labelled."""
import sys, time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emulator_bench import minibatch_trainer as mt  # noqa: E402
from emulator_bench import train_single_target_tvt as single  # noqa: E402

B = int(sys.argv[1]) if len(sys.argv) > 1 else 512
F = ROOT.parent.parent / "outputs/baselines/DeepEnzyme/kcat/features/random_splits_grouped_sequence/all"
dev = torch.device("cuda:0")
args = mt.make_parser().parse_args(["--train_dir", "x", "--val_dir", "x", "--test_dir", "x", "--out_dir", "x", "--batch_size", str(B), "--fused_adamw"])
model, _, _ = single.build_model(args, Path(args.dict_dir), dev)
opt = single.make_optimizer(model, args)
train = mt.load_split(F / "train")
train = train.take(np.random.RandomState(666).permutation(len(train)))
jobs = [np.arange(s, min(s + B, len(train))) for s in range(0, len(train), B)][3:13]
model.train(True)


def tight(lengths, positions, max_tokens, ratio):
    order = positions[np.argsort(-lengths[positions], kind="stable")]
    groups, start = [], 0
    while start < len(order):
        end = start + 1
        while end < len(order) and (end - start + 1) * lengths[order[start]] <= max_tokens and lengths[order[end]] >= ratio * lengths[order[start]]:
            end += 1
        groups.append(order[start:end]); start = end
    return groups


def prepare(grouper):
    out = []
    for pos in jobs:
        gs = grouper(pos)
        out.append((pos, gs, [mt.collate(train, g, dev) for g in gs]))
    return out


def run(prep, ctx=nullcontext, amp=None, reps=2):
    def step(item):
        pos, gs, bs = item
        opt.zero_grad(set_to_none=True)
        for g, mb in zip(gs, bs):
            with ctx():
                with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                    out = mt.forward(model, mb, 3, 0.3)
                loss = ((out.float() - mb["labels"]) ** 2).sum() / len(pos)
            loss.backward()
        opt.step()
    for it in prep[:2]: step(it)
    torch.cuda.synchronize(); best = 1e9
    for _ in range(reps):
        a = time.perf_counter()
        for it in prep: step(it)
        torch.cuda.synchronize(); best = min(best, (time.perf_counter() - a) / len(prep))
    return best * 1000, np.mean([len(x[1]) for x in prep])


base = prepare(lambda p: mt.micro_batches(train.lengths, p, args.max_tokens))
print("baseline mem-eff fp32      %.1f ms/step micro=%.1f" % run(base), flush=True)
torch.backends.cuda.matmul.allow_tf32 = False
try:
    print("math sdpa fp32 (exact)     %.1f ms/step micro=%.1f" % run(base, lambda: sdpa_kernel(SDPBackend.MATH)), flush=True)
except torch.OutOfMemoryError:
    print("math sdpa fp32 at max_tokens 32768: OOM", flush=True)
    opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
b8 = prepare(lambda p: mt.micro_batches(train.lengths, p, 8192))
print("mem-eff max_tokens 8192     %.1f ms/step micro=%.1f" % run(b8), flush=True)
print("math sdpa max_tokens 8192   %.1f ms/step micro=%.1f" % run(b8, lambda: sdpa_kernel(SDPBackend.MATH)), flush=True)
del b8; torch.cuda.empty_cache()
for r in (0.95, 0.9, 0.8):
    p = prepare(lambda p_: tight(train.lengths, p_, args.max_tokens, r))
    w = sum((len(g) * train.lengths[g].max()) for _, gs, _ in p for g in gs) / sum(train.lengths[g].sum() for _, gs, _ in p for g in gs)
    print(f"tight bucketing ratio {r} (exact) %.1f ms/step micro=%.1f  padded/real tokens {w:.3f}" % run(p), flush=True)
    if r == 0.9:
        pass
torch.backends.cuda.matmul.allow_tf32 = True
print("TF32 baseline (prec-changing) %.1f ms/step micro=%.1f" % run(base), flush=True)
torch.backends.cuda.matmul.allow_tf32 = False
print("bf16 autocast (prec-changing) %.1f ms/step micro=%.1f" % run(base, amp=torch.bfloat16), flush=True)
p9 = prepare(lambda p_: tight(train.lengths, p_, args.max_tokens, 0.9))
print("bf16 + tight0.9             %.1f ms/step micro=%.1f" % run(p9, amp=torch.bfloat16), flush=True)
