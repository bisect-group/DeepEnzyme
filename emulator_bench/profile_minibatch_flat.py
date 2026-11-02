"""Prototype: protein transformer on packed (unpadded) tokens with nested-tensor attention, one micro-batch per step.
Same parameters and maths as minibatch_trainer.forward; checks eval-mode agreement and times a GPU-only step. Usage: profile_minibatch_flat.py [B]."""
import sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emulator_bench import minibatch_trainer as mt  # noqa: E402
from emulator_bench import train_single_target_tvt as single  # noqa: E402

B = int(sys.argv[1]) if len(sys.argv) > 1 else 512
F = ROOT.parent.parent / "outputs/baselines/DeepEnzyme/kcat/features/random_splits_grouped_sequence/all"
dev = torch.device("cuda:0")
args = mt.make_parser().parse_args(["--train_dir", "x", "--val_dir", "x", "--test_dir", "x", "--out_dir", "x", "--batch_size", str(B), "--fused_adamw"])
torch.backends.cuda.matmul.allow_tf32 = False
torch.manual_seed(0)
model, _, _ = single.build_model(args, Path(args.dict_dir), dev)
opt = single.make_optimizer(model, args)
train = mt.load_split(F / "train")
train = train.take(np.random.RandomState(666).permutation(len(train)))
H, D = 4, 16


def flat_layer(layer, x, offsets, p, training):
    a = layer.self_attn
    qkv = Fn.linear(x, a.in_proj_weight, a.in_proj_bias).view(-1, 3, H, D)
    q, k, v = (torch.nested.nested_tensor_from_jagged(qkv[:, i].contiguous(), offsets).transpose(1, 2) for i in range(3))
    o = Fn.scaled_dot_product_attention(q, k, v, dropout_p=p if training else 0.0).transpose(1, 2).values().reshape(-1, H * D)
    x = layer.norm1(x + Fn.dropout(a.out_proj(o), p, training))
    ff = layer.linear2(Fn.dropout(Fn.relu(layer.linear1(x)), p, training))
    return layer.norm2(x + Fn.dropout(ff, p, training))


def flat_forward(model, mb, offsets, p=0.3):
    tr = model.protein_transformer
    gl = model.gcn.layer4.linear
    sub = mt.segment_mean(mt.graph_conv(gl(model.embed_fingerprint(mb["fingerprints"])), mb["substrate"]), mb["atoms"])
    x = model.embed_wordTrans(mb["flat_words"]) + tr.PositionalEncoding.pe[0]
    for layer in tr.transformer_encoder.layers:
        x = flat_layer(layer, x, offsets, p, model.training)
    seq = mt.segment_mean(tr.linear(x), mb["lengths"])
    prot = mt.segment_mean(mt.graph_conv(gl(model.embed_wordGCN(mb["flat_words"])), mb["protein"]), mb["lengths"])
    x = torch.cat((sub, prot, seq), 1)
    for j in range(3):
        x = model.W_out[j](Fn.dropout(Fn.relu(x), p, model.training))
    return model.W_interaction(Fn.dropout(Fn.relu(x), p, model.training)).squeeze(1)


def offs(mb):
    return torch.cat([mb["lengths"].new_zeros(1), mb["lengths"].cumsum(0)]).to(torch.int64)


# agreement in eval mode on a mixed micro-batch
model.train(False)
idx = np.arange(64)
mb = mt.collate(train, idx, dev)
with torch.no_grad():
    ref = mt.forward(model, mb, 3, 0.3)
    new = flat_forward(model, mb, offs(mb))
print("eval max |diff|", (ref - new).abs().max().item(), "ref scale", ref.abs().mean().item(), flush=True)
# gradient agreement (dropout off): train-mode flag off, compare grads
out = mt.forward(model, mb, 3, 0.3); out.sum().backward(); g1 = [p.grad.clone() for p in model.parameters() if p.grad is not None]
model.zero_grad(); out = flat_forward(model, mb, offs(mb)); out.sum().backward(); g2 = [p.grad.clone() for p in model.parameters() if p.grad is not None]
print("max rel grad diff", max(((a - b).abs().max() / (a.abs().max() + 1e-12)).item() for a, b in zip(g1, g2)), flush=True)

model.train(True)
jobs = [np.arange(s, min(s + B, len(train))) for s in range(0, len(train), B)][3:13]
batches = [(pos, mt.collate(train, pos, dev)) for pos in jobs]


def step(pos, mb):
    opt.zero_grad(set_to_none=True)
    out = flat_forward(model, mb, offs(mb))
    (((out - mb["labels"]) ** 2).sum() / len(pos)).backward()
    opt.step()


for b in batches[:2]:
    step(*b)
torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
a = time.perf_counter()
for b in batches:
    step(*b)
torch.cuda.synchronize()
print(f"flat NJT one micro-batch fp32: {(time.perf_counter() - a) / len(batches) * 1000:.1f} ms/step peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)

from torch.profiler import ProfilerActivity, profile
with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
    for b in batches[:5]:
        step(*b)
    torch.cuda.synchronize()
rows = sorted([e for e in prof.key_averages() if e.device_type == torch.autograd.DeviceType.CUDA], key=lambda e: -e.device_time_total)
tot = sum(e.device_time_total for e in rows)
print(f"flat total CUDA {tot / 5e3:.1f} ms/step")
for e in rows[:8]:
    print(f"{100 * e.device_time_total / tot:5.1f}% {e.device_time_total / 5e3:7.2f} ms/step {e.key[:100]}")

orig = mt.graph_conv
def _gc(x, g):
    with torch.autocast("cuda", enabled=False):
        return orig(x.float(), g)
mt.graph_conv = _gc
seg = mt.segment_mean
mt.segment_mean = lambda x, s: seg(x.float(), s)
def timeit(label, amp=None, tf32=False):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    def st(pos, mb):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            out = flat_forward(model, mb, offs(mb))
        (((out.float() - mb["labels"]) ** 2).sum() / len(pos)).backward()
        opt.step()
    for b in batches[:2]:
        st(*b)
    torch.cuda.synchronize(); a = time.perf_counter()
    for b in batches:
        st(*b)
    torch.cuda.synchronize()
    print(f"{label}: {(time.perf_counter() - a) / len(batches) * 1000:.1f} ms/step", flush=True)
timeit("flat fp32 (float-cast graph_conv)")
timeit("flat TF32 (prec-changing)", tf32=True)
timeit("flat bf16 autocast (prec-changing)", amp=torch.bfloat16)
