"""Training throughput of the minibatch trainer per batch size. Usage: sweep_minibatch.py FEATURES_DIR B[,B...] [max_tokens]"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emulator_bench import minibatch_trainer as mb  # noqa: E402
from emulator_bench import train_single_target_tvt as single  # noqa: E402

split_dir, sizes = Path(sys.argv[1]), [int(b) for b in sys.argv[2].split(",")]
max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 32768
parser = mb.make_parser()
device = torch.device("cuda:0")
torch.backends.cuda.matmul.allow_tf32 = False
train = mb.load_split(split_dir / "train")
train = train.take(np.random.RandomState(666).permutation(len(train)))
for B in sizes:
    args = parser.parse_args(["--train_dir", "x", "--val_dir", "x", "--test_dir", "x", "--out_dir", "x", "--batch_size", str(B), "--max_tokens", str(max_tokens), "--fused_adamw"])
    torch.manual_seed(666)
    model = single.build_model(args, Path(args.dict_dir), device)[0]
    optimizer = single.make_optimizer(model, args)
    steps = max(10, min(200, 16384 // B))
    n = (steps + 5) * B
    sub = train.take(np.arange(n) % len(train))
    torch.cuda.reset_peak_memory_stats()
    args.batch_size = B
    mb.train_epoch(model, sub.take(np.arange(5 * B)), optimizer, args, device)  # warm-up
    torch.cuda.synchronize()
    t = time.time()
    mb.train_epoch(model, sub.take(np.arange(5 * B, n)), optimizer, args, device)
    torch.cuda.synchronize()
    dt = time.time() - t
    print(f"B {B} max_tokens {max_tokens} steps {steps} samples/s {steps * B / dt:.1f} peak_alloc_GB {torch.cuda.max_memory_allocated() / 2**30:.2f} peak_reserved_GB {torch.cuda.max_memory_reserved() / 2**30:.2f}", flush=True)
    del model, optimizer
