"""Batch-1 runs of the exact batched-across-runs trainer with val and test evaluated only every --eval_every epochs and at the end.

Evaluation does not change training. Skipped epochs log placeholder val/test values (predictions of 0), so only the evaluated epochs are valid.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emulator_bench import batched_trainer as bt  # noqa: E402

eval_every = int(sys.argv.pop(sys.argv.index("--eval_every") + 1)) if "--eval_every" in sys.argv else 20
sys.argv.remove("--eval_every") if "--eval_every" in sys.argv else None
args = bt.parse_args()
epoch = 0
original = bt.Runner.pass_over


def pass_over(self, splits, orders, training):
    global epoch
    if training:
        epoch += 1
    if training or epoch % eval_every == 0 or epoch == args.iteration:
        return original(self, splits, orders, training)
    return [np.zeros(len(o), dtype=np.float32) for o in orders]


bt.Runner.pass_over = pass_over
bt.main(args, [bt.Job(*job) for job in args.job])
