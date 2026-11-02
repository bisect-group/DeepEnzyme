"""Packed-token trainer on the GPU against fp32 references with a noise-control bound (needs CUDA; run with -m gpu).

Two tests, each with ref, prod and noise runs of the same seed, batches and steps (see test_gpu_noise.py for the bound):
  packed_fp32  prod = packed fp32 trainer, ref = padded minibatch trainer, noise = ref with weights perturbed by u = 1e-6.
  packed_bf16  prod = packed bf16 trainer, ref = packed fp32 trainer, noise = prod with weights perturbed by u = 2**-8.
floor is the mean over DRAWS perturbation draws; gap = |prod - ref|. Every ratio must be at most K.
With EQUIV_NEGATIVE_CONTROL=1 the initial weights of the production run are multiplied by 1.05, so both tests must fail.
"""

import os
from argparse import Namespace

import numpy as np
import pytest
import torch

from conftest import synthetic_rows, train_args, write_feature_dir, write_tiny_dicts
from emulator_bench import minibatch_trainer as mt
from emulator_bench import packed_trainer as pk
from emulator_bench import train_single_target_tvt as single
from emulator_bench.common import set_seed

pytestmark = [pytest.mark.gpu, pytest.mark.noise]

K = 10  # a change of path or precision may move a result by ten times one rounding of the weights at most
TINY = 1e-7  # relative; guards a floor that is exactly zero
EPOCHS = 2
DRAWS = 3  # the floor is the mean over perturbation draws, because one draw can be small by chance
LENGTHS = (300, 420)
SIZES = {"train": 64, "val": 16, "test": 16}
BATCH = 16
PRODUCTION_DIMS = {"dim": 64, "hidden_dim1": 64, "hidden_dim2": 64, "hid_size": 64, "layers_trans": 3, "nhead": 4}  # the default model
NEGATIVE_CONTROL = os.environ.get("EQUIV_NEGATIVE_CONTROL") == "1"
# name: (production trainer, production bf16, reference trainer, u, control runs the production path)
# u is one rounding of the weights at the production precision (bf16 8 bits; 1e-6 for the pure fp32 change of path).
CASES = {
    "packed_fp32": (pk, False, mt, 1e-6, False),
    "packed_bf16": (pk, True, pk, 2**-8, True),
}


@pytest.fixture(scope="module")
def toy_packed(tmp_path_factory):
    root = tmp_path_factory.mktemp("packed")
    dirs = {name: write_feature_dir(root / name, synthetic_rows(n, i, LENGTHS)) for i, (name, n) in enumerate(SIZES.items(), start=1)}
    return Namespace(dirs=dirs, dict_dir=write_tiny_dicts(root / "dicts"), root=root)


def run(toy, trainer, bf16, perturb=0.0, scale=1.0, draw=0):
    """Per-step training losses, then the final validation loss and test predictions (log2 units, as trained)."""
    device = torch.device("cuda:0")
    args = train_args(toy, toy.root / "unused", EPOCHS, device="cuda:0", dropout=0.0, fused_adamw=True, batch_size=BATCH, eval_batch_size=8,
                      eval_every=1, bf16=bf16, max_tokens=10**7, **PRODUCTION_DIMS)
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, _, _ = single.build_model(args, toy.dict_dir, device)
    generator = torch.Generator().manual_seed(1234 + draw)
    with torch.no_grad():
        for parameter in model.parameters():
            noise = torch.randn(parameter.shape, generator=generator).to(device)
            parameter.mul_(scale).add_(perturb * parameter.abs() * noise)
    optimizer = single.make_optimizer(model, args)
    splits = {name: mt.load_split(path) for name, path in toy.dirs.items()}
    splits["train"] = splits["train"].take(np.random.RandomState(args.seed).permutation(len(splits["train"])))
    if trainer is pk:
        splits = {name: pk.Packed(split) for name, split in splits.items()}
    losses = np.concatenate([trainer.train_epoch(model, splits["train"], optimizer, args, device)[2] for _ in range(EPOCHS)])
    val = trainer.evaluate(model, splits["val"], args, device)[0]["Loss"]
    return losses, val, trainer.evaluate(model, splits["test"], args, device)[1]


@pytest.mark.parametrize("name", list(CASES))
def test_packed_within_noise(toy_packed, name):
    trainer, bf16, ref_trainer, u, same_path = CASES[name]
    ref = run(toy_packed, ref_trainer, False)
    clean = run(toy_packed, trainer, bf16)
    prod = run(toy_packed, trainer, bf16, scale=1.05) if NEGATIVE_CONTROL else clean
    controls = [run(toy_packed, trainer if same_path else ref_trainer, bf16 if same_path else False, perturb=u, draw=d) for d in range(DRAWS)]
    base = clean if same_path else ref
    floor_steps = np.mean([np.abs(c[0] - base[0]) for c in controls], axis=0)
    floor_val = np.mean([abs(c[1] - base[1]) for c in controls])
    floor_pred = np.mean([np.abs(c[2] - base[2]).mean() for c in controls])
    gap_steps = np.abs(prod[0] - ref[0])
    gap_val, gap_pred = abs(prod[1] - ref[1]), np.abs(prod[2] - ref[2]).mean()
    ratios = {
        "step0": gap_steps[0] / max(floor_steps[0], TINY * abs(ref[0][0])),
        "steps": gap_steps.max() / max(floor_steps.max(), TINY * np.abs(ref[0]).max()),
        "final_val_loss": gap_val / max(floor_val, TINY * abs(ref[1])),
        "final_pred": gap_pred / max(floor_pred, TINY * np.abs(ref[2]).mean()),
    }
    print(
        f"\n{name}: step0 gap={gap_steps[0]:.3e} floor={floor_steps[0]:.3e} ratio={ratios['step0']:.2f}; "
        f"max gap={gap_steps.max():.3e} max floor={floor_steps.max():.3e} ratio={ratios['steps']:.2f}; "
        f"final val gap={gap_val:.3e} floor={floor_val:.3e} ratio={ratios['final_val_loss']:.2f}; "
        f"final pred gap={gap_pred:.3e} floor={floor_pred:.3e} ratio={ratios['final_pred']:.2f}; K={K}, u={u}"
    )
    assert max(ratios.values()) <= K, ratios
