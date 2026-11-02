"""Production precision against fp32 with a noise-control bound (needs CUDA; run with --gpu).

For every candidate (bf16, fp16, TF32, fused AdamW) the same seed, batches and steps are run three ways:
  ref   the fp32 path of the original code;
  prod  the production path (the wrapper functions with the candidate flags);
  noise the control: prod with its initial weights perturbed by w += u * |w| * randn.
For bf16, fp16 and TF32 the control runs in the production precision and floor = |noise - prod|. A pure fp32 change
(fused AdamW) uses the fp32 path with u = 1e-6 as control and floor = |noise - ref|. gap = |prod - ref|.
The test passes when gap <= K * floor at step 0, for the largest step gap against the largest step floor, and for the final metrics.
With EQUIV_NEGATIVE_CONTROL=1 the initial weights of the production run are multiplied by 1.05, so every test must fail.
"""

import os

import numpy as np
import pytest
import torch

from conftest import synthetic_rows, train_args, write_feature_dir, write_tiny_dicts
from emulator_bench import train_single_target_tvt as wrapper
from emulator_bench.common import set_seed

pytestmark = [pytest.mark.gpu, pytest.mark.noise]

K = 10  # a switch of precision may change a result by ten times one rounding of the weights at most
TINY = 1e-7  # relative; guards a floor that is exactly zero
EPOCHS = 2
DRAWS = 3  # perturbation draws; the floor is their mean, because one draw can be small by chance
# Dropout is off: a dropout mask drawn for a 16-bit tensor differs from the mask for a float32 tensor of the same seed,
# so a gap with dropout would measure the masks, not the precision.
# name: (flags of the production path, u, control runs in the production precision)
# u is one rounding of the weights at the production precision (bf16 8 bits, fp16 and TF32 11 bits, fp32 1e-6 for fused).
CANDIDATES = {
    "bf16": ({"amp": True}, 2**-8, True),
    "fp16": ({"fp16": True}, 2**-11, True),
    "tf32": ({"tf32": True}, 2**-11, True),
    "fused_adamw": ({"fused_adamw": True}, 1e-6, False),
}
LENGTHS = (300, 420)  # TF32 changes a matrix product only above about 300 along the contracted axis
PRODUCTION_DIMS = {"dim": 64, "hidden_dim1": 64, "hidden_dim2": 64, "hid_size": 64, "layers_trans": 3}  # the default model
NEGATIVE_CONTROL = os.environ.get("EQUIV_NEGATIVE_CONTROL") == "1"
# Candidates that fail the bound are rejected. The test records that and fails if one starts to pass.
REJECTED = {
    "fp16": "GradScaler starts at scale 65536 and skips the first updates; the fp32 run does not",
    "tf32": "the final validation loss differs by more than K times the floor",
}


@pytest.fixture(scope="module")
def toy_noise(tmp_path_factory):
    root = tmp_path_factory.mktemp("noise")
    sizes = {"train": 24, "val": 8, "test": 8}
    dirs = {name: write_feature_dir(root / name, synthetic_rows(n, i, LENGTHS)) for i, (name, n) in enumerate(sizes.items(), start=1)}
    from argparse import Namespace

    return Namespace(dirs=dirs, dict_dir=write_tiny_dicts(root / "dicts"), root=root)


def run(toy, flags, perturb=0.0, scale=1.0, draw=0):
    """Per-step training losses, then the final validation loss and test predictions (log2 units, as trained)."""
    args = train_args(toy, toy.root / "unused", EPOCHS, device="cuda:0", dropout=0.0, **PRODUCTION_DIMS, **flags)
    device = torch.device("cuda:0")
    set_seed(args.seed)
    model, _, _ = wrapper.build_model(args, toy.dict_dir, device)
    generator = torch.Generator().manual_seed(1234 + draw)
    with torch.no_grad():
        for parameter in model.parameters():
            noise = torch.randn(parameter.shape, generator=generator).to(device)
            parameter.mul_(scale).add_(perturb * parameter.abs() * noise)
    amp_dtype, _, scaler = wrapper.configure_precision(args, device)
    optimizer = wrapper.make_optimizer(model, args)
    loss_fn = torch.nn.MSELoss().to(device)
    data = {name: wrapper.load_split_dataset(path, device) for name, path in toy.dirs.items()}
    data["train"] = wrapper.shuffled_once(data["train"], args.seed)
    labels = {name: wrapper.label_array(d) for name, d in data.items()}
    losses = []
    model.train(True)
    for _ in range(EPOCHS):
        for i, sample in enumerate(data["train"]):
            metrics, _, _ = wrapper.run_pass(model, [sample], labels["train"][i : i + 1], loss_fn, args, device, amp_dtype, "step", optimizer, scaler)
            losses.append(metrics["Loss"])
    val, _, _ = wrapper.evaluate(model, data["val"], labels["val"], loss_fn, args, device, amp_dtype, "val")
    _, _, test_pred = wrapper.evaluate(model, data["test"], labels["test"], loss_fn, args, device, amp_dtype, "test")
    return np.array(losses), val["Loss"], test_pred


@pytest.fixture(scope="module")
def reference(toy_noise):
    return run(toy_noise, {})


@pytest.mark.parametrize(
    "name",
    [pytest.param(name, marks=pytest.mark.xfail(strict=True, reason=REJECTED[name])) if name in REJECTED and not NEGATIVE_CONTROL else name for name in CANDIDATES],
)
def test_precision_stays_within_noise(toy_noise, reference, name):
    flags, u, same_precision = CANDIDATES[name]
    ref_losses, ref_val, ref_pred = reference
    clean = run(toy_noise, flags)
    prod = run(toy_noise, flags, scale=1.05) if NEGATIVE_CONTROL else clean
    controls = [run(toy_noise, flags if same_precision else {}, perturb=u, draw=d) for d in range(DRAWS)]
    base = clean if same_precision else reference
    floor_steps = np.mean([np.abs(c[0] - base[0]) for c in controls], axis=0)
    floor_val = np.mean([abs(c[1] - base[1]) for c in controls])
    floor_pred = np.mean([np.abs(c[2] - base[2]).mean() for c in controls])
    gap_steps = np.abs(prod[0] - ref_losses)
    gap_val, gap_pred = abs(prod[1] - ref_val), np.abs(prod[2] - ref_pred).mean()
    floor_tiny = TINY * np.abs(ref_losses)
    # The budget of the steps is the largest floor over the steps, as one floor value per step is a single random draw.
    ratios = {
        "step0": gap_steps[:1] / np.maximum(floor_steps[:1], floor_tiny[:1]),
        "steps": gap_steps.max(keepdims=True) / np.maximum(floor_steps.max(), floor_tiny.max()),
        "final_val_loss": np.array([gap_val / max(floor_val, TINY * abs(ref_val))]),
        "final_pred": np.array([gap_pred / max(floor_pred, TINY * np.abs(ref_pred).mean())]),
    }
    print(
        f"\n{name}: step0 gap={gap_steps[0]:.3e} floor={floor_steps[0]:.3e} ratio={ratios['step0'][0]:.2f}; "
        f"max gap={gap_steps.max():.3e} max floor={floor_steps.max():.3e} ratio={ratios['steps'][0]:.2f}; "
        f"final val gap={gap_val:.3e} floor={floor_val:.3e} ratio={ratios['final_val_loss'][0]:.2f}; "
        f"final pred gap={gap_pred:.3e} floor={floor_pred:.3e} ratio={ratios['final_pred'][0]:.2f}; K={K}, u={u}"
    )
    assert all(r.max() <= K for r in ratios.values()), ratios
