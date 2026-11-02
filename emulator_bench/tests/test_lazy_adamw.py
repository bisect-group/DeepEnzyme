"""Lazy AdamW for the word embedding tables against torch AdamW (dense), with the noise-control bound of test_gpu_noise.py.

gap = |lazy - dense|; floor = mean over DRAWS of |dense with the initial weights perturbed by u = 1e-6 (relative) - dense|; every
ratio gap / floor must be at most K. The CPU tests use the pure-torch reference of the kernel against torch AdamW; the GPU tests use the Triton kernel against fused AdamW,
whose float32 constants (1 - beta, 1 - lr * wd) differ from the double constants of the CPU optimizer by about 1e-5 in v.
"""

from argparse import Namespace

import numpy as np
import pandas as pd
import pytest
import torch

from conftest import DIMS, synthetic_rows, train_args, write_feature_dir, write_tiny_dicts
from emulator_bench import lazy_adamw as lazy_module
from emulator_bench import lazy_train
from emulator_bench import train_single_target_tvt as single

K, U, DRAWS, TINY = 10, 1e-6, 3, 1e-7
PRODUCTION = {"dim": 64, "hidden_dim1": 64, "hidden_dim2": 64, "hid_size": 64, "layers_trans": 3, "layer_output": 3, "nhead": 4}


def ratio(gap, floors, reference):
    return float(gap / max(np.mean(floors), TINY * float(reference)))


# Toy optimizer: a table with a skewed, random sparse access (a few frequent rows, a long tail), 2,000 steps, 20 epochs of 100 steps.
V, D, STEPS, PER_EPOCH, LR = 400, 8, 2000, 100, 1e-2


def toy_run(kind, device, perturb=0.0, draw=0, catch_up=True):
    generator = torch.Generator().manual_seed(0)
    start, target = torch.randn(V, D, generator=generator), torch.randn(V, D, generator=generator)
    if perturb:
        start = start + perturb * start.abs() * torch.randn(V, D, generator=torch.Generator().manual_seed(100 + draw))
    weight, target = torch.nn.Parameter(start.to(device)), target.to(device)
    rng = np.random.RandomState(1)
    probability = 1.0 / np.arange(1, V + 1) ** 1.2
    probability /= probability.sum()
    if kind == "dense":
        optimizer = torch.optim.AdamW([weight], lr=LR, weight_decay=0.01, fused=device != "cpu")
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, 0.9)
    else:
        lazy = lazy_module.LazyRows({"w": weight}, lazy_module.Schedule(LR, 0.01, STEPS // PER_EPOCH, PER_EPOCH, device), 0.01, use_kernel=kind == "kernel")
    for step in range(1, STEPS + 1):
        rows = torch.as_tensor(np.unique(rng.choice(V, size=rng.randint(3, 20), p=probability)), device=device)
        if kind == "dense":
            grad = torch.zeros_like(weight)
            grad[rows] = (weight.data[rows] - target[rows]) * 0.5
            weight.grad = grad
            optimizer.step()
            epoch = step // PER_EPOCH
            if step % PER_EPOCH == 0 and epoch // 10 > 0 and epoch % 10 > 0:
                scheduler.step()
        else:
            if catch_up:
                lazy.advance(rows, step - 1)
            lazy.step(rows, {"w": (weight.data[rows] - target[rows]) * 0.5}, step)
    if kind == "dense":
        state = optimizer.state[weight]
        return [t.detach().cpu() for t in (weight, state["exp_avg"], state["exp_avg_sq"])]
    lazy.finalize(STEPS)
    return [weight.detach().cpu(), lazy.exp_avg["w"].cpu(), lazy.exp_avg_sq["w"].cpu()]


def toy_ratios(kind, device, catch_up=True):
    dense = toy_run("dense", device)
    lazy = toy_run(kind, device, catch_up=catch_up)
    floors = [toy_run("dense", device, U, draw) for draw in range(DRAWS)]
    return {name: ratio((lazy[i] - dense[i]).abs().max(), [(f[i] - dense[i]).abs().max() for f in floors], dense[i].abs().max()) for i, name in enumerate(("p", "m", "v"))}


def test_lazy_matches_dense_on_toy_table():
    ratios = toy_ratios("reference", "cpu")
    print("\ntoy reference:", {k: round(v, 3) for k, v in ratios.items()}, f"K={K}")
    assert max(ratios.values()) <= K, ratios


def test_lazy_without_catch_up_breaks_the_bound():
    assert max(toy_ratios("reference", "cpu", catch_up=False).values()) > K


@pytest.mark.gpu
def test_kernel_matches_dense_on_toy_table():
    ratios = toy_ratios("kernel", "cuda:0")
    print("\ntoy kernel:", {k: round(v, 3) for k, v in ratios.items()}, f"K={K}")
    assert max(ratios.values()) <= K, ratios


# Full model on the toy rows: lazy trainer against the single-run trainer, 14 epochs (the learning rate decays from epoch 13), dropout 0.
EPOCHS = 14


def make_toy(root, lengths):
    dirs = {name: write_feature_dir(root / name, synthetic_rows(n, 10 + i, lengths)) for i, (name, n) in enumerate((("train", 6), ("val", 3), ("test", 3)))}
    return Namespace(dirs=dirs, dict_dir=write_tiny_dicts(root / "dicts"), root=root)


def vectors(out):
    """Final log and flat vectors of the model weights and of the optimizer moments of a run."""
    log = pd.read_csv(out / "logfile.csv", float_precision="round_trip")
    checkpoint = torch.load(out / "checkpoint_last.pt", weights_only=False, map_location="cpu")
    weights = torch.cat([v.float().flatten() for v in checkpoint["model_state"].values()])
    moments = torch.cat([entry[k].flatten() for _, entry in sorted(checkpoint["optimizer_state"]["state"].items()) for k in ("exp_avg", "exp_avg_sq")])
    return log, weights, moments


def run_dense(toy, out, seed, device, dims, epochs=EPOCHS, perturb=0.0, draw=0, scale=1.0, monkeypatch=None):
    build = single.build_model
    if perturb or scale != 1.0:

        def changed(*args, **kwargs):
            model, *rest = build(*args, **kwargs)
            generator = torch.Generator().manual_seed(1234 + draw)
            with torch.no_grad():
                for p in model.parameters():
                    p.mul_(scale).add_(perturb * p.abs() * torch.randn(p.shape, generator=generator).to(p.device))
            return (model, *rest)

        monkeypatch.setattr(single, "build_model", changed)
    single.main(train_args(toy, out, epochs, seed=seed, dropout=0.0, device=device, fused_adamw=device != "cpu", **dims))
    monkeypatch and monkeypatch.setattr(single, "build_model", build)
    return vectors(out)


def run_lazy(toy, out, seed, device, dims, epochs=EPOCHS, scale=1.0, monkeypatch=None, **flags):
    build = single.build_model
    if scale != 1.0:

        def scaled(*args, **kwargs):
            model, *rest = build(*args, **kwargs)
            with torch.no_grad():
                for p in model.parameters():
                    p.mul_(scale)
            return (model, *rest)

        monkeypatch.setattr(single, "build_model", scaled)
    lazy_train.main(train_args(toy, out, epochs, seed=seed, dropout=0.0, device=device, **dims, **flags))
    monkeypatch and monkeypatch.setattr(single, "build_model", build)
    return vectors(out)


def compare_runs(candidate, reference, floors):
    ratios = {}
    for column in ("Loss_train", "Loss_val", "RMSE_test"):
        gap = np.abs(candidate[0][column] - reference[0][column]).max()
        ratios[column] = ratio(gap, [np.abs(f[0][column] - reference[0][column]).max() for f in floors], np.abs(reference[0][column]).max())
    for i, name in ((1, "weights"), (2, "moments")):
        ratios[name] = ratio((candidate[i] - reference[i]).abs().mean(), [(f[i] - reference[i]).abs().mean() for f in floors], reference[i].abs().mean())
    assert candidate[0]["Lr"].tolist() == reference[0]["Lr"].tolist()
    return ratios


def floors_for(toy, root, seed, device, dims, monkeypatch, epochs=EPOCHS):
    return [run_dense(toy, root / f"floor{d}", seed, device, dims, epochs, perturb=U, draw=d, monkeypatch=monkeypatch) for d in range(DRAWS)]


def full_model_ratios(tmp_path, monkeypatch, device, dims, lengths, scale=1.0, **flags):
    toy = make_toy(tmp_path, lengths)
    dense = run_dense(toy, tmp_path / "dense", 666, device, dims, monkeypatch=monkeypatch)
    lazy = run_lazy(toy, tmp_path / "lazy", 666, device, dims, scale=scale, monkeypatch=monkeypatch, **flags)
    ratios = compare_runs(lazy, dense, floors_for(toy, tmp_path, 666, device, dims, monkeypatch))
    print("\nlazy vs dense:", {k: round(v, 3) for k, v in ratios.items()}, f"K={K}")
    return ratios


def test_lazy_trainer_matches_single_run_within_noise(tmp_path, monkeypatch):
    assert max(full_model_ratios(tmp_path, monkeypatch, "cpu", DIMS, (6, 14)).values()) <= K


def test_scaled_initial_weights_break_the_bound(tmp_path, monkeypatch):
    assert max(full_model_ratios(tmp_path, monkeypatch, "cpu", DIMS, (6, 14), scale=1.05).values()) > K


@pytest.mark.gpu
def test_lazy_trainer_matches_single_run_within_noise_gpu(tmp_path, monkeypatch):
    assert max(full_model_ratios(tmp_path, monkeypatch, "cuda:0", PRODUCTION, (300, 420)).values()) <= K


def test_resume_equals_uninterrupted(tmp_path, monkeypatch):
    toy = make_toy(tmp_path, (6, 14))
    whole = run_lazy(toy, tmp_path / "whole", 666, "cpu", DIMS)
    run_lazy(toy, tmp_path / "part", 666, "cpu", DIMS, epochs=7)
    (tmp_path / "part" / "completed.json").unlink()
    resumed = run_lazy(toy, tmp_path / "part", 666, "cpu", DIMS)
    ratios = compare_runs(resumed, whole, floors_for(toy, tmp_path, 666, "cpu", DIMS, monkeypatch))
    print("\nlazy resume:", {k: round(v, 3) for k, v in ratios.items()})
    assert max(ratios.values()) <= K


@pytest.mark.parametrize("first, second", [("dense", "lazy"), ("lazy", "dense")])
def test_checkpoints_load_in_both_directions(tmp_path, monkeypatch, first, second):
    toy = make_toy(tmp_path, (6, 14))
    runners = {"dense": run_dense, "lazy": run_lazy}
    dense = run_dense(toy, tmp_path / "dense", 666, "cpu", DIMS, monkeypatch=monkeypatch)
    runners[first](toy, tmp_path / "mixed", 666, "cpu", DIMS, epochs=7, **({"monkeypatch": monkeypatch}))
    (tmp_path / "mixed" / "completed.json").unlink()
    mixed = runners[second](toy, tmp_path / "mixed", 666, "cpu", DIMS, monkeypatch=monkeypatch)
    ratios = compare_runs(mixed, dense, floors_for(toy, tmp_path, 666, "cpu", DIMS, monkeypatch))
    print(f"\n{first} then {second}:", {k: round(v, 3) for k, v in ratios.items()})
    assert max(ratios.values()) <= K


def test_sparse_adjacency_matches_dense_graph_convolution():
    from emulator_bench.modeling_fast import normalize_adj
    import scipy.sparse as sp

    generator = np.random.RandomState(0)
    for dtype in (np.int32, np.float64):
        dense = (generator.rand(30, 30) < 0.2).astype(dtype)
        dense = np.maximum(dense, dense.T)
        if dtype == np.float64:
            dense[3, 3] = dense[9, 9] = 1.0  # unit diagonal of positions without a C-alpha
        x = torch.randn(30, 8)
        expected = torch.mm(normalize_adj(torch.as_tensor(dense) + torch.eye(30)).to(torch.float32), x)
        got = torch.sparse.mm(lazy_train.normalized_sparse(sp.csr_matrix(dense), torch.device("cpu")), x)
        assert (got - expected).abs().max() <= 1e-6 * expected.abs().max()


def test_lazy_trainer_with_sparse_adjacency_matches_single_run_within_noise(tmp_path, monkeypatch):
    assert max(full_model_ratios(tmp_path, monkeypatch, "cpu", DIMS, (6, 14), sparse_adjacency=True).values()) <= K


@pytest.mark.gpu
def test_lazy_trainer_with_sparse_adjacency_matches_single_run_within_noise_gpu(tmp_path, monkeypatch):
    assert max(full_model_ratios(tmp_path, monkeypatch, "cuda:0", PRODUCTION, (300, 420), sparse_adjacency=True).values()) <= K
