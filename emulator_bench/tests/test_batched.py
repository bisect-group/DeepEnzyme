"""The batched trainer (several runs in one process) against separate single-run trainer runs.

Dropout is off in the comparisons: the batched trainer draws its dropout masks from one RNG stream, so only the stream differs.
The per-run bound is the noise-control bound of test_gpu_noise.py: gap = |batched - single|, floor = |single with the initial
weights perturbed by u = 1e-6 - single|, mean of DRAWS draws. Every ratio gap / floor must be at most K.
"""

from argparse import Namespace

import numpy as np
import pandas as pd
import pytest
import torch

from conftest import DIMS, synthetic_rows, train_args, write_feature_dir, write_tiny_dicts
from emulator_bench import batched_trainer as batched
from emulator_bench import train_single_target_tvt as single
from emulator_bench.batched_model import forward
from emulator_bench.common import set_seed

K, U, DRAWS, TINY = 10, 1e-6, 3, 1e-7
EPOCHS = 14  # the learning rate first decays at epoch 13
SEEDS = (666, 777, 888)
SIZES = ((6, 3, 3), (5, 4, 3), (7, 3, 2))  # train, val, test rows per run; unequal, so runs wait for each other
COLUMNS = ("Loss_train", "Loss_val", "RMSE_test", "Lr")


def make_jobs(root, dims, lengths, device="cpu"):
    jobs, toys = [], []
    dicts = write_tiny_dicts(root / "dicts")
    for i, (seed, sizes) in enumerate(zip(SEEDS, SIZES)):
        dirs = {name: write_feature_dir(root / f"s{i}" / name, synthetic_rows(n, 20 + 3 * i + j, lengths)) for j, (name, n) in enumerate(zip(("train", "val", "test"), sizes))}
        jobs.append(batched.Job(dirs["train"], dirs["val"], dirs["test"], root / f"batched{i}", seed))
        toys.append(Namespace(dirs=dirs, dict_dir=dicts, root=root))
    return jobs, toys, dicts


def batched_args(dicts, device, iteration, dims, group_size=1):
    return Namespace(group_size=group_size, device=device, dict_dir=str(dicts), overwrite=False, lr=1e-3, iteration=iteration, weight_decay=0.01, dropout=0.0, **dims)


def perturbing(monkeypatch, draw):
    """Single-run trainer whose initial weights are perturbed by w += U * |w| * randn."""
    build = single.build_model

    def perturbed(*args, **kwargs):
        model, *rest = build(*args, **kwargs)
        generator = torch.Generator().manual_seed(1234 + draw)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(U * p.abs() * torch.randn(p.shape, generator=generator).to(p.device))
        return (model, *rest)

    monkeypatch.setattr(single, "build_model", perturbed)
    return lambda: monkeypatch.setattr(single, "build_model", build)


def single_run(toy, out, seed, device, dims):
    single.main(train_args(toy, out, EPOCHS, seed=seed, dropout=0.0, device=device, **dims))
    log = pd.read_csv(out / "logfile.csv", float_precision="round_trip")
    state = torch.load(out / "checkpoint_last.pt", weights_only=False, map_location="cpu")["model_state"]
    return log, torch.cat([v.float().flatten() for v in state.values()])


def compare(tmp_path, monkeypatch, device, dims, lengths, scale=1.0):
    jobs, toys, dicts = make_jobs(tmp_path, dims, lengths)
    build = single.build_model

    def scaled(*args, **kwargs):
        model, *rest = build(*args, **kwargs)
        with torch.no_grad():
            for p in model.parameters():
                p.mul_(scale)
        return (model, *rest)

    monkeypatch.setattr(single, "build_model", scaled)
    batched.main(batched_args(dicts, device, EPOCHS, dims), jobs)
    monkeypatch.setattr(single, "build_model", build)
    for r, (job, toy, seed) in enumerate(zip(jobs, toys, SEEDS)):
        log, weights = single_run(toy, tmp_path / f"single{r}", seed, device, dims)
        assert sorted(p.name for p in job.out_dir.iterdir()) == sorted(p.name for p in (tmp_path / f"single{r}").iterdir())
        ours = pd.read_csv(job.out_dir / "logfile.csv", float_precision="round_trip")
        state = torch.load(job.out_dir / "checkpoint_last.pt", weights_only=False, map_location="cpu")["model_state"]
        ours_weights = torch.cat([v.float().flatten() for v in state.values()])
        controls = []
        for draw in range(DRAWS):
            restore = perturbing(monkeypatch, draw)
            controls.append(single_run(toy, tmp_path / f"control{r}_{draw}", seed, device, dims))
            restore()
        ratios = {}
        for column in COLUMNS[:3]:
            gap = np.abs(ours[column] - log[column]).max()
            floor = np.mean([np.abs(c[0][column] - log[column]).max() for c in controls])
            ratios[column] = gap / max(floor, TINY * np.abs(log[column]).max())
        gap = (ours_weights - weights).abs().mean()
        floor = np.mean([(c[1] - weights).abs().mean() for c in controls])
        ratios["weights"] = float(gap / max(floor, TINY * weights.abs().mean()))
        assert ours["Lr"].tolist() == log["Lr"].tolist()
        print(f"\nrun {r}: " + ", ".join(f"{k} {v:.2f}" for k, v in ratios.items()) + f"; K={K}")
        assert max(ratios.values()) <= K, ratios


def test_batched_matches_single_runs_within_noise(tmp_path, monkeypatch):
    compare(tmp_path, monkeypatch, "cpu", DIMS, (6, 14))


def test_scaled_initial_weights_break_the_bound(tmp_path, monkeypatch):
    with pytest.raises(AssertionError):
        compare(tmp_path, monkeypatch, "cpu", DIMS, (6, 14), scale=1.05)


@pytest.mark.gpu
def test_batched_matches_single_runs_within_noise_gpu(tmp_path, monkeypatch):
    compare(tmp_path, monkeypatch, "cuda:0", {"dim": 64, "hidden_dim1": 64, "hidden_dim2": 64, "hid_size": 64, "layers_trans": 3, "layer_output": 3, "nhead": 4}, (300, 420))


def test_padded_forward_and_gradient_match_single_samples(tmp_path):
    jobs, _, dicts = make_jobs(tmp_path, DIMS, (6, 14))
    args = batched_args(dicts, "cpu", 1, DIMS)
    args.dropout = 0.0
    models = []
    for seed in SEEDS:
        set_seed(seed)
        models.append(single.build_model(args, dicts, torch.device("cpu"))[0])
    group = batched.RunGroup(models, args, torch.device("cpu"), torch.tensor(1e-3))
    splits = [batched.load_host_split(j.dirs["train"]) for j in jobs]
    picks = [2, 4, 1]
    key = batched.shape_key(splits, picks)
    host = {n: torch.zeros(shape, dtype=dtype) for n, (shape, dtype) in batched.buffer_shapes(3, key).items()}
    batched.fill_batch(splits, picks, key, {n: t.numpy() for n, t in host.items()})
    Ab, Lb = key[:2]
    batch = {k: host[k] for k in ("fingerprints", "words", "substrate_adjacency")}
    batch["atom_mask"] = (torch.arange(Ab)[None] < host["lengths"][:, :1]).float()
    batch["residue_mask"] = (torch.arange(Lb)[None] < host["lengths"][:, 1:]).float()
    flat = torch.zeros(3 * Lb * Lb + 1)
    flat[host["adjacency_index"]] = host["adjacency_value"]
    batch["protein_adjacency"] = flat[:-1].view(3, Lb, Lb)
    pred = forward(group.P, batch, group.position0, group.hp, False)
    ((pred - host["labels"]) ** 2).sum().backward()
    for r, model in enumerate(models):
        s, i = splits[r], picks[r]
        model.train(False)
        sample = (torch.from_numpy(s["fingerprints"][i]), torch.from_numpy(s["substrate_adjacency"][i]), torch.from_numpy(s["words"][i]), s["protein_adjacency"][i])
        out = model(sample, args.layer_output, 0.0).reshape(())
        ((out - torch.tensor(s["labels"][i])) ** 2).backward()
        assert abs(float(out) - float(pred[r])) <= 1e-5 * max(1.0, abs(float(out)))
        for name, p in model.named_parameters():
            if name in group.P:
                assert (p.grad - group.P[name].grad[r]).abs().max() <= 1e-4 * p.grad.abs().max() + 1e-10, name


def test_resume_equals_uninterrupted(tmp_path):
    jobs, _, dicts = make_jobs(tmp_path / "a", DIMS, (6, 14))
    batched.main(batched_args(dicts, "cpu", 6, DIMS), jobs)
    resumed, _, _ = make_jobs(tmp_path / "b", DIMS, (6, 14))
    batched.main(batched_args(dicts, "cpu", 3, DIMS), resumed)
    for job in resumed:
        (job.out_dir / "completed.json").unlink()
    batched.main(batched_args(dicts, "cpu", 6, DIMS), resumed)
    for whole, part in zip(jobs, resumed):
        a = pd.read_csv(whole.out_dir / "logfile.csv", float_precision="round_trip").drop(columns="Time(sec)")
        b = pd.read_csv(part.out_dir / "logfile.csv", float_precision="round_trip").drop(columns="Time(sec)")
        pd.testing.assert_frame_equal(a, b)
        sa = torch.load(whole.out_dir / "checkpoint_last.pt", weights_only=False, map_location="cpu")["model_state"]
        sb = torch.load(part.out_dir / "checkpoint_last.pt", weights_only=False, map_location="cpu")["model_state"]
        assert all(torch.equal(sa[k], sb[k]) for k in sa)


def test_groups_of_several_runs_match_groups_of_one(tmp_path):
    jobs, _, dicts = make_jobs(tmp_path / "a", DIMS, (6, 14))
    batched.main(batched_args(dicts, "cpu", 4, DIMS, group_size=1), jobs)
    grouped, _, _ = make_jobs(tmp_path / "b", DIMS, (6, 14))
    batched.main(batched_args(dicts, "cpu", 4, DIMS, group_size=3), grouped)
    for one, three in zip(jobs, grouped):
        a = pd.read_csv(one.out_dir / "logfile.csv").drop(columns="Time(sec)")
        b = pd.read_csv(three.out_dir / "logfile.csv").drop(columns="Time(sec)")
        pd.testing.assert_frame_equal(a, b, rtol=1e-5, atol=1e-6)
