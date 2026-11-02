"""Packed-token trainer against the padded minibatch trainer (CPU): same steps, inert packing, resume and completion guard."""

import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch

from conftest import train_args
from emulator_bench import minibatch_trainer as mt
from emulator_bench import packed_trainer as pk
from emulator_bench import train_single_target_tvt as single
from emulator_bench.common import set_seed

DEVICE = torch.device("cpu")
EXTRA = {"batch_size": 3, "eval_batch_size": 4, "eval_every": 1, "bf16": False, "max_tokens": 10**6}


def setup(toy, **overrides):
    args = train_args(toy, toy.root / "out", 1, **{**EXTRA, "dropout": 0.0, **overrides})
    set_seed(args.seed)
    model, _, _ = single.build_model(args, toy.dict_dir, DEVICE)
    return args, model


@pytest.mark.parametrize("max_tokens", [10**6, 20])
def test_steps_match_padded_trainer(toy, max_tokens):
    args, model = setup(toy)
    padded_model = copy.deepcopy(model)
    split = mt.load_split(toy.dirs["train"])
    packed = pk.Packed(split)
    optimizer, padded_optimizer = (single.make_optimizer(m, args) for m in (model, padded_model))
    args.max_tokens = max_tokens
    for _ in range(3):
        _, pred, losses = pk.train_epoch(model, packed, optimizer, args, DEVICE)
        _, padded_pred, padded_losses = mt.train_epoch(padded_model, split, padded_optimizer, args, DEVICE)
        np.testing.assert_allclose(losses, padded_losses, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(pred, padded_pred, rtol=1e-4, atol=1e-5)
    for p, q in zip(model.parameters(), padded_model.parameters()):
        np.testing.assert_allclose(p.detach().numpy(), q.detach().numpy(), atol=5e-4)  # Adam amplifies fp noise of near-zero gradients
    test_split = mt.load_split(toy.dirs["test"])
    np.testing.assert_allclose(pk.evaluate(model, pk.Packed(test_split), args, DEVICE)[1], mt.evaluate(padded_model, test_split, args, DEVICE)[1], rtol=1e-3, atol=1e-4)
    assert len(pk.token_chunks(split.lengths, np.arange(6), 20)) > 1


def test_eval_forward_matches_with_dropout_configured(toy):
    args, model = setup(toy, dropout=0.3)
    model.train(False)
    split = mt.load_split(toy.dirs["train"])
    idx, store = np.arange(len(split)), pk.Packed(split)
    with torch.no_grad():
        padded = mt.forward(model, mt.collate(split, idx, DEVICE), args.layer_output, args.dropout)
        packed = pk.forward(model, pk.collate(store, idx, DEVICE), args.layer_output, args.dropout)
    np.testing.assert_allclose(packed.numpy(), padded.numpy(), rtol=1e-4, atol=1e-6)
    model.train(True)
    with torch.no_grad():
        assert not torch.allclose(pk.forward(model, pk.collate(store, idx, DEVICE), args.layer_output, args.dropout), packed)


def test_packing_is_inert(toy):
    args, model = setup(toy)
    model.train(False)
    split = mt.load_split(toy.dirs["train"])
    store = pk.Packed(split)
    with torch.no_grad():
        batch = pk.forward(model, pk.collate(store, np.arange(len(split)), DEVICE), args.layer_output, 0.0).numpy()
        singles = [pk.forward(model, pk.collate(store, np.array([i]), DEVICE), args.layer_output, 0.0).item() for i in range(len(split))]
    assert len(set(split.lengths)) > 1
    np.testing.assert_allclose(batch, singles, rtol=1e-4, atol=1e-5)


def test_collate_matches_padded_collate(toy):
    split = mt.load_split(toy.dirs["train"])
    idx = np.array([4, 1, 5, 0])
    packed, padded = pk.collate(pk.Packed(split), idx, DEVICE), mt.collate(split, idx, DEVICE)
    for key in ("fingerprints", "atoms", "lengths", "flat_words", "labels"):
        torch.testing.assert_close(packed[key], padded[key])
    for key in ("substrate", "protein"):
        for a, b in zip(packed[key], padded[key]):
            torch.testing.assert_close(a, b)
    torch.testing.assert_close(packed["offsets"], torch.tensor([0, *np.cumsum(split.lengths[idx])]))


def run(toy, out_name, iteration, **overrides):
    out = toy.root / out_name
    pk.main(train_args(toy, out, iteration, **{**EXTRA, **overrides}))
    return out


def test_resume_equals_uninterrupted(toy):
    full = run(toy, "full", 3)
    part = run(toy, "part", 2)
    (part / "completed.json").unlink()
    run(toy, "part", 3)
    for name in ("pred_label_test.csv", "pred_label_val.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(full / name), pd.read_csv(part / name), rtol=1e-6, atol=1e-9)
    columns = [c for c in pk.HEADERS if c != "Time(sec)"]
    pd.testing.assert_frame_equal(pd.read_csv(full / "logfile.csv")[columns], pd.read_csv(part / "logfile.csv")[columns], rtol=1e-6, atol=1e-9)
    assert json.loads((part / "completed.json").read_text()) == {"completed": True, "final_epoch": 3}


def test_completion_guard(toy):
    out = run(toy, "done", 1)
    run(toy, "done", 1)  # same epoch count: nothing to do
    with pytest.raises(RuntimeError, match="completed 1 epochs"):
        run(toy, "done", 2)
    assert json.loads((out / "completed.json").read_text())["final_epoch"] == 1
