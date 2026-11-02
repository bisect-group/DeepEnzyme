"""Minibatch trainer: batch 1 against the original-protocol trainer, inert padding, and micro-batching without effect on the update."""

import copy

import numpy as np
import torch
from torch import optim

from conftest import train_args
from emulator_bench import minibatch_trainer as mb
from emulator_bench import train_single_target_tvt as single
from emulator_bench.common import set_seed

DEVICE = torch.device("cpu")


def setup(toy, batch_size, max_tokens=10**6, seed=666):
    args = train_args(toy, toy.root / "out", 1, seed=seed, dropout=0.0, batch_size=batch_size, max_tokens=max_tokens, eval_batch_size=4)
    set_seed(seed)
    model, _, _ = single.build_model(args, toy.dict_dir, DEVICE)
    return args, model


def test_padding_is_inert(toy):
    args, model = setup(toy, 1)
    model.train(False)
    split = mb.load_split(toy.dirs["train"])
    with torch.no_grad():
        batch = mb.forward(model, mb.collate(split, np.arange(len(split)), DEVICE), args.layer_output, 0.0).numpy()
        singles = []
        for i in range(len(split)):
            row = single.load_split_dataset(toy.dirs["train"], DEVICE)[i]
            singles.append(model(row[:4], args.layer_output, 0.0).item())
    assert len(set(split.lengths)) > 1
    np.testing.assert_allclose(batch, singles, rtol=1e-4, atol=1e-5)


def test_batch_one_matches_original_trainer(toy):
    args, model = setup(toy, 1)
    reference = copy.deepcopy(model)
    split = mb.load_split(toy.dirs["train"])
    order = np.random.RandomState(666).permutation(len(split))
    dataset = single.shuffled_once(single.load_split_dataset(toy.dirs["train"], DEVICE), 666)
    labels, loss_fn = single.label_array(dataset), torch.nn.MSELoss()
    optimizer = single.make_optimizer(reference, args)
    optimizer_mb = single.make_optimizer(model, args)
    for _ in range(3):
        reference.train(True)
        _, _, ref_pred = single.run_pass(reference, dataset, labels, loss_fn, args, DEVICE, None, "t", optimizer)
        _, pred, step_losses = mb.train_epoch(model, split.take(order), optimizer_mb, args, DEVICE)
        np.testing.assert_allclose(pred * single.LOG10_OF_2, ref_pred, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(step_losses, (pred - split.take(order).labels) ** 2, rtol=1e-4, atol=1e-5)
    for p, q in zip(model.parameters(), reference.parameters()):
        np.testing.assert_allclose(p.detach().numpy(), q.detach().numpy(), atol=5e-4)  # Adam amplifies fp noise of near-zero gradients
    _, final = mb.evaluate(model, mb.load_split(toy.dirs["test"]), args, DEVICE)
    reference.train(False)
    with torch.no_grad():
        expected = [reference(row[:4], args.layer_output, 0.0).item() for row in single.load_split_dataset(toy.dirs["test"], DEVICE)]
    np.testing.assert_allclose(final, expected, rtol=1e-3, atol=1e-4)


def test_micro_batches_do_not_change_the_update(toy):
    results = []
    for max_tokens in (10**6, 20):
        args, model = setup(toy, 6, max_tokens)
        split = mb.load_split(toy.dirs["train"])
        mb.train_epoch(model, split, optim.SGD(model.parameters(), lr=0.1), args, DEVICE)  # SGD: Adam's first step is lr * sign(grad)
        results.append([p.detach().clone() for p in model.parameters()])
    for p, q in zip(*results):
        torch.testing.assert_close(p, q, rtol=1e-3, atol=1e-6)
    assert len(mb.micro_batches(split.lengths, np.arange(6), 20)) > 1
