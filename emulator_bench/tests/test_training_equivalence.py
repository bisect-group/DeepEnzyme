"""Training script against the authors' training loop (Code/Model/train.py): order, losses, weights, resume, log10 metrics."""

import math

import numpy as np
import pandas as pd
import pytest
import torch

from conftest import DIMS, N_FINGERPRINT, N_WORD, train_args
from emulator_bench import train_single_target_tvt as wrapper
from emulator_bench.common import set_seed

EPOCHS = 14  # the original steps the scheduler from epoch 11 and StepLR(3) first decays at its 3rd step


class OriginalModel:
    """Code.Model.DeepEnzyme with a forward that drops the label and returns a scalar.

    train.py passes the whole 5-tuple of a sample, which forward cannot unpack. It also calls float() on one-element
    arrays, which fails on numpy 2, so labels and outputs are 0-d here. The loss values are the same.
    """

    def __new__(cls, original, *args):
        class Model(original.model.DeepEnzyme):
            def forward(self, data, layer_output, dropout):
                return super().forward(data[:4], layer_output, dropout).reshape(())

        return Model(*args)


def original_run(original, toy, iteration, seed, device="cpu"):
    """The loop of Code/Model/train.py main() on the toy features, with the train list shuffled once by shuffle_dataset.
    """
    train = original.train
    sets = {}
    for name, folder in toy.dirs.items():
        columns = [
            train.load_tensor(str(folder / "fingerprint"), torch.LongTensor),
            train.load_tensor(str(folder / "smileadjacencies"), torch.FloatTensor),
            train.load_tensor(str(folder / "sequences"), torch.LongTensor),
            np.load(folder / "proteinadjacencies.npy", allow_pickle=True),
            [torch.as_tensor(np.asarray(d, dtype=np.float32).reshape(())).to(device) for d in np.load(folder / "logkcat.npy", allow_pickle=True)],
        ]
        sets[name] = list(zip(*columns))
    train.shuffle_dataset(sets["train"], seed)
    torch.manual_seed(seed)
    model = OriginalModel(original, N_FINGERPRINT, DIMS["dim"], N_WORD, DIMS["layer_output"], DIMS["hidden_dim1"], DIMS["hidden_dim2"], 0.3, DIMS["nhead"], DIMS["hid_size"], DIMS["layers_trans"]).to(device)
    trainer, validater, tester = train.Trainer(model, 0.001, 1e-6), train.Validater(model), train.Tester(model)
    log = []
    for epoch in range(1, iteration + 1):
        loss_train, _, rmse_train, _ = trainer.train(sets["train"], DIMS["layer_output"], 0.3)
        loss_val, _, rmse_val, _ = validater.val(sets["val"], DIMS["layer_output"], 0.3)
        tester.test(sets["test"], DIMS["layer_output"], 0.3)
        if epoch // 10 > 0 and epoch % 10 > 0:
            trainer.scheduler.step()
        log.append({"Loss_train": loss_train, "Loss_val": loss_val, "RMSE_train": rmse_train, "RMSE_val": rmse_val, "Lr": trainer.optimizer.param_groups[0]["lr"]})
    model.train(False)
    with torch.no_grad():
        preds = [float(model(data, DIMS["layer_output"], 0.3)) for data in sets["test"]]
    return model, pd.DataFrame(log), np.asarray(preds)


def test_training_matches_original(cpu_original, toy, tmp_path):
    out = tmp_path / "run"
    wrapper.main(train_args(toy, out, EPOCHS))
    model, log, preds = original_run(cpu_original, toy, EPOCHS, 666)

    ours = pd.read_csv(out / "logfile.csv", float_precision="round_trip")
    for column in ("Loss_train", "Loss_val", "Lr"):
        assert ours[column].tolist() == log[column].tolist(), column
    # Reported metrics are in log10 (the original code reports log2).
    for column in ("RMSE_train", "RMSE_val"):
        np.testing.assert_allclose(ours[column], log[column] * math.log10(2), rtol=1e-12)
    assert ours["Lr"].iloc[-1] < ours["Lr"].iloc[0]
    state = torch.load(out / "checkpoint_last.pt", weights_only=False)["model_state"]
    assert all(torch.equal(state[key], value) for key, value in model.state_dict().items())
    saved = pd.read_csv(out / "pred_label_test.csv", float_precision="round_trip")
    np.testing.assert_array_equal(saved["pred"].to_numpy(), preds * math.log10(2))


def test_metrics_are_log10_of_the_log2_values(toy, tmp_path):
    out = tmp_path / "run"
    wrapper.main(train_args(toy, out, 1))
    saved = pd.read_csv(out / "pred_label_test.csv", float_precision="round_trip")
    log2_labels = np.concatenate([row[4] for row in toy.rows["test"]]).astype(np.float64)
    np.testing.assert_allclose(saved["label"], np.log10(2.0**log2_labels), rtol=1e-12)
    results = pd.read_csv(out / "final_results_test.csv", float_precision="round_trip").iloc[0]
    assert results["MAE"] == pytest.approx(np.mean(np.abs(saved["pred"] - saved["label"])), rel=1e-12)


def test_shuffle_is_the_original_permutation():
    items = list(range(50))
    expected = list(items)
    np.random.seed(666)
    np.random.shuffle(expected)
    assert wrapper.shuffled_once(items, 666) == expected
    assert wrapper.shuffled_once(items, 666) != wrapper.shuffled_once(items, 777)


def test_order_is_fixed_over_epochs_and_resume(toy, tmp_path, monkeypatch):
    orders = []
    real = wrapper.train_epoch

    def recording(model, dataset, *args):
        orders.append([float(data[-1]) for data in dataset])
        return real(model, dataset, *args)

    monkeypatch.setattr(wrapper, "train_epoch", recording)
    real_save = wrapper.save_checkpoint

    def interrupt(path, epoch, *args):
        real_save(path, epoch, *args)
        if epoch == 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(wrapper, "save_checkpoint", interrupt)
    with pytest.raises(KeyboardInterrupt):
        wrapper.main(train_args(toy, tmp_path / "run", 4))
    monkeypatch.setattr(wrapper, "save_checkpoint", real_save)
    wrapper.main(train_args(toy, tmp_path / "run", 4))

    labels = [float(row[4][0]) for row in toy.rows["train"]]
    expected = [labels[i] for i in np.random.RandomState(666).permutation(len(labels))]
    assert len(orders) == 4  # 2 epochs, then 2 resumed epochs
    assert all(order == expected for order in orders)


def test_resume_equals_uninterrupted_run(toy, tmp_path, monkeypatch):
    wrapper.main(train_args(toy, tmp_path / "full", 5))

    real_save = wrapper.save_checkpoint

    def interrupt(path, epoch, *args):
        real_save(path, epoch, *args)
        if epoch == 3:
            raise KeyboardInterrupt

    monkeypatch.setattr(wrapper, "save_checkpoint", interrupt)
    with pytest.raises(KeyboardInterrupt):
        wrapper.main(train_args(toy, tmp_path / "resumed", 5))
    monkeypatch.setattr(wrapper, "save_checkpoint", real_save)
    set_seed(1)  # a resumed process starts with unrelated generator states
    wrapper.main(train_args(toy, tmp_path / "resumed", 5))

    full, resumed = (pd.read_csv(tmp_path / name / "logfile.csv", float_precision="round_trip").drop(columns="Time(sec)") for name in ("full", "resumed"))
    pd.testing.assert_frame_equal(full, resumed, check_exact=True)
    states = [torch.load(tmp_path / name / "checkpoint_last.pt", weights_only=False)["model_state"] for name in ("full", "resumed")]
    assert all(torch.equal(value, states[1][key]) for key, value in states[0].items())
    for name in ("train", "val", "test"):
        pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "full" / f"pred_label_{name}.csv", float_precision="round_trip"), pd.read_csv(tmp_path / "resumed" / f"pred_label_{name}.csv", float_precision="round_trip"), check_exact=True)
    assert (tmp_path / "resumed" / "completed.json").exists()
