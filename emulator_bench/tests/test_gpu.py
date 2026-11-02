"""CUDA checks. They run only with --gpu (verify_equivalence.sh) and need a visible CUDA device."""

import numpy as np
import pandas as pd
import pytest
import torch

from conftest import DIMS, synthetic_rows, train_args
from emulator_bench import train_single_target_tvt as wrapper
from emulator_bench.modeling_fast import normalize_adj
from test_model_equivalence import inputs_of, make_pair
from test_training_equivalence import original_run

pytestmark = pytest.mark.gpu


@pytest.fixture
def gpu_original(original):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device visible")
    original.device["device"] = "cuda"
    yield original
    original.device["device"] = "cpu"


@pytest.mark.parametrize("size", [200, 2000])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_normalize_adj_is_bit_exact_on_cuda(gpu_original, size, dtype):
    """Bound: 0. Row and column scaling multiplies the same two numbers in the same order as the dense products."""
    contacts = (torch.rand(size, size, device="cuda") < 0.2).to(dtype)
    adj = torch.maximum(contacts, contacts.T) + torch.eye(size, dtype=dtype, device="cuda")
    assert torch.equal(normalize_adj(adj.clone()), gpu_original.gcn.normalize_adj(adj.clone()))


def test_forward_matches_original_on_cuda(gpu_original):
    """Bound: 0. The wrapper forward and its pinned, non-blocking adjacency copy give the bits of the original forward."""
    reference, fast = make_pair(gpu_original, 3)
    reference, fast = reference.to("cuda").eval(), fast.to("cuda").eval()
    for row in synthetic_rows(6, 11):
        data = tuple(t.cuda() if torch.is_tensor(t) else t for t in inputs_of(row))
        assert torch.equal(fast(data, DIMS["layer_output"], 0.3), reference(data, DIMS["layer_output"], 0.3))


def test_training_matches_original_on_cuda(gpu_original, toy, tmp_path):
    """Bounds: relative loss difference per epoch 1e-5, weight difference 1e-5.

    Both runs use fp32 kernels. Embedding and attention backward kernels can differ in the last bits between runs on a GPU.
    """
    out = tmp_path / "run"
    wrapper.main(train_args(toy, out, 3, device="cuda:0"))
    model, log, _ = original_run(gpu_original, toy, 3, 666, device="cuda")
    ours = pd.read_csv(out / "logfile.csv", float_precision="round_trip")
    for column in ("Loss_train", "Loss_val"):
        np.testing.assert_allclose(ours[column], log[column], rtol=1e-5)
    state = torch.load(out / "checkpoint_last.pt", weights_only=False)["model_state"]
    assert max((state[key].cpu() - value.cpu()).abs().max().item() for key, value in model.state_dict().items()) < 1e-5
