"""The wrapper model against the authors' Code.Model on the CPU."""

import numpy as np
import pytest
import torch

from conftest import DIMS, N_FINGERPRINT, N_WORD, synthetic_rows
from emulator_bench.modeling_fast import DeepEnzymeBench, normalize_adj

ARGS = (N_FINGERPRINT, DIMS["dim"], N_WORD, DIMS["layer_output"], DIMS["hidden_dim1"], DIMS["hidden_dim2"], 0.3, DIMS["nhead"], DIMS["hid_size"], DIMS["layers_trans"])


def make_pair(original, seed):
    torch.manual_seed(seed)
    reference = original.model.DeepEnzyme(*ARGS)
    torch.manual_seed(seed)
    return reference, DeepEnzymeBench(*ARGS)


def inputs_of(row):
    fingerprints, adjacency, words, protein = row[:4]
    return torch.as_tensor(fingerprints), torch.as_tensor(adjacency), torch.as_tensor(words), protein


def test_init_weights_are_identical(cpu_original):
    for seed in (0, 666):
        reference, fast = make_pair(cpu_original, seed)
        assert reference.state_dict().keys() == fast.state_dict().keys()
        assert all(torch.equal(value, fast.state_dict()[key]) for key, value in reference.state_dict().items())


@pytest.mark.parametrize("training", [False, True])
def test_forward_is_identical(cpu_original, training):
    reference, fast = make_pair(cpu_original, 3)
    reference.train(training)
    fast.train(training)
    for row in synthetic_rows(6, 11):
        torch.manual_seed(5)
        expected = reference(inputs_of(row), DIMS["layer_output"], 0.3)
        torch.manual_seed(5)
        assert torch.equal(fast(inputs_of(row), DIMS["layer_output"], 0.3), expected)


@pytest.mark.parametrize("size", [1, 7, 120, 2000])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_normalize_adj_is_bit_exact(cpu_original, size, dtype):
    """Row and column scaling gives the same bits as the two dense products with the diagonal matrix."""
    generator = torch.Generator().manual_seed(size)
    contacts = (torch.rand(size, size, generator=generator) < 0.2).to(dtype)
    contacts = torch.maximum(contacts, contacts.T)
    adj = contacts + torch.eye(size, dtype=dtype)
    assert torch.equal(normalize_adj(adj.clone()), cpu_original.gcn.normalize_adj(adj.clone()))


def test_normalize_adj_zero_row_keeps_zero_scale(cpu_original):
    adj = torch.tensor([[0.0, 0.0, 0.0], [0.0, 2.0, 1.0], [0.0, 1.0, 3.0]])
    assert torch.equal(normalize_adj(adj.clone()), cpu_original.gcn.normalize_adj(adj.clone()))


def test_no_structure_map_is_self_loops_only(cpu_original):
    from emulator_bench.feature_utils import padded_contact_map

    for length in (1, 9):
        adj = torch.as_tensor(padded_contact_map(np.zeros((0, 0)), length).toarray())
        assert torch.equal(normalize_adj(adj + torch.eye(length)).to(torch.float32), torch.eye(length))
