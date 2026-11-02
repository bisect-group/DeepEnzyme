"""The DeepEnzyme model for R independent runs at once, on padded batches.

Run r has its own parameters (stacked along dimension 0) and takes its own sample. Padded atoms and residues never reach real ones:
padded adjacency rows and columns are zero, attention ignores padded keys, and the pooling is a mean over the real positions.
The maths per run is that of `DeepEnzymeBench` (modeling_fast.py) for one sample.
"""

import torch
import torch.nn.functional as F

LAYER_NORM_EPS = 1e-5  # nn.TransformerEncoderLayer default
ENCODER_KEYS = (
    "self_attn.in_proj_weight",
    "self_attn.in_proj_bias",
    "self_attn.out_proj.weight",
    "self_attn.out_proj.bias",
    "linear1.weight",
    "linear1.bias",
    "linear2.weight",
    "linear2.bias",
    "norm1.weight",
    "norm1.bias",
    "norm2.weight",
    "norm2.bias",
)


def trained_names(layer_output, layers_trans):
    """Names of the parameters that receive a gradient. The other parameters of the model are never used by its forward."""
    names = ["embed_fingerprint.weight", "embed_wordGCN.weight", "embed_wordTrans.weight", "gcn.layer4.linear.weight", "gcn.layer4.linear.bias"]
    names += [f"W_out.{j}.{kind}" for j in range(layer_output) for kind in ("weight", "bias")]
    names += ["W_interaction.weight", "W_interaction.bias"]
    names += [f"protein_transformer.transformer_encoder.layers.{i}.{key}" for i in range(layers_trans) for key in ENCODER_KEYS]
    names += ["protein_transformer.linear.weight", "protein_transformer.linear.bias"]
    return names


def embed(table, index):
    """Per-run embedding lookup. table [R, V, D], index [R, N]. The gradient is dense, as nn.Embedding's."""
    runs, vocab, dim = table.shape
    offset = torch.arange(runs, device=index.device)[:, None] * vocab
    return F.embedding((index + offset).reshape(-1), table.reshape(runs * vocab, dim)).reshape(runs, -1, dim)


def linear(x, weight, bias):
    return torch.baddbmm(bias[:, None, :], x, weight.transpose(1, 2))


def graph_conv(x, adjacency):
    """The original GCN layer: x is already the output of the linear layer; adjacency [R, N, N] without self-loops of the padding."""
    adjacency = adjacency + torch.eye(adjacency.size(1), device=adjacency.device)
    scale = adjacency.sum(2).pow(-0.5)
    scale = torch.where(torch.isinf(scale), torch.zeros_like(scale), scale)
    return torch.bmm(adjacency * scale[:, :, None] * scale[:, None, :], x)


def masked_mean(x, mask):
    return (x * mask[:, :, None]).sum(1) / mask.sum(1)[:, None]


def encoder_layer(x, keep, P, prefix, nhead, dropout, training):
    runs, length, hidden = x.shape
    qkv = linear(x, P[f"{prefix}.self_attn.in_proj_weight"], P[f"{prefix}.self_attn.in_proj_bias"])
    q, k, v = qkv.reshape(runs, length, 3, nhead, hidden // nhead).permute(2, 0, 3, 1, 4)
    attended = F.scaled_dot_product_attention(q, k, v, attn_mask=keep[:, None, None, :], dropout_p=dropout if training else 0.0)
    attended = linear(attended.transpose(1, 2).reshape(runs, length, hidden), P[f"{prefix}.self_attn.out_proj.weight"], P[f"{prefix}.self_attn.out_proj.bias"])
    x = F.layer_norm(x + F.dropout(attended, dropout, training), (hidden,), eps=LAYER_NORM_EPS)
    x = x * P[f"{prefix}.norm1.weight"][:, None, :] + P[f"{prefix}.norm1.bias"][:, None, :]
    inner = F.dropout(F.relu(linear(x, P[f"{prefix}.linear1.weight"], P[f"{prefix}.linear1.bias"])), dropout, training)
    x = F.layer_norm(x + F.dropout(linear(inner, P[f"{prefix}.linear2.weight"], P[f"{prefix}.linear2.bias"]), dropout, training), (hidden,), eps=LAYER_NORM_EPS)
    return x * P[f"{prefix}.norm2.weight"][:, None, :] + P[f"{prefix}.norm2.bias"][:, None, :]


def forward(P, batch, position0, hp, training):
    """Predictions [R] (log2 units) of the R runs. `position0` is the first row of the positional encoding table.

    The original adds pe[:x.size(0)] to a (1, L, hidden) tensor, so every residue gets the same row 0.
    """
    atoms = linear(embed(P["embed_fingerprint.weight"], batch["fingerprints"]), P["gcn.layer4.linear.weight"], P["gcn.layer4.linear.bias"])
    substrate = masked_mean(graph_conv(atoms, batch["substrate_adjacency"]), batch["atom_mask"])

    keep = batch["residue_mask"] > 0
    x = embed(P["embed_wordTrans.weight"], batch["words"]) + position0
    for i in range(hp["layers_trans"]):
        x = encoder_layer(x, keep, P, f"protein_transformer.transformer_encoder.layers.{i}", hp["nhead"], hp["dropout"], training)
    sequence = masked_mean(linear(x, P["protein_transformer.linear.weight"], P["protein_transformer.linear.bias"]), batch["residue_mask"])

    residues = linear(embed(P["embed_wordGCN.weight"], batch["words"]), P["gcn.layer4.linear.weight"], P["gcn.layer4.linear.bias"])
    protein = masked_mean(graph_conv(residues, batch["protein_adjacency"]), batch["residue_mask"])

    x = torch.cat((substrate, protein, sequence), 1)
    for j in range(hp["layer_output"]):
        x = F.dropout(F.relu(x), hp["dropout"], training)
        x = torch.baddbmm(P[f"W_out.{j}.bias"][:, None, :], x[:, None, :], P[f"W_out.{j}.weight"].transpose(1, 2))[:, 0]
    x = F.dropout(F.relu(x), hp["dropout"], training)
    return torch.baddbmm(P["W_interaction.bias"][:, None, :], x[:, None, :], P["W_interaction.weight"].transpose(1, 2))[:, 0, 0]
