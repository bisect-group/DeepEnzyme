import math
import warnings

import torch
from torch import nn
import torch.nn.functional as F


def normalize_adj(adj):
    rowsum = adj.sum(1)
    r_inv_sqrt = torch.pow(rowsum, -0.5)
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
    return torch.mm(torch.mm(r_mat_inv_sqrt, adj), r_mat_inv_sqrt)


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads, dropout):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.attention = nn.MultiheadAttention(out_features, num_heads)

    def forward(self, x, adj):
        x = self.linear(x)
        adj = normalize_adj(adj + torch.eye(adj.size(0), device=adj.device))
        return torch.mm(adj.to(torch.float32), x.to(torch.float32))


class GCN(nn.Module):
    def __init__(self, in_features, hidden_featrures1, hidden_featrures2, out_features, num_heads, dropout):
        super().__init__()
        self.layer1 = GCNLayer(in_features, hidden_featrures2, num_heads, dropout)
        self.layer2 = GCNLayer(hidden_featrures1, hidden_featrures2, num_heads, dropout)
        self.layer3 = GCNLayer(hidden_featrures2, out_features, num_heads, dropout)
        self.layer4 = GCNLayer(in_features, out_features, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        return self.layer4(x, adj)


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, dim):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0), :].to(device=x.device, dtype=x.dtype)


class TransformerBlock(nn.Module):
    def __init__(self, nhead, dropout, d_model, hid_size, layers_trans, max_len):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(hid_size, nhead, hid_size * 4, dropout=dropout)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="enable_nested_tensor is True, but self.use_nested_tensor is False.*",
                category=UserWarning,
            )
            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=layers_trans)
        self.PositionalEncoding = PositionalEncoding(max_len, hid_size)
        self.linear = nn.Linear(hid_size, d_model)

    def forward(self, word_vectors):
        word_vectors = self.PositionalEncoding(word_vectors)
        word_vectors = word_vectors.permute(1, 0, 2)
        word_vectors = self.transformer_encoder(word_vectors)
        word_vectors = word_vectors.permute(1, 0, 2)
        return self.linear(word_vectors).squeeze(0)


class DeepEnzymeBench(nn.Module):
    def __init__(self, n_fingerprint, dim, n_word, layer_output, hidden_dim1, hidden_dim2, dropout, nhead, hid_size, layers_trans):
        super().__init__()
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.embed_wordGCN = nn.Embedding(n_word, dim)
        self.embed_wordTrans = nn.Embedding(n_word, hid_size)
        self.W_out = nn.ModuleList([nn.Linear(3 * dim, 3 * dim) for _ in range(layer_output)])
        self.W_interaction = nn.Linear(3 * dim, 1)
        self.gcn = GCN(dim, hidden_dim1, hidden_dim2, dim, nhead, dropout)
        self.dropout = nn.Dropout(dropout)
        self.smiles_transformer = TransformerBlock(nhead, dropout, dim, hid_size, layers_trans, max_len=n_fingerprint)
        self.protein_transformer = TransformerBlock(nhead, dropout, dim, hid_size, layers_trans, max_len=n_word)
        self.softmax = nn.Softmax(dim=1)
        self.ELU = nn.ELU(1.0)

    def fingerprint_gcn(self, smileadjacency, fingerprints, dropout):
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        return self.gcn(fingerprint_vectors, smileadjacency)

    def seq_transformer(self, words, dropout):
        words = words.unsqueeze(0)
        words = self.embed_wordTrans(words)
        return self.protein_transformer(words)

    def protein_gcn(self, adjacency, words, dropout):
        if hasattr(adjacency, "toarray"):
            adjacency = adjacency.toarray()
        adjacency = torch.as_tensor(adjacency, device=words.device)
        word_vectors = self.embed_wordGCN(words)
        return self.gcn(word_vectors, adjacency)

    def forward(self, inputs, layer_output, dropout):
        fingerprints, smileadjacency, words, seqadjacency = inputs
        substrate_vectors = self.fingerprint_gcn(smileadjacency, fingerprints, dropout)
        substrate_vectors = torch.unsqueeze(torch.mean(substrate_vectors, 0), 0)
        seq_vectors = self.seq_transformer(words, dropout)
        seq_vectors = torch.unsqueeze(torch.mean(seq_vectors, 0), 0)
        protein_vectors = self.protein_gcn(seqadjacency, words, dropout)
        protein_vectors = torch.unsqueeze(torch.mean(protein_vectors, 0), 0)
        cat_vector = torch.cat((substrate_vectors, protein_vectors, seq_vectors), 1)
        for j in range(layer_output):
            cat_vector = F.relu(cat_vector)
            cat_vector = F.dropout(cat_vector, dropout, training=self.training)
            cat_vector = self.W_out[j](cat_vector)
        cat_vector = F.relu(cat_vector)
        cat_vector = F.dropout(cat_vector, dropout, training=self.training)
        return torch.squeeze(self.W_interaction(cat_vector), 0)
