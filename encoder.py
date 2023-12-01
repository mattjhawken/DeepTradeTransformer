import torch
import math
from torch import nn, Tensor


class PositionalEncoder(nn.Module):
    def __init__(self, dropout, max_seq_len):
        super(PositionalEncoder, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.pe = torch.zeros(max_seq_len, max_seq_len)
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, max_seq_len, 2) * (-math.log(10_000) / max_seq_len))

        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        y = self.pe[:x.size(1), :x.size(2)]
        return x + y


class Attention(nn.Module):
    def __init__(self, embedding_size, n_heads):
        super(Attention, self).__init__()

        self.embedding_size = embedding_size
        self.n_heads = n_heads
        self.head_dim = embedding_size // n_heads

        assert self.head_dim * n_heads == embedding_size, "Embedding must be divisible by n_heads!"

    def forward(self, values: Tensor, keys: Tensor, queries: Tensor, mask):
        n_batch = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # Split embeddings into heads
        values = values.view(n_batch, value_len, self.n_heads, self.head_dim)
        keys = keys.view(n_batch, key_len, self.n_heads, self.head_dim)
        queries = queries.view(n_batch, query_len, self.n_heads, self.head_dim)

        # Multiply keys and queries to get correlation matrices
        corr = torch.einsum("bkhd,bqhd->bhqk", [queries, keys])

        if mask is not None:
            corr = corr.masked_fill(mask == 0, float("-1e20"))

        # Softmax attention
        attention = torch.softmax(corr / (self.embedding_size ** (1/2)), dim=3)
        attention = torch.einsum("bhqx,bxhd->bqhd", [attention, values]).reshape(
            n_batch, query_len, self.embedding_size
        )

        return attention


class TransformerBlock(nn.Module):
    def __init__(self, view_size, embedding_size, n_heads, dropout, fwex):
        super(TransformerBlock, self).__init__()

        self.attention = Attention(embedding_size, n_heads)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)

        self.values = nn.Linear(view_size, embedding_size, bias=False)
        self.keys = nn.Linear(view_size, embedding_size, bias=False)
        self.queries = nn.Linear(view_size, embedding_size, bias=False)

        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, fwex),
            nn.ReLU(),
            nn.Linear(fwex, embedding_size),
            nn.LayerNorm(embedding_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        values = self.values(x)
        keys = self.keys(x)
        queries = self.queries(x)

        attention = self.attention(values, keys, queries, mask)

        x = self.dropout(self.norm1(attention + queries))
        fwd = self.feed_forward(x)
        x = self.dropout(self.norm2(fwd + x))

        return x


class Encoder(nn.Module):
    def __init__(self, embedding_size, n_layers, n_heads, fwex, dropout, view_size):
        super(Encoder, self).__init__()

        self.embedding_size = embedding_size
        self.pe = PositionalEncoder(dropout=0.1, max_seq_len=view_size)

        self.layers = nn.ModuleList([
                TransformerBlock(
                    view_size, embedding_size, n_heads, dropout=dropout, fwex=fwex,
                ) for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.pe(x)

        for layer in self.layers:
            x = layer(x, mask)

        return x
