import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class SelfAttention:
    def __init__(self):
        self.data = np.random.normal(size=(50, 50))

        self.weight_key = np.random.normal(size=(50, 50))
        self.weight_query = np.random.normal(size=(50, 50))
        self.weight_value = np.random.normal(size=(50, 50))
    
    def key(self):
        return self.weight_key @ self.data

    def query(self):
        return self.weight_query @ self.data

    def value(self):
        return self.weight_value @ self.data
    
    def forward(self):
        key = self.key()
        query = self.query()
        value = self.value()

        scores = key @ query.T
        scores = scores / key.shape[0]**0.5
        scores = np.exp(scores) / np.sum(scores)

        attn = scores @ value
        return attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        Q = self.query(q)
        K = self.key(k)
        V = self.value(v)

        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)    

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            mask = torch.broadcast_to(mask.unsqueeze(1), (batch_size, self.num_heads, mask.shape[-1], mask.shape[-1]))
            scores = scores.masked_fill(mask == 0, -1e20)

        attention_weights = F.softmax(scores, dim=-1)

        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output = self.out(attention_output)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class EncodingLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncodingLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, vocab_size, max_seq_len, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self._generate_positional_encoding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            EncodingLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
    
    def _generate_positional_encoding(self, max_seq_len, d_model):
        position = torch.arange(max_seq_len).unsqueeze(1)
        i = torch.arange(d_model).unsqueeze(0)
        angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / torch.tensor(d_model, dtype=torch.float32))
        positional_encoding = position * angle_rates
        positional_encoding[:, 0::2] = torch.sin(positional_encoding[:, 0::2])
        positional_encoding[:, 1::2] = torch.cos(positional_encoding[:, 1::2])
        return positional_encoding.unsqueeze(0)

    def forward(self, x, mask=None):
        seq_len = x.size(1)
        x = self.embedding(x)
        x += self.positional_encoding[:, :seq_len, :]
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        self_attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))

        cross_attn_output = self.cross_attn(x, enc_output, enc_output, memory_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, vocab_size, max_seq_len, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self._generate_positional_encoding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
    
    def _generate_positional_encoding(self, max_seq_len, d_model):
        position = torch.arange(max_seq_len).unsqueeze(1)
        i = torch.arange(d_model).unsqueeze(0)
        angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / torch.tensor(d_model, dtype=torch.float32))
        positional_encoding = position * angle_rates
        positional_encoding[:, 0::2] = torch.sin(positional_encoding[:, 0::2])
        positional_encoding[:, 1::2] = torch.cos(positional_encoding[:, 1::2])
        return positional_encoding.unsqueeze(0)

    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        seq_len = x.size(1)
        x = self.embedding(x)
        x += self.positional_encoding[:, :seq_len, :]
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, memory_mask)
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, src_vocab_size, max_seq_len, dropout)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers, tgt_vocab_size, max_seq_len, dropout)
        self.final_layer = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, tgt_mask, memory_mask)
        final_output = self.final_layer(dec_output)
        return final_output
    
    def create_src_mask(self, src, padding_idx):
        return (src != padding_idx).unsqueeze(-2)

    def create_tgt_mask(self, tgt, padding_idx):
        tgt_pad_mask = (tgt != padding_idx).unsqueeze(-2)
        tgt_len = tgt.size(1)
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        return tgt_mask
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    src = torch.tensor([
        [1, 5, 6, 4, 3, 9, 5, 2, 0],
        [1, 8, 7, 3, 4, 5, 6, 7, 2]
    ]).to(device)

    tgt = torch.tensor([
        [1, 7, 4, 3, 5, 9, 2, 0],
        [1, 5, 6, 2, 4, 7, 6, 2]
    ]).to(device)

    src_vocab_size = 10000
    tgt_vocab_size = 10000
    d_model = 512
    num_heads = 8
    d_ff = 2048
    num_layers = 6
    max_seq_len = 100
    dropout = 0.1
    padding_idx = 0

    model = Transformer(
        src_vocab_size, tgt_vocab_size,
        d_model, num_heads, d_ff,
        num_layers, max_seq_len, dropout
    ).to(device)

    src = torch.randint(0, src_vocab_size, (32, max_seq_len)).to(device)
    tgt = torch.randint(0, tgt_vocab_size, (32, max_seq_len)).to(device)
    print("Source Shape:", src.shape)
    print("Target Shape:", tgt.shape)
    src_mask = model.create_src_mask(src, padding_idx).to(device)
    tgt_mask = model.create_tgt_mask(tgt, padding_idx).to(device)

    output = model(src, tgt, src_mask, tgt_mask)

    print("Transformer Output Shape:", output.shape)
