import torch
from torch import nn
from torch.nn import functional as F
import math

class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, output_dim, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, dropout)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 4, dropout)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.layer_norm(x)
        x = self.output_projection(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = math.sqrt(hidden_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * -(math.log(10000.0) / hidden_dim))
        pe = torch.zeros(max_len, hidden_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x * self.scale
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class MultiHeadedAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super(MultiHeadedAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.qkv_proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_length, hidden_dim = x.size()
        qkv = self.qkv_proj(x).view(batch_size, seq_length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        scores = torch.einsum('bqhd,bkhd->bhqk', q, k) / self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.einsum('bhqk,bkhd->bqhd', attn_weights, v).contiguous()
        context = context.view(batch_size, seq_length, hidden_dim)
        return self.out_proj(context)

class FeedForward(nn.Module):
    def __init__(self, hidden_dim, feedforward_dim, dropout):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, feedforward_dim)
        self.linear2 = nn.Linear(feedforward_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, feedforward_dim, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(hidden_dim, num_heads, dropout)
        self.feed_forward = FeedForward(hidden_dim, feedforward_dim, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.self_attn(x)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        return x

if __name__ == "__main__":
    batch_size = 16
    seq_length = 100
    input_dim = 64
    hidden_dim = 256
    num_heads = 8
    num_layers = 12
    output_dim = 10

    model = TransformerModel(input_dim, hidden_dim, num_heads, num_layers, output_dim)
    x = torch.randn(batch_size, seq_length, input_dim)
    output = model(x)
    print(output.shape)