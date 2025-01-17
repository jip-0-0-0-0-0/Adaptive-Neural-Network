import torch
from torch import nn
import math

def scaled_dot_product_attention(q, k, v, mask=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    return output, attn_weights

class MemoryModule(nn.Module):
    def __init__(self, memory_size, input_dim, hidden_dim, num_heads, dropout=0.1):
        super(MemoryModule, self).__init__()
        self.memory_size = memory_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.key_memory = nn.Parameter(torch.randn(memory_size, hidden_dim))
        self.value_memory = nn.Parameter(torch.randn(memory_size, hidden_dim))
        self.query_projection = nn.Linear(input_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_length, input_dim = x.size()
        queries = self.query_projection(x).view(batch_size, seq_length, self.num_heads, self.hidden_dim // self.num_heads).transpose(1, 2)
        keys = self.key_memory.unsqueeze(0).repeat(batch_size, 1, 1).view(batch_size, self.memory_size, self.num_heads, self.hidden_dim // self.num_heads).transpose(1, 2)
        values = self.value_memory.unsqueeze(0).repeat(batch_size, 1, 1).view(batch_size, self.memory_size, self.num_heads, self.hidden_dim // self.num_heads).transpose(1, 2)
        context, attn_weights = scaled_dot_product_attention(queries, keys, values)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_dim)
        context = self.output_projection(context)
        context = self.dropout(context)
        context = self.layer_norm(context + x)
        return context

class MemoryNetwork(nn.Module):
    def __init__(self, memory_size, input_dim, hidden_dim, num_heads, num_layers, dropout=0.1):
        super(MemoryNetwork, self).__init__()
        self.layers = nn.ModuleList([
            MemoryModule(memory_size, input_dim, hidden_dim, num_heads, dropout) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

if __name__ == "__main__":
    batch_size = 16
    seq_length = 50
    input_dim = 64
    hidden_dim = 256
    memory_size = 128
    num_heads = 8
    num_layers = 6

    model = MemoryNetwork(memory_size, input_dim, hidden_dim, num_heads, num_layers)
    x = torch.randn(batch_size, seq_length, input_dim)
    output = model(x)
    print(output.shape)