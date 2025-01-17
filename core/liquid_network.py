import torch
from torch import nn
import math

class LiquidTimeConstant(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dt=0.01, alpha=0.9):
        super(LiquidTimeConstant, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dt = dt
        self.alpha = alpha

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.recurrent_weights = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.recurrent_weights, a=math.sqrt(5))

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        outputs = []
        for t in range(seq_length):
            u_t = self.input_projection(x[:, t])
            h = self.alpha * h + self.dt * torch.tanh(u_t + h @ self.recurrent_weights)
            outputs.append(h)
        outputs = torch.stack(outputs, dim=1)
        return self.output_projection(outputs)

class LiquidNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dt=0.01, alpha=0.9):
        super(LiquidNeuralNetwork, self).__init__()
        self.layers = nn.ModuleList([
            LiquidTimeConstant(input_dim if i == 0 else hidden_dim, hidden_dim, hidden_dim if i < num_layers - 1 else output_dim, dt, alpha)
            for i in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

if __name__ == "__main__":
    batch_size = 16
    seq_length = 100
    input_dim = 64
    hidden_dim = 128
    output_dim = 10
    num_layers = 4
    dt = 0.01
    alpha = 0.9

    model = LiquidNeuralNetwork(input_dim, hidden_dim, output_dim, num_layers, dt, alpha)
    x = torch.randn(batch_size, seq_length, input_dim)
    output = model(x)
    print(output.shape)
