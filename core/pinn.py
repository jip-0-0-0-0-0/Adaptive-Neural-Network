import torch
from torch import nn
import math

class PhysicsInformedLoss(nn.Module):
    def __init__(self, physics_fn):
        super(PhysicsInformedLoss, self).__init__()
        self.physics_fn = physics_fn

    def forward(self, predictions, inputs):
        physics_residual = self.physics_fn(predictions, inputs)
        data_loss = torch.mean((predictions - inputs[:, -1].unsqueeze(-1))**2)
        physics_loss = torch.mean(physics_residual**2)
        return data_loss + physics_loss

class PINNLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PINNLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

class PhysicsInformedNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, physics_fn):
        super(PhysicsInformedNeuralNetwork, self).__init__()
        self.pinn_layer = PINNLayer(input_dim, hidden_dim, output_dim)
        self.loss_fn = PhysicsInformedLoss(physics_fn)

    def forward(self, x):
        return self.pinn_layer(x)

    def compute_loss(self, predictions, inputs):
        return self.loss_fn(predictions, inputs)

if __name__ == "__main__":
    def physics_fn(predictions, inputs):
        time = inputs[:, 0]
        position = predictions[:, 0]
        velocity = torch.gradient(position, spacing=time)
        acceleration = torch.gradient(velocity, spacing=time)
        residual = acceleration + 9.8
        return residual

    batch_size = 16
    input_dim = 2
    hidden_dim = 128
    output_dim = 1
    model = PhysicsInformedNeuralNetwork(input_dim, hidden_dim, output_dim, physics_fn)
    
    inputs = torch.rand(batch_size, input_dim)
    predictions = model(inputs)
    loss = model.compute_loss(predictions, inputs)

    print(f"Predictions: {predictions}")
    print(f"Loss: {loss}")
