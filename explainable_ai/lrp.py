import torch
from torch import nn

class LayerwiseRelevancePropagation:
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.relevance_logs = []
        self.register_hooks()

    def register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                module.register_forward_hook(self.save_activation(name))

    def save_activation(self, name):
        def hook(module, input, output):
            self.activations[name] = (input[0], output)
        return hook

    def explain(self, x, target_class):
        self.model.eval()
        output = self.model(x)
        relevance = torch.zeros_like(output)
        relevance[:, target_class] = output[:, target_class]
        for name, module in reversed(list(self.model.named_modules())):
            if isinstance(module, nn.Linear):
                relevance = self.propagate_relevance(module, name, relevance)
        self.log_relevance(x, relevance, target_class)
        return relevance

    def propagate_relevance(self, module, name, relevance):
        input_activation, output_activation = self.activations[name]
        weights = module.weight
        z = torch.matmul(input_activation, weights.T) + 1e-9
        s = relevance / z
        c = torch.matmul(s, weights)
        return input_activation * c

    def log_relevance(self, inputs, relevance, target_class):
        log_entry = {
            "inputs": inputs.detach().cpu().numpy().tolist(),
            "relevance": relevance.detach().cpu().numpy().tolist(),
            "target_class": target_class
        }
        self.relevance_logs.append(log_entry)

    def export_logs(self, file_path):
        import json
        with open(file_path, "w") as f:
            json.dump(self.relevance_logs, f, indent=4)
        print(f"Relevance logs exported to {file_path}")

if __name__ == "__main__":
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 5)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = SimpleModel()
    lrp = LayerwiseRelevancePropagation(model)

    x = torch.randn(1, 10)
    relevance = lrp.explain(x, target_class=2)

    print("Input:", x)
    print("Relevance:", relevance)

    lrp.export_logs("relevance_logs.json")
