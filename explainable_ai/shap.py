import torch
from torch import nn
import numpy as np

class SHAPExplainer:
    def __init__(self, model, baseline=None):
        self.model = model
        self.baseline = baseline
        if self.baseline is None:
            self.baseline = torch.zeros(1, model.fc1.in_features)

    def explain(self, inputs, num_samples=100):
        inputs = inputs.detach()
        baseline = self.baseline.to(inputs.device)
        shap_values = torch.zeros_like(inputs)

        for i in range(inputs.size(1)):
            perturbation = self._generate_perturbations(inputs, baseline, i, num_samples)
            scores = self._compute_marginal_contributions(inputs, perturbation)
            shap_values[:, i] = scores

        return shap_values

    def _generate_perturbations(self, inputs, baseline, feature_idx, num_samples):
        perturbations = []
        for _ in range(num_samples):
            random_mask = np.random.uniform(0, 1, inputs.size(1))
            random_mask[feature_idx] = 1
            mask = torch.tensor(random_mask, dtype=torch.float32).unsqueeze(0).to(inputs.device)
            perturbation = baseline * (1 - mask) + inputs * mask
            perturbations.append(perturbation)
        return torch.cat(perturbations, dim=0)

    def _compute_marginal_contributions(self, inputs, perturbations):
        original_predictions = self.model(inputs).detach()
        perturbation_predictions = self.model(perturbations).detach()

        marginal_contributions = perturbation_predictions - original_predictions.unsqueeze(0)
        scores = marginal_contributions.mean(dim=0)
        return scores

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
    explainer = SHAPExplainer(model)
    x = torch.randn(1, 10)
    shap_values = explainer.explain(x)
    print("Input:", x)
    print("SHAP Values:", shap_values)
