import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import random

class CandidateModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super(CandidateModel, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(layers):
            self.layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return x

class NeuralArchitectureSearch:
    def __init__(self, input_dim, output_dim, search_space, max_generations, population_size):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.search_space = search_space
        self.max_generations = max_generations
        self.population_size = population_size
        self.population = self._initialize_population()

    def _initialize_population(self):
        population = []
        for _ in range(self.population_size):
            hidden_dim = random.choice(self.search_space['hidden_dim'])
            layers = random.choice(self.search_space['layers'])
            model = CandidateModel(self.input_dim, hidden_dim, self.output_dim, layers)
            population.append((model, 0))
        return population

    def _evaluate_model(self, model, data_loader, criterion):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y in data_loader:
                predictions = model(x)
                loss = criterion(predictions, y)
                total_loss += loss.item()
        return total_loss / len(data_loader)

    def _mutate_model(self, model):
        mutated_layers = random.choice(range(len(model.layers) // 2))
        hidden_dim = random.choice(self.search_space['hidden_dim'])
        model.layers[mutated_layers * 2] = nn.Linear(self.input_dim if mutated_layers == 0 else hidden_dim, hidden_dim)

    def _crossover_models(self, model1, model2):
        child_layers = []
        for i in range(len(model1.layers)):
            if isinstance(model1.layers[i], nn.Linear):
                child_layers.append(random.choice([model1.layers[i], model2.layers[i]]))
            else:
                child_layers.append(model1.layers[i])
        child = CandidateModel(self.input_dim, model1.layers[0].out_features, self.output_dim, len(child_layers) // 2)
        child.layers = nn.ModuleList(child_layers)
        return child

    def run_search(self, data_loader, criterion):
        for generation in range(self.max_generations):
            fitness_scores = []
            for model, _ in self.population:
                score = self._evaluate_model(model, data_loader, criterion)
                fitness_scores.append((model, score))

            fitness_scores.sort(key=lambda x: x[1])
            self.population = fitness_scores[:self.population_size]

            new_population = []
            for i in range(len(self.population)):
                if random.random() < 0.5:
                    new_population.append(self._mutate_model(self.population[i][0]))
                else:
                    new_population.append(self._crossover_models(
                        self.population[random.randint(0, len(self.population) - 1)][0],
                        self.population[random.randint(0, len(self.population) - 1)][0]
                    ))
            self.population = [(model, 0) for model in new_population]

if __name__ == "__main__":
    input_dim = 10
    output_dim = 1
    search_space = {
        'hidden_dim': [32, 64, 128],
        'layers': [2, 3, 4, 5]
    }
    max_generations = 10
    population_size = 10

    criterion = nn.MSELoss()

    x_data = torch.rand(100, input_dim)
    y_data = torch.rand(100, output_dim)
    dataset = TensorDataset(x_data, y_data)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    nas = NeuralArchitectureSearch(input_dim, output_dim, search_space, max_generations, population_size)
    nas.run_search(data_loader, criterion)
