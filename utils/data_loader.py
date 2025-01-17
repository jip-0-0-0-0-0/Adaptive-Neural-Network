import torch
from torch.utils.data import Dataset, DataLoader
import json
import os

class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def log_data_statistics(data, targets):
    stats = {
        "num_samples": len(data),
        "num_features": data.size(1) if data.dim() > 1 else 1,
        "target_min": targets.min().item(),
        "target_max": targets.max().item(),
        "target_mean": targets.mean().item(),
    }
    os.makedirs("logs", exist_ok=True)
    with open("logs/data_statistics.json", "w") as f:
        json.dump(stats, f, indent=4)
    print("Data statistics logged.")

def create_data_loader(data, targets, batch_size, shuffle=True):
    log_data_statistics(data, targets)
    dataset = CustomDataset(data, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

if __name__ == "__main__":
    data = torch.randn(100, 10)
    targets = torch.randint(0, 2, (100,))
    batch_size = 16

    data_loader = create_data_loader(data, targets, batch_size)

    for batch_idx, (x, y) in enumerate(data_loader):
        print(f"Batch {batch_idx + 1}")
        print("Data:", x)
        print("Targets:", y)
