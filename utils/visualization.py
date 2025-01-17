import matplotlib.pyplot as plt
import numpy as np

def plot_loss_curve(train_losses, val_losses=None):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    if val_losses is not None:
        plt.plot(val_losses, label='Validation Loss', linewidth=2, linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_feature_importance(feature_names, importances):
    indices = np.argsort(importances)[::-1]
    sorted_feature_names = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    plt.figure(figsize=(12, 8))
    plt.barh(sorted_feature_names, sorted_importances, color='skyblue')
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()
    plt.grid(True, axis='x')
    plt.show()

if __name__ == "__main__":
    train_losses = np.random.rand(20)
    val_losses = np.random.rand(20)
    plot_loss_curve(train_losses, val_losses)

    feature_names = [f'Feature {i+1}' for i in range(10)]
    importances = np.random.rand(10)
    plot_feature_importance(feature_names, importances)
