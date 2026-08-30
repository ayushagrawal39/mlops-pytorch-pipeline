"""Small CIFAR-10 CNN used for training and inference."""
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_model(architecture: str = "cnn", num_classes: int = 10):
    if architecture.lower() not in {"cnn", "simple_cnn"}:
        raise ValueError("Supported architecture: cnn")
    return CNN(num_classes)
