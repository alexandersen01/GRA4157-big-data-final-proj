import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        torch.manual_seed(42)
        super(CNN, self).__init__()
        self.convolutional_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=3, padding=1
            ),  # (N, 1, 28, 28) to (N, 16, 28, 28)
            nn.ReLU(),  # (N, 16, 28, 28)
            nn.MaxPool2d(kernel_size=2, stride=2),  # (N, 16, 28, 28) to (N, 16, 14, 14)
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, padding=1
            ),  # (N, 16, 14, 14) to (N, 32, 14, 14)
            nn.ReLU(),  # (N, 32, 14, 14)
            nn.MaxPool2d(kernel_size=2, stride=2),  # (N, 32, 14, 14) to (N, 32, 7, 7)
        )

        self.fully_connected_layers = nn.Sequential(
            nn.Linear(
                32 * 7 * 7, 64
            ),  # (N, 32 * 7 * 7) to (N, 64) 
            nn.ReLU(),
            nn.Dropout(0.25),  # (N, 64) to (N, 64)
            nn.Linear(64, num_classes),  # (N, 64) to (N, num_classes)
        )

    def forward(self, x):
        x = self.convolutional_layers(x)
        x = torch.flatten(x, 1)
        x = self.fully_connected_layers(x)
        return x
