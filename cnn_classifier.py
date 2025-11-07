from typing import Any
from bootleg_aah_cnn_classifier import CNN
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn


class CNNClassifier:
    def __init__(self, learning_rate=0.001, num_epochs=10):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        self.loss_fn = nn.CrossEntropyLoss()
        torch.manual_seed(42)

    def fit(self, X: Tensor, y: Tensor, batch_size=128):
        self.model = CNN().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        X_tensor = torch.FloatTensor(X).view(-1, 1, 28, 28)  # reshape for MNIST
        y_tensor = torch.LongTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.model.train()
        for _ in range(self.num_epochs):
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                # forward pass
                outputs = self.model(X_batch)
                loss = self.loss_fn(outputs, y_batch)

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X):
        if not self.model:
            raise ValueError("fit must be called before calling predict")

        X_tensor = (
            torch.FloatTensor(X)
            .view(-1, 1, 28, 28)
            .to(self.device)  # reshape for MNIST
        )
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.cpu().numpy()

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"num_epochs": self.num_epochs, "learning_rate": self.learning_rate}

    def set_params(
        self, **params
    ):  # allows us to dynamically update the hyperparams of the CNN
        for param, value in params.items():
            setattr(self, param, value)
        return self
