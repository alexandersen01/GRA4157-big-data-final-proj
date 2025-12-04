from typing import Any, Optional
from simple_bootleg_cnn_classifier import CNN
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
import numpy as np


class CNNClassifier:
    def __init__(
        self,
        learning_rate=0.001,
        num_epochs=10,
        model=None,
        input_shape=(1, 28, 28),
        seed=42,
    ):
        """
        Args:
            learning_rate: Initial learning rate for optimizer
            num_epochs: Number of training epochs
            model: Custom CNN model (if None, uses default CNN)
            input_shape: (channels, height, width) - e.g., (3, 32, 32) for cifar-10
            seed: Random seed for reproducibility
        """
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.input_shape = input_shape
        self.seed = seed
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() 
            else "mps" if torch.backends.mps.is_available() 
            else "cpu"
            )
        self.loss_fn = nn.CrossEntropyLoss()
        self.custom_model = model
        self.model = None
        self.history = {}
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

    def fit(self, X: Tensor, y: Tensor, batch_size=128):
        if self.custom_model is not None:
            self.model = self.custom_model.to(self.device)
        else:
            self.model = CNN().to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        #reshape based on input_shape
        channels, height, width = self.input_shape
        X_tensor = torch.FloatTensor(X).view(-1, channels, height, width)
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

    def fit_with_tracking(
        self,
        X_train: Tensor,
        y_train: Tensor,
        X_val: Optional[Tensor] = None,
        y_val: Optional[Tensor] = None,
        batch_size=128,
        verbose=True,
    ):
        if self.custom_model is not None:
            self.model = self.custom_model.to(self.device)
        else:
            self.model = CNN().to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        #reshape based on input_shape
        channels, height, width = self.input_shape
        X_train_tensor = torch.FloatTensor(X_train).view(-1, channels, height, width)
        y_train_tensor = torch.LongTensor(y_train)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        #init history tracking
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": [],
        }

        for epoch in range(self.num_epochs):
            #training
            self.model.train()
            train_losses = []

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                outputs = self.model(X_batch)
                loss = self.loss_fn(outputs, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)
            self.history["train_loss"].append(avg_train_loss)

            #val phase (if validation data provided)
            # TODO: may have to remove this, check the vibes
            if X_val is not None and y_val is not None:
                self.model.eval()
                # move validation through the network in batches to avoid OOM
                X_val_tensor = torch.FloatTensor(X_val).view(
                    -1, channels, height, width
                )
                y_val_tensor = torch.LongTensor(y_val)
                val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
                val_loader = DataLoader(
                    val_dataset, batch_size=batch_size, shuffle=False
                )

                val_losses = []
                all_val_preds = []
                all_val_targets = []

                with torch.no_grad():
                    for X_val_batch, y_val_batch in val_loader:
                        X_val_batch = X_val_batch.to(self.device)
                        y_val_batch = y_val_batch.to(self.device)

                        val_outputs = self.model(X_val_batch)
                        batch_loss = self.loss_fn(val_outputs, y_val_batch).item()
                        val_losses.append(batch_loss)

                        _, val_preds_batch = torch.max(val_outputs, 1)
                        all_val_preds.append(val_preds_batch.cpu().numpy())
                        all_val_targets.append(y_val_batch.cpu().numpy())

                val_loss = float(np.mean(val_losses))
                val_preds_np = np.concatenate(all_val_preds)
                val_targets_np = np.concatenate(all_val_targets)

                val_accuracy = accuracy_score(val_targets_np, val_preds_np)
                val_f1 = f1_score(val_targets_np, val_preds_np, average="weighted")

                self.history["val_loss"].append(val_loss)
                self.history["val_accuracy"].append(val_accuracy)
                self.history["val_f1"].append(val_f1)

                if verbose and (epoch + 1) % 10 == 0:
                    print(
                        f"epoch {epoch+1}/{self.num_epochs} | "
                        f"train loss: {avg_train_loss:.4f} | "
                        f"val loss: {val_loss:.4f} | "
                        f"val acc: {val_accuracy:.4f} | "
                        f"val F1: {val_f1:.4f}"
                    )
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(
                        f"epoch {epoch+1}/{self.num_epochs}, train Loss: {avg_train_loss:.4f}"
                    )

        return self.history

    def predict(self, X, batch_size=128):
        if self.model is None:
            raise ValueError("fit must be called before calling predict")

        channels, height, width = self.input_shape
        X_tensor = torch.FloatTensor(X).view(-1, channels, height, width)
        
        # batched prediction to avoid OOM on large datasets/models
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for (X_batch,) in dataloader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                _, predicted = torch.max(outputs, 1)
                all_preds.append(predicted.cpu().numpy())
        
        return np.concatenate(all_preds)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "input_shape": self.input_shape,
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self
