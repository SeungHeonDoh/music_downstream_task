import torch
import torch.nn as nn

class LinearProbe(nn.Module):
    def __init__(self, feature_dim: int, n_calss: int, prediction_type: str):
        super().__init__()
        self.projection = torch.nn.Linear(feature_dim, n_calss)
        if prediction_type == "multilabel":
            self.activation = torch.nn.Sigmoid()
        elif prediction_type == "multiclass":
            self.activation = torch.nn.Softmax()
        else:
            raise ValueError(f"Unknown prediction_type {prediction_type}")

    def forward(self, x: torch.Tensor):
        x = self.projection(x)
        x = self.activation(x)
        return x

class FineTunning(nn.Module):
    def __init__(self, backborn: nn.Module, feature_dim: int, n_calss: int, prediction_type: str):
        super().__init__()
        self.backborn = backborn
        self.projection = torch.nn.Linear(feature_dim, n_calss)
        if prediction_type == "multilabel":
            self.activation = torch.nn.Sigmoid()
        elif prediction_type == "multiclass":
            self.activation = torch.nn.Softmax()
        else:
            raise ValueError(f"Unknown prediction_type {prediction_type}")

    def forward(self, x: torch.Tensor):
        x = self.backborn(x)
        x = self.projection(x)
        x = self.activation(x)
        return x
