import torch
import torch.nn as nn

class LinearCls(nn.Module):
    def __init__(self, backborn:nn.Module, feature_dim: int, n_calss: int, prediction_type: str, eval_type: str):
        super().__init__()
        self.projection = nn.Linear(feature_dim, n_calss)
        self.eval_type = eval_type
        self.backborn = backborn
        if prediction_type == "multilabel":
            self.activation = nn.Sigmoid()
        elif prediction_type == "multiclass":
            self.activation = nn.Softmax()
        else:
            raise ValueError(f"Unknown prediction_type {prediction_type}")
        
    def forward(self, x: torch.Tensor):
        if self.eval_type == "linear_prob":
            with torch.no_grad():
                embedding = self.backborn.get_embedding(x)
                embedding = embedding.detach()

        elif self.eval_type == "finetunning":
            embedding = self.backborn.get_embedding(x)

        x = self.projection(embedding)
        x = self.activation(x)
        return x