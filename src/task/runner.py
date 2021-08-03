import torch
import torch.nn as nn
import numpy as np 
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from ..metric import get_aucs

class Runner(LightningModule):
    def __init__(self, model, lr, weight_decay, T_0, prediction_type, eval_type, test_case):
        super().__init__()
        self.model = model
        self.test_case = test_case
        self.lr = lr
        self.T_0 = T_0
        self.weight_decay = weight_decay
        if prediction_type == "multilabel":
            self.criterion = nn.BCELoss()
        elif prediction_type == "multiclass":
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def batch_inference(self, x):
        x = x.squeeze(0)
        return self.model(x)

    def configure_optimizers(self):
        opt = Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay= self.weight_decay
        )
        scheduler = CosineAnnealingWarmRestarts(
            optimizer=opt,
            T_0= self.T_0
        )
        lr_scheduler = {
            'scheduler': scheduler, 
            'interval': 'epoch',
            'frequency': 1, 
            'reduce_on_plateau': False, 
            'monitor': 'val_loss' 
        }
        return [opt], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        audio, label = batch
        prediction = self.forward(audio)
        loss = self.criterion(prediction, label)
        self.log_dict(
            {
                "train_loss": loss,
            },
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        return loss
    
    def validation_step(self, batch, batch_idx):
        audio, label = batch
        prediction = self.forward(audio)
        loss = self.criterion(prediction, label)
        return {"val_loss": loss}

    def validation_step_end(self, batch_parts):
        # sigle gpu case
        return batch_parts

    def validation_epoch_end(self, outputs):
        val_loss = torch.mean(torch.stack([output["val_loss"] for output in outputs]))
        self.log_dict(
            {
                "val_loss": val_loss,
            },
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return {"val_loss": val_loss,}

    def test_step(self, batch, batch_idx):
        audio, label = batch
        prediction = self.batch_inference(audio)
        return {"prediction": prediction, "label":label}

    def test_step_end(self, batch_parts):
        # sigle gpu case
        return batch_parts

    def test_epoch_end(self, outputs):
        predictions = torch.stack([output["prediction"] for output in outputs])
        labels = torch.stack([output["label"] for output in outputs])
        roc_aucs, pr_aucs = get_aucs(labels.squeeze(1).detach().cpu().numpy(), predictions.squeeze(1).detach().cpu().numpy())
        result = {
            "roc_aucs":roc_aucs, 
            "pr_aucs":pr_aucs
        }
        self.test_results = result  