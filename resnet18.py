import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchmetrics import Accuracy
import pytorch_lightning as pl


class resnet18(pl.LightningModule):
    def __init__(self, K, ft=True):
        super().__init__()
        self.save_hyperparameters()
        self.model = models.resnet18(pretrained=True)
        if ft:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.fc = nn.Linear(self.model.fc.in_features, K)

        self.train_acc = Accuracy(task='multiclass', num_classes=K)
        self.val_acc = Accuracy(task='multiclass', num_classes=K)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        X, y = batch
        preds = self.model(X)
        loss = F.cross_entropy(preds, y)
        self.train_acc(preds.softmax(dim=-1), y)
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=False,
                 on_epoch=True, batch_size=X.size(0))
        self.log('train_loss', loss.item(), prog_bar=False, on_step=True,
                 on_epoch=False, batch_size=X.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        preds = self.model(X)
        loss = F.cross_entropy(preds, y)
        self.val_acc(preds.softmax(dim=-1), y)
        self.log('val_acc', self.val_acc, prog_bar=True, on_step=False,
                 on_epoch=True, batch_size=X.size(0))
        self.log('val_loss', loss.item(), prog_bar=True, on_step=False,
                 on_epoch=True, batch_size=X.size(0))

    def predict_step(self, batch, batch_idx):
        X, y = batch
        preds = self.model(X)
        return preds.softmax(dim=-1)

    # TODO: allow args for this.
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', patience=3, cooldown=2, verbose=True)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "monitor": "val_acc",
                "interval": "epoch",
                "frequency": 1
            }
        }