import math
import torch
import torch.nn as nn
import numpy as np
from torch import optim
import lightning.pytorch as lightning
import wandb
from typing import Any, Tuple
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class Classifier(nn.Module):
    def __init__(self,
        d_input: int = 1024,
        d_hidden: int = 256,
        n_hidden: int = 0,
        d_output: int = 3,
    ):
        super().__init__()
        self.model = nn.ModuleList()
        self.model.append(nn.Linear(d_input, d_hidden))
        self.model.append(nn.GELU())
        for _ in range(n_hidden):
            self.model.append(nn.Linear(d_hidden, d_hidden))
            self.model.append(nn.GELU())
        self.model.append(nn.Linear(d_hidden, d_output))
        self.model.append(nn.Softmax(dim=-1))
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

    def loss(self, y_pred, y):
        return nn.CrossEntropyLoss()(y_pred, y)

class ClassifierLit(lightning.LightningModule):
    def __init__(self,
        d_input: int = 1024,
        d_hidden = None,
        d_output: int = 3,
        lr: float = 5e-5,
        soft: bool = False,
    ):
        super().__init__()
        if d_hidden is None:
            d_hidden = d_input
        self.save_hyperparameters()
        self.model = Classifier(d_input, d_hidden, d_output)
        self.d_input = d_input
        self.d_output = d_output
        self.lr = lr
        self.train_preds, self.val_preds, self.test_preds = [], [], []
        self.train_labels, self.val_labels, self.test_labels = [], [], []
        self.soft = soft

    def forward(self, x):
        return self.model(x)

    def __shared_step(self, batch, batch_idx, stage):
        x, y = batch
        if not self.soft:
            y = torch.argmax(y, dim=-1)
        y_pred = self(x)
        loss = self.model.loss(y_pred, y)
        preds = torch.argmax(y_pred, dim=-1).cpu().numpy()
        labels = torch.argmax(y, dim=-1).cpu().numpy() if self.soft else y.cpu().numpy()
        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss, preds, labels
    
    def __on_shared_epoch_end(self, preds, labels, stage):
        plt.clf()
        preds, labels = np.concatenate(preds), np.concatenate(labels)
        cm = confusion_matrix(labels, preds)
        ax = sns.heatmap(cm.astype(np.int32), annot=True, fmt="d", vmin=0, vmax=len(labels))
        ax.set_xlabel("Predicted")
        ax.xaxis.set_ticklabels(["G1", "S", "G2"])
        ax.set_ylabel("True")
        ax.yaxis.set_ticklabels(["G1", "S", "G2"])
        fig = ax.get_figure()
        self.logger.experiment.log({
            f"{stage}/cm": wandb.Image(fig),
        })

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self.__shared_step(batch, batch_idx, "train")
        self.train_preds.append(preds)
        self.train_labels.append(labels)
        return loss

    def on_train_epoch_end(self):
        self.__on_shared_epoch_end(self.train_preds, self.train_labels, "train")
        self.train_preds, self.train_labels = [], []

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self.__shared_step(batch, batch_idx, "validate")
        self.val_preds.append(preds)
        self.val_labels.append(labels)
        return loss

    def on_validation_epoch_end(self):
        self.__on_shared_epoch_end(self.val_preds, self.val_labels, "validate")
        self.val_preds, self.val_labels = [], []
    
    def test_step(self, batch, batch_idx):
        loss, preds, labels = self.__shared_step(batch, batch_idx, "test")
        self.test_preds.append(preds)
        self.test_labels.append(labels)
        return loss

    def on_test_epoch_end(self):
        self.__on_shared_epoch_end(self.test_preds, self.test_labels, "test")
        self.test_preds, self.test_labels = [], []

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

class DINO(nn.Module):
    PATCH_SIZE = 14
    CLS_DIM = 1024
    def __init__(self,
        imsize: Tuple[int, int] = (256, 256),
        margin_tolerance: int = -1,
    ):
        super().__init__()
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        self.crop_slices = DINO.__get_crop_slices(imsize, margin_tolerance, DINO.PATCH_SIZE)

    @torch.no_grad()
    def forward(self, x):
        return self.dinov2(x[..., self.crop_slices[-2], self.crop_slices[-1]])

    def __get_crop_slices(imsize, margin_tolerance, dino_patch_size):
        crop_slices = []
        for image_size in imsize:
            closest_multiple = math.floor(image_size / dino_patch_size)
            margin_size = image_size - closest_multiple * dino_patch_size
            print(f"Margin cropped out to fit DINO patches: {margin_size}")
            if margin_tolerance >= 0:
                assert margin_size <= margin_tolerance, f"Error in creating the crop slices for use with the DINO model. Margin size is {margin_size} but margin tolerance is {margin_tolerance}."
            crop_slice = slice(margin_size // 2, image_size - margin_size // 2)
            cropped_image_size = closest_multiple * dino_patch_size
            assert cropped_image_size == image_size - margin_size == crop_slice.stop - crop_slice.start, \
                f"Error in creating the crop slices for use with the DINO model. {cropped_image_size} {image_size} {margin_size} {crop_slice.stop - crop_slice.start}"
            crop_slices.append(crop_slice)
        assert len(crop_slices) == 2, "Error in creating the crop slices for use with the DINO model. Only 2D images are supported. imsize might have more than 2 elements."
        return crop_slices
        
    
class DINOClassifier(lightning.LightningModule):
    def __init__(self,
        imsize: Tuple[int, int] = 256,
        d_output: int = 3,
        margin_tolerance: int = -1,
        lr: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dino = DINO(imsize=imsize, margin_tolerance=margin_tolerance)
        self.classifier = Classifier(d_input=DINO.CLS_DIM, d_output=d_output)
        self.imsize = imsize
        self.lr = lr

    def forward(self, x):
        return self.classifier(self.dino(x))

    def loss(self, y_pred, y):
        return self.classifier.loss(y_pred, y)
    
    def __shared_step(self, batch, stage):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        self.log(f'{stage}/loss', loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.__shared_step(batch, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self.__shared_step(batch, 'validate')
    
    def test_step(self, batch, batch_idx):
        return self.__shared_step(batch, 'test')
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.classifier.parameters(), lr=self.lr)
