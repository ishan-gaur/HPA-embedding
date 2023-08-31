import math
import torch
import torch.nn as nn
import numpy as np
from torch import optim
import lightning.pytorch as lightning
import wandb
from typing import Any, Tuple
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

class Classifier(nn.Module):
    def __init__(self,
        d_input: int = 1024,
        d_hidden: int = 256,
        n_hidden: int = 0,
        d_output: int = 3,
        dropout: bool = False,
    ):
        super().__init__()
        self.model = nn.ModuleList()
        self.model.append(nn.BatchNorm1d(d_input))
        if dropout:
            self.model.append(nn.Dropout(0.5))
        self.model.append(nn.Linear(d_input, d_hidden))
        self.model.append(nn.GELU())
        for _ in range(n_hidden):
            if dropout:
                self.model.append(nn.Dropout(0.5))
            self.model.append(nn.BatchNorm1d(d_hidden))
            self.model.append(nn.Linear(d_hidden, d_hidden))
            self.model.append(nn.GELU())
        if dropout:
            self.model.append(nn.Dropout(0.2))
        self.model.append(nn.BatchNorm1d(d_hidden))
        self.model.append(nn.Linear(d_hidden, d_output))
        self.model.append(nn.Softmax(dim=-1))
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

    def loss(self, y_pred, y):
        return nn.CrossEntropyLoss()(y_pred, y)
    

class ConvClassifier(nn.Module):
    def __init__(self,
        imsize: int = 256,
        nc: int = 2,
        nf: int = 4,
        d_hidden: int = 256,
        n_hidden: int = 0,
        d_output: int = 3,
        dropout: bool = False,
    ):
        super().__init__()
        self.model = nn.ModuleList()
        self.num_down = int(math.log2(imsize))
        if dropout:
            self.model.append(nn.Dropout(0.8))
        self.model.append(nn.Conv2d(nc, nf, 4, 2, 1))
        for _ in range(self.num_down - 1):
            self.model.append(nn.GELU())
            self.model.append(nn.BatchNorm2d(nf))
            if dropout:
                self.model.append(nn.Dropout(0.8))
            self.model.append(nn.Conv2d(nf, nf*2, 4, 2, 1))
            nf *= 2
        self.model = nn.Sequential(*self.model)

        input_size = nf * (imsize // 2**self.num_down)**2 
        self.fully_connected = nn.ModuleList()

        self.fully_connected.append(nn.BatchNorm1d(input_size))
        if dropout:
            self.fully_connected.append(nn.Dropout(0.5))
        self.fully_connected.append(nn.Linear(input_size, d_hidden))
        self.fully_connected.append(nn.GELU())

        for _ in range(n_hidden):
            self.fully_connected.append(nn.BatchNorm1d(d_hidden))
            if dropout:
                self.fully_connected.append(nn.Dropout(0.5))
            self.fully_connected.append(nn.Linear(d_hidden, d_hidden))
            self.fully_connected.append(nn.GELU())

        self.fully_connected.append(nn.BatchNorm1d(d_hidden))
        if dropout:
            self.fully_connected.append(nn.Dropout(0.5))
        self.fully_connected.append(nn.Linear(d_hidden, d_output))

        self.fully_connected = nn.Sequential(*self.fully_connected)

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1)
        return self.fully_connected(x)

    def loss(self, y_pred, y):
        return nn.CrossEntropyLoss()(y_pred, y)
        

class ClassifierLit(lightning.LightningModule):
    def __init__(self,
        conv: bool = False,
        d_input: int = 1024,
        d_hidden = None,
        n_hidden: int = 0,
        imsize: int = 256,
        nc: int = 2,
        nf: int = 64,
        d_output: int = 3,
        lr: float = 5e-5,
        soft: bool = False,
        dropout: bool = False,
    ):
        super().__init__()
        if d_hidden is None:
            d_hidden = d_input
        self.save_hyperparameters()
        if conv:
            self.model = ConvClassifier(imsize=imsize, nc=nc, nf=nf, d_hidden=d_hidden, n_hidden=n_hidden, d_output=d_output, dropout=dropout)
        else:
            self.model = Classifier(d_input=d_input, d_hidden=d_hidden, n_hidden=n_hidden, d_output=d_output, dropout=dropout)
        self.model = torch.compile(self.model)
        self.lr = lr
        self.train_preds, self.val_preds, self.test_preds = [], [], []
        self.train_labels, self.val_labels, self.test_labels = [], [], []
        self.soft = soft
        self.num_classes = d_output

    def forward(self, x):
        return self.model(x)

    def __shared_step(self, batch, batch_idx, stage):
        x, y = batch
        y_pred = self(x)
        # y = torch.argmax(y, dim=-1) if not self.soft else nn.Softmax(dim=-1)(y)
        y = torch.argmax(y, dim=-1) if not self.soft else (y / torch.sum(y, dim=-1, keepdim=True))
        loss = self.model.loss(y_pred, y)
        preds = torch.argmax(y_pred, dim=-1)
        labels = torch.argmax(y, dim=-1) if self.soft else y
        preds, labels = preds.cpu().numpy(), labels.cpu().numpy()
        self.log(f"{stage}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss, preds, labels
    
    def __on_shared_epoch_end(self, preds, labels, stage):
        plt.clf()
        if self.num_classes == 3:
            classes = ["G1", "S", "G2"]
        elif self.num_classes == 6:
            classes = ["Stop-G1", "G1", "G1-S", "S-G2", "G2", "G2-M"]
        elif self.num_classes == 4:
            classes = ["M/G1", "G1", "S", "G2/M"]
        preds, labels = np.concatenate(preds), np.concatenate(labels)
        cm = confusion_matrix(labels, preds)
        # ax = sns.heatmap(cm.astype(np.int32), annot=True, fmt="d", vmin=0, vmax=len(labels))
        ax = sns.heatmap(cm.astype(np.int32), annot=True, fmt="d", vmin=0, vmax=len(labels) / 3)
        ax.set_xlabel("Predicted")
        ax.xaxis.set_ticklabels(classes)
        ax.set_ylabel("True")
        ax.yaxis.set_ticklabels(classes)
        fig = ax.get_figure()
        self.logger.experiment.log({
            f"{stage}/cm": wandb.Image(fig),
        })

        for i, class_name in enumerate(classes):
            self.log(f"{stage}/accuracy_{class_name}", cm[i, i] / np.sum(cm[i]), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

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


class Regressor(nn.Module):
    """
    Simple feed-forward regression module
    """
    def __init__(self,
        d_input: int = 1024,
        d_hidden: int = 256,
        n_hidden: int = 0,
        d_output: int = 2,
        dropout: bool = False,
        batchnorm: bool = False,
    ):
        super().__init__()
        self.model = nn.ModuleList()
        self.build_model(d_input, d_hidden, n_hidden, d_output, dropout, batchnorm)
        self.model = nn.Sequential(*self.model)

    def build_model(self, d_input, d_hidden, n_hidden, d_output, dropout, batchnorm):
        # input layer
        # if dropout:
        #     self.model.append(nn.Dropout(0.5))
        self.model.append(nn.Linear(d_input, d_hidden))
        self.model.append(nn.GELU())

        # hidden layers
        for _ in range(n_hidden):
            if batchnorm:
                self.model.append(nn.BatchNorm1d(d_hidden))
            if dropout:
                self.model.append(nn.Dropout(0.5))
            self.model.append(nn.Linear(d_hidden, d_hidden))
            self.model.append(nn.GELU())

        # output layer
        if batchnorm:
            self.model.append(nn.BatchNorm1d(d_hidden))
        if dropout:
            self.model.append(nn.Dropout(0.2))
        self.model.append(nn.Linear(d_hidden, d_output))
        self.model.append(nn.GELU())

    def forward(self, x):
        return -1 * self.model(x)

    def loss(self, y_pred, y):
        return nn.MSELoss()(y_pred, y)

    
class RegressorLit(lightning.LightningModule):
    """
    Lightning module for training a regression model
    Supports logging for:
    - MSE loss
    - KDEplot of predicted intensities
    - Heatmap of residuals across the output space
    TODO:
    - Intensity residuals over the DINO embedding NMF components
    """
    def __init__(self,
        # conv: bool = False,
        d_input: int = 1024,
        d_hidden = None,
        n_hidden: int = 0,
        # imsize: int = 256,
        # nc: int = 2,
        # nf: int = 64,
        d_output: int = 2,
        lr: float = 5e-5,
        # soft: bool = False,
        dropout: bool = False,
        batchnorm: bool = False,
    ):
        super().__init__()
        if d_hidden is None:
            d_hidden = d_input
        self.save_hyperparameters()
        # if conv:
        #     self.model = ConvClassifier(imsize=imsize, nc=nc, nf=nf, d_hidden=d_hidden, n_hidden=n_hidden, d_output=d_output, dropout=dropout)
        # else:
        self.model = Regressor(d_input=d_input, d_hidden=d_hidden, n_hidden=n_hidden, d_output=d_output, dropout=dropout, batchnorm=batchnorm)
        self.model = torch.compile(self.model)
        self.lr = lr
        self.train_preds, self.val_preds, self.test_preds = [], [], []
        self.train_labels, self.val_labels, self.test_labels = [], [], []
        self.num_classes = d_output

    def forward(self, x):
        return self.model(x)

    def __shared_step(self, batch, batch_idx, stage):
        x, y = batch
        y_pred = self(x)
        loss = self.model.loss(y_pred, y)
        preds, labels = y_pred.detach().cpu(), y.detach().cpu()
        self.log(f"{stage}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss, preds, labels

    def _log_image(self, stage, name, ax):
        fig = ax.get_figure()
        self.logger.experiment.log({
            f"{stage}/{name}": wandb.Image(fig),
        })
    
    def __on_shared_epoch_end(self, preds, labels, stage):
        preds, labels = torch.cat(preds), torch.cat(labels)
    
        # plot the intensity kdeplot
        plt.clf()
        ax = sns.kdeplot(x=preds[:, 0], y=preds[:, 1])
        self._log_image(stage, "intensity_kde", ax)

        # these residuals are 2D, residual for CDT1 and for GMNN
        # we want to make a 2D grid in the label intensity space (also a 2D grid)
        # and color the grid squares by the mix of CDT1 and GMNN intensities defined by the 
        # average residual for all points in that grid square
        # this is a "heatmap" of the residuals across the output space

        preds_color = torch.pow(torch.ones_like(preds) * 10, preds)
        # print(preds_color.min(), preds_color.max())


        # the actual labels have values that are pretty low, because they're averaged so
        # we're going to rescale labels accordingly
        # by the preds min/max values and then clip before remapping to 0-255
        labels_color = torch.pow(torch.ones_like(labels) * 10, labels)
        # print(labels_color.min(), labels_color.max())
        preds_color = (preds_color - labels_color.min()) / (labels_color.max() - labels_color.min())
        # print(preds_color.min(), preds_color.max())

        preds_color = preds_color.transpose(0, 1)
        preds_color = torch.stack([preds_color[0], preds_color[1], torch.zeros_like(preds_color[0])])
        preds_color = preds_color.transpose(0, 1)
        # print(preds_color.min(), preds_color.max())

        # get the grid
        grid_size = 10
        grid_x = torch.linspace(labels[:, 0].max(), labels[:, 0].min(), grid_size)
        grid_y = torch.linspace(labels[:, 1].min(), labels[:, 1].max(), grid_size)
        grid_x, grid_y = torch.meshgrid(grid_x, grid_y)
        grid = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)

        # get the average residual for each grid square
        grid_residuals = torch.zeros([grid.shape[0], 3])
        for i, point in enumerate(grid):
            # get the points in the grid square
            x_min, x_max = point[0] - 0.5, point[0] + 0.5
            y_min, y_max = point[1] - 0.5, point[1] + 0.5
            mask = (labels[:, 0] >= x_min) & (labels[:, 0] < x_max) & (labels[:, 1] >= y_min) & (labels[:, 1] < y_max)
            if mask.sum() == 0:
                grid_residuals[i] = torch.ones_like(grid_residuals[i])
            else:
                grid_residuals[i] = torch.mean(preds_color[mask], dim=0)
        grid_residuals = grid_residuals.reshape(grid_size, grid_size, 3)
        grid_residuals = torch.clamp(grid_residuals, 0, 1) * 255
        # print(grid_residuals.min(), grid_residuals.max())
        # print(grid_residuals.shape)

        # plot the heatmap
        plt.clf()
        ax = plt.gca()
        ax.imshow(grid_residuals.int())
        self._log_image(stage, "prediction_map", ax)

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