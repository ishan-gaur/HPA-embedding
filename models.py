import math
import torch
import torch.nn as nn
import numpy as np
from torch import optim
import lightning.pytorch as lightning
from lightning.pytorch.utilities import rank_zero_only
import wandb
from typing import Any, Tuple
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
from pathlib import Path

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

    def _shared_step(self, batch, batch_idx, stage):
        x, y = batch
        y_pred = self(x)
        loss = self.model.loss(y_pred, y)
        preds, labels = y_pred.detach().cpu(), y.detach().cpu()
        self.log(f"{stage}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss, preds, labels

    def _log_image(self, stage, name, ax):
        if type(ax) == sns.JointGrid:
            fig = ax.figure
        else:
            fig = ax.get_figure()
        self.logger.experiment.log({
            f"{stage}/{name}": wandb.Image(fig),
        })
    
    @rank_zero_only
    def _on_shared_epoch_end(self, preds, labels, stage):
        preds, labels = torch.cat(preds), torch.cat(labels)
    
        # plot the intensity kdeplot
        plt.clf()
        plt.title("Predicted Intensity Distribution")
        plt.xlabel("CDT1 Intensity (log10)")
        plt.ylabel("GMNN Intensity (log10)")
        ax = sns.jointplot(x=preds[:, 0], y=preds[:, 1], kind="kde")
        plt.tight_layout()
        self._log_image(stage, "intensity_kde", ax)
        plt.close()

        # plot the residuals per channel
        plt.clf()
        plt.title("GMNN Residuals")
        plt.ylabel("Pred - Label (log10)")
        plt.xlabel("GMNN Label Intensity (log10)")
        ax = sns.jointplot(x=labels[:, 0], y=(preds[:, 0] - labels[:, 0]), kind="kde")
        self._log_image(stage, "residuals_gmnn", ax)
        plt.close()
        plt.clf()
        plt.title("CDT1 Residuals")
        plt.ylabel("Pred - Label (log10)")
        plt.xlabel("CDT1 Label Intensity (log10)")
        ax = sns.jointplot(x=labels[:, 1], y=(preds[:, 1] - labels[:, 1]), kind="kde")
        plt.tight_layout()
        self._log_image(stage, "residuals_cdt1", ax)
        plt.close()

        # these residuals are 2D, residual for CDT1 and for GMNN
        # we want to make a 2D grid in the label intensity space (also a 2D grid)
        # and color the grid squares by the mix of CDT1 and GMNN intensities defined by the 
        # average residual for all points in that grid square
        # this is a "heatmap" of the residuals across the output space

        preds_color = torch.pow(torch.ones_like(preds) * 10, preds) # 10 ** preds

        # the actual labels have values that are pretty low, because they're averaged so
        # we're going to rescale labels accordingly (looked at some outputs--Train/Val mix max are (0.02, 0.7), (0.02, 0.9))
        # so actually not going to rescale, just clip
        preds_color = torch.clamp(preds_color, min=0.0, max=1.0)

        # Add B channel to get RGB colors
        preds_color = preds_color.transpose(0, 1) # C x B
        preds_color = torch.stack([preds_color[0], preds_color[1], torch.zeros_like(preds_color[0])])
        preds_color = preds_color.transpose(0, 1)
        # print(preds_color.min(), preds_color.max())

        # get the grid: x is CDT1, y is GMNN; axis 0 is y, axis 1 is x
        grid_size = 10
        vertices_x = torch.linspace(-3, 0, grid_size + 1)
        vertices_y = torch.linspace(-3, 0, grid_size + 1)

        vertices_x, vertices_y = torch.meshgrid(vertices_x, vertices_y)
        vertices = torch.stack([vertices_x, vertices_y], dim=-1).reshape(grid_size + 1, grid_size + 1, 2)

        # get the average residual for each grid square
        grid_residuals = torch.zeros((grid_size, grid_size, 3))
        for i in range(grid_size):
            for j in range(grid_size):
                x_min, y_min = vertices[i, j]
                x_max, y_max = vertices[i + 1, j + 1]
                mask = (labels[:, 0] >= x_min) & (labels[:, 0] < x_max) & (labels[:, 1] >= y_min) & (labels[:, 1] < y_max)
                if mask.sum() == 0:
                    grid_residuals[i, j] = torch.ones_like(grid_residuals[i, j])
                else:
                    grid_residuals[i, j] = torch.mean(preds_color[mask], dim=0)
        grid_residuals = torch.clamp(grid_residuals, 0, 1) * 255
        grid_residuals = grid_residuals.flip(0) # because the y axis is decreasing in the plot

        # plot the heatmap
        plt.clf()
        plt.title("Average Predicted Color binned by Label Intensity")
        plt.xlabel("CDT1 Label Intensity (log10)")
        plt.ylabel("GMNN Label Intensity (log10)")
        ax = plt.gca()
        plt.tight_layout()
        ax.imshow(grid_residuals.int())
        self._log_image(stage, "prediction_map", ax)
        plt.close()

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self._shared_step(batch, batch_idx, "train")
        self.train_preds.append(preds)
        self.train_labels.append(labels)
        return loss

    def on_train_epoch_end(self):
        self._on_shared_epoch_end(self.train_preds, self.train_labels, "train")
        self.train_preds, self.train_labels = [], []

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self._shared_step(batch, batch_idx, "validate")
        self.val_preds.append(preds)
        self.val_labels.append(labels)
        return loss

    def on_validation_epoch_end(self):
        self._on_shared_epoch_end(self.val_preds, self.val_labels, "validate")
        self.val_preds, self.val_labels = [], []
    
    def test_step(self, batch, batch_idx):
        loss, preds, labels = self._shared_step(batch, batch_idx, "test")
        self.test_preds.append(preds)
        self.test_labels.append(labels)
        return loss

    def on_test_epoch_end(self):
        self._on_shared_epoch_end(self.test_preds, self.test_labels, "test")
        self.test_preds, self.test_labels = [], []

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


class PseudoRegressor(nn.Module):
    """
    Simple feed-forward regression module
    """
    def __init__(self,
        d_input: int = 1024,
        d_hidden: int = 256,
        n_hidden: int = 0,
        d_output: int = 1,
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
        # self.model.append(nn.GELU())
        # self.model.append(nn.Sigmoid())

    def forward(self, x):
        return self.model(x)

    def cart_distance(theta_pred, y):
        theta = 2 * torch.pi * y
        xy_pred = torch.stack([torch.cos(theta_pred), torch.sin(theta_pred)], dim=-1)
        xy = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
        return torch.norm(xy_pred - xy, dim=-1)

    def arc_distance(theta_pred, y):
        theta_pred = torch.clone(theta_pred).remainder(2 * torch.pi)
        # print("theta_pred", theta_pred.min(), theta_pred.max())
        theta = 2 * torch.pi * y
        # print("theta", theta.min(), theta.max())
        # assert torch.abs(theta_pred - theta).max() <= 2 * torch.pi, f"{torch.abs(theta_pred - theta).max()}"
        return torch.where(
            torch.abs(theta - theta_pred) > torch.pi,
            (2 * torch.pi - torch.abs(theta - theta_pred)) * torch.sign(theta - theta_pred),
            theta - theta_pred
        ) / (2 * torch.pi)

    def cart_loss(self, theta_pred, y):
        loss = PseudoRegressor.cart_distance(theta_pred, y)
        return loss

    def arc_loss(self, theta_pred, y):
        loss = PseudoRegressor.arc_distance(theta_pred, y)
        return torch.pow(loss, 2)

    def bin_reweight(self, loss, y, nbin):
        bin_ct = torch.histc(y, nbin)
        bin_ct_avg = torch.mean(bin_ct)
        bin_weights = bin_ct_avg / bin_ct
        bin_weights[bin_weights == float("inf")] = bin_weights[bin_weights != float("inf")].max()
        bins = torch.linspace(y.min(), y.max(), nbin + 1).to(y.device)
        binned_y = torch.bucketize(y, bins)
        binned_y[binned_y == 0] = 1
        binned_y -= 1
        loss_weights = torch.index_select(bin_weights, 0, binned_y)
        loss = loss * loss_weights
        return loss



class PseudoRegressorLit(RegressorLit):
    """
    Lightning module for training a regression model
    Supports logging for:
    - MSE loss
    - Histogram of predicted pseudotimes
    - Psuedotime residuals
    """
    def __init__(self,
        d_input: int = 1024,
        d_hidden = 4 * 1024,
        n_hidden: int = 1,
        d_output: int = 1,
        lr: float = 1e-4,
        dropout: bool = False,
        batchnorm: bool = False,
        loss_type: str = "arc",
        reweight_loss: bool = False,
        bins: int = 10,
    ):
        super().__init__()
        if d_hidden is None:
            d_hidden = d_input
        self.save_hyperparameters()
        self.model = PseudoRegressor(d_input=d_input, d_hidden=d_hidden, n_hidden=n_hidden, d_output=d_output, dropout=dropout, 
                                    batchnorm=batchnorm)
        self.model = torch.compile(self.model)
        self.lr = lr
        self.train_preds, self.val_preds, self.test_preds = [], [], []
        self.train_labels, self.val_labels, self.test_labels = [], [], []
        self.num_classes = d_output
        self.loss_type = loss_type
        self.reweight_loss = reweight_loss
        self.bins = bins

    def _shared_step(self, batch, batch_idx, stage):
        x, y = batch
        theta_pred = self(x)
        theta_pred = theta_pred.squeeze()
        y = y.squeeze()
        cart_loss = self.model.cart_loss(theta_pred, y)
        arc_loss = self.model.arc_loss(theta_pred, y)
        preds, labels = theta_pred.detach().cpu(), y.detach().cpu()
        self.log(f"{stage}/loss", torch.mean(arc_loss), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"{stage}/cart_loss", torch.mean(cart_loss), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        loss = cart_loss if self.loss_type == "cart" else arc_loss
        if self.reweight_loss:
            loss = self.model.bin_reweight(loss, y, self.bins)
            self.log(f"{stage}/loss_reweighted", torch.mean(loss), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        loss = torch.mean(loss)
        return loss, preds, labels

    @rank_zero_only
    def _on_shared_epoch_end(self, preds, labels, stage):
        preds, labels = torch.cat(preds), torch.cat(labels)
        preds, labels = preds.flatten(), labels.flatten()
    
        # plot the intensity kdeplot
        plt.clf()
        plt.title("Predicted Pseudotime Distribution")
        plt.xlabel("Pseudotime")
        plt.ylabel("Counts")
        # print("preds", preds.min(), preds.max())
        preds_pseudotime = preds.remainder(2 * torch.pi)
        # print("after mod", preds_pseudotime[preds.argmin()], preds_pseudotime[preds.argmax()])
        # print("new endpoints", preds_pseudotime.min(), preds_pseudotime.max())
        preds_pseudotime = preds_pseudotime / (2 * torch.pi)
        # print("after rescale", preds_pseudotime[preds_pseudotime.argmin()], preds_pseudotime[preds_pseudotime.argmax()])
        # print("after rescale", preds_pseudotime.min(), preds_pseudotime.max())
        # assert preds_pseudotime.min() >= 0 and preds_pseudotime.max() <= 1, f"{preds_pseudotime.min()} {preds_pseudotime.max()}"
        ax = sns.histplot(preds_pseudotime, bins=50)
        plt.tight_layout()
        super()._log_image(stage, "pseudotime_hist", ax)
        plt.close()

        plt.clf()
        ax = sns.histplot(preds, bins=50)
        plt.tight_layout()
        super()._log_image(stage, "preds_raw_hist", ax)
        plt.close()

        # plot the residuals
        plt.clf()
        plt.title("Residuals")
        plt.ylabel("Label - Pred")
        plt.xlabel("Label Pseudotime")
        residuals = PseudoRegressor.arc_distance(preds, labels)
        ax = sns.jointplot(x=labels, y=residuals, kind="hist")
        super()._log_image(stage, "residuals", ax)
        plt.close()

        # create confusion matrix
        bins = torch.linspace(0, 1, self.bins + 1).to(preds.device)
        binned_preds = torch.bucketize(preds_pseudotime, bins)
        binned_preds[binned_preds == 0] = 1
        binned_preds -= 1
        binned_labels = torch.bucketize(labels, bins)
        binned_labels[binned_labels == 0] = 1
        binned_labels -= 1

        # need to add one example of each class prediction pair to the confusion matrix
        preds_lapl = [binned_preds]
        preds_lapl.extend([torch.arange(len(bins) - 1).to(preds.device)] * (len(bins) - 1))
        binned_preds = torch.cat(preds_lapl)
        labels_lapl = [binned_labels]
        labels_lapl.extend([torch.ones(len(bins) - 1).to(preds.device) * i for i in range(len(bins) - 1)])
        binned_labels = torch.cat(labels_lapl)

        cm = confusion_matrix(binned_labels, binned_preds)
        # normalize the rows
        cm = cm / cm.sum(axis=1, keepdims=True)
        # print(cm)
        # ax = sns.heatmap(cm.astype(np.int32), annot=True, fmt="d", vmin=0, vmax=len(labels))
        plt.clf()
        ax = sns.heatmap(cm, vmin=0, vmax=1.0, annot=True, fmt=".2f")
        ax.set_xlabel("Predicted")
        ax.xaxis.set_ticklabels([f"{i:.2f}" for i in bins[1:]])
        ax.set_ylabel("True")
        ax.yaxis.set_ticklabels([f"{i:.2f}" for i in bins[1:]])
        super()._log_image(stage, f"cm_{self.bins}", ax)
        plt.close()

        # create confusion matrix
        eval_bins = 3
        bins = torch.linspace(0, 1, eval_bins + 1).to(preds.device)
        binned_preds = torch.bucketize(preds_pseudotime, bins)
        binned_preds[binned_preds == 0] = 1
        binned_preds -= 1
        binned_labels = torch.bucketize(labels, bins)
        binned_labels[binned_labels == 0] = 1
        binned_labels -= 1

        # need to add one example of each class prediction pair to the confusion matrix
        preds_lapl = [binned_preds]
        preds_lapl.extend([torch.arange(len(bins) - 1).to(preds.device)] * (len(bins) - 1))
        binned_preds = torch.cat(preds_lapl)
        labels_lapl = [binned_labels]
        labels_lapl.extend([torch.ones(len(bins) - 1).to(preds.device) * i for i in range(len(bins) - 1)])
        binned_labels = torch.cat(labels_lapl)

        cm = confusion_matrix(binned_labels, binned_preds)
        # normalize the rows
        cm = cm / cm.sum(axis=1, keepdims=True)
        # print(cm)
        plt.clf()
        # ax = sns.heatmap(cm.astype(np.int32), annot=True, fmt="d", vmin=0, vmax=len(labels))
        ax = sns.heatmap(cm, vmin=0, vmax=1.0, annot=True, fmt=".2f")
        ax.set_xlabel("Predicted")
        ax.xaxis.set_ticklabels([f"{i:.2f}" for i in bins[1:]])
        ax.set_ylabel("True")
        ax.yaxis.set_ticklabels([f"{i:.2f}" for i in bins[1:]])
        super()._log_image(stage, f"cm", ax)
        plt.close()

def load_model(checkpoint_hash, log_dirs_home):
    chkpt_dir_pattern = f"{log_dirs_home}/*/*/*-{checkpoint_hash}/"
    checkpoint_folder = glob(chkpt_dir_pattern)
    if len(checkpoint_folder) > 1:
        raise ValueError(f"Multiple possible checkpoints found: {checkpoint_folder}")
    if len(checkpoint_folder) == 0:
        raise ValueError(f"No checkpoint found for glob pattern: {chkpt_dir_pattern}")
    models_folder = Path(checkpoint_folder[0]).parent.parent / "lightning_logs"
    models_list = list(models_folder.iterdir())
    models_list.sort()
    # the elements should be ###-##.ckpt, 'epoch=###.ckpt', and 'last.ckpt'
    checkpoint = models_list[0]
    if not checkpoint.exists():
        raise ValueError(f"Checkpoint path {checkpoint} does not exist")

    print(f"Loading {checkpoint}")
    cls_to_pseudotime = PseudoRegressorLit.load_from_checkpoint(checkpoint)
    print(cls_to_pseudotime.device)
    return cls_to_pseudotime