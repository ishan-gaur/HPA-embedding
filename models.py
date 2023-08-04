import math
import torch
import torch.nn as nn
import lightning.pytorch as lightning
from typing import Tuple
import pickle
from pathlib import Path
from data import CellImageDataset
from torch.utils.data import DataLoader
from sklearn.mixture import GaussianMixture
import seaborn as sns
import matplotlib.pyplot as plt

class Classifier(nn.Module):
    def __init__(self,
        d_input: int = 1024,
        d_hidden: int = 256,
        d_output: int = 3,
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_output),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.model(x)

    def loss(self, y_pred, y):
        return nn.CrossEntropyLoss()(y_pred, y)

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


class CellCycleGMM:
    def __init__(self, index_file, n_components=3):
        self.index_file = index_file
        self.n_components = n_components
        if self.index_file.exists() and CellCycleGMM.__gmm_pickle_path(self.index_file).exists():
            print("Loading GMM from pickle file at " + str(CellCycleGMM.__gmm_pickle_path(self.index_file)))
            self.gmm = pickle.load(open(CellCycleGMM.__gmm_pickle_path(self.index_file), "rb"))
        else:
            print("Fitting GMM to data since no pickle file was found at " + str(CellCycleGMM.__gmm_pickle_path(self.index_file)))
            dataset = CellImageDataset(index_file)
            self.gmm = GaussianMixture(n_components=self.n_components)
            self.gmm.fit(dataset[:])
            pickle.dump(self.gmm, open(CellCycleGMM.__gmm_pickle_path(self.index_file), "wb"))
            print("Saved GMM to pickle file at " + str(CellCycleGMM.__gmm_pickle_path(self.index_file)))

    def __gmm_pickle_path(index_file):
        name = "gmm_" + "_".join(index_file.stem.split("_")[1:]) + ".pkl"
        return index_file.parent / name

    def predict(self, x):
        return self.gmm.predict(x)

    def predict_proba(self, x):
        return self.gmm.predict_proba(x)
    
    def plot_predictions(self, x):
        sns.kdeplot(x[:, 0], x[:, 1], hue=self.predict(x), color_palette="Set2")

    def plot_probabilities(self, x):
        plt.scatter(x[:, 0], x[:, 1], c=self.predict_proba(x))