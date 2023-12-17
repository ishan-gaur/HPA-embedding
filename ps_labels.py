from tqdm import tqdm
from glob import glob
from pathlib import Path
from warnings import warn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import base64
import zlib
from pycocotools import _mask as coco_mask
from PIL import Image

import torch
from torch.utils.data import TensorDataset

from hpa_dino import HPA_DINO
from utils import get_images_percentiles
from models import PseudoRegressorLit

label_file = Path("/home/ishang/cellmaps_toolkit/1.sc_embeddings/u2os_labels.csv")
dataset_folder = Path("/data/ishang/hpa_u2os/")
df = pd.read_csv(label_file)

def decodeToBinaryMask(rleStrings, imWidth, imHeight):
    detlist = []
    for rleCodedStr in rleStrings:
        uncodedStr = base64.b64decode(rleCodedStr)
        uncompressedStr = zlib.decompress(uncodedStr, wbits=zlib.MAX_WBITS)
        detection = {"size": [imWidth, imHeight], "counts": uncompressedStr}
        detlist.append(detection)
    masks = []
    for det in detlist:
        masks.append(coco_mask.decode([det])[:, :, 0])
    masks = np.stack(masks)
    # print(masks.shape)
    # masks = coco_mask.decode(detlist)
    binaryMasks = masks.astype("uint8")
    return binaryMasks

df["well_id"] = df["ID"].apply(lambda x: "_".join(x.split("_")[:2]))
well_ids = df["well_id"].unique()
sizes = []
imsize = df.iloc[0]["ImageWidth"]
for well_id in tqdm(well_ids, total=len(well_ids)):
    well_samples = df[df["well_id"] == well_id]
    masks = well_samples["cellmask"]
    masks = decodeToBinaryMask(masks, imsize, imsize)
    mask_sum = np.sum(masks, axis=0)
    print(len(masks))
    print(mask_sum.shape)
    print(len(np.argwhere(mask_sum == 0)))
    print(len(np.argwhere(mask_sum == 1)))
    print(len(np.argwhere(mask_sum == 2)))
    print(mask_sum.max())
    plt.imshow(mask_sum, cmap='gray')
    plt.axis('off')
    plt.savefig(f"{well_id}_mask_sum.png", bbox_inches='tight', pad_inches=0)
    plt.close()
    # Create a figure to display the masks in a grid
    num_masks = len(masks)
    grid_size = int(np.ceil(np.sqrt(num_masks)))
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(20, 20))
    axs = axs.flatten()
    for idx, mask in enumerate(masks):
        axs[idx].imshow(mask, cmap='gray')
        axs[idx].axis('off')
    # Hide any remaining subplots that don't have an image
    for ax in axs[num_masks:]:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"{well_id}_masks_grid.png", bbox_inches='tight', pad_inches=0)
    plt.close()
    break