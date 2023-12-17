from tqdm import tqdm
from glob import glob
from pathlib import Path
from warnings import warn

import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import base64
import zlib
from pycocotools import _mask as coco_mask
from PIL import Image

import torch
from torch.utils.data import TensorDataset, DataLoader

from hpa_dino import HPA_DINO
from models import PseudoRegressorLit
from utils import get_images_percentiles, silent
silent = True
overwrite = False

imsize = 512
channels = ["blue", "red", "yellow", "green"]
resolution = 6 # bit
buckets = 2 ** resolution
eval_percentiles = np.array(list(range(0, buckets, 1))) / (buckets - 1) * 100
batch_size = 8
dino = HPA_DINO(imsize, batch_size)
checkpoint_hash = "8ud1u6y4"

dataset_folder = Path("/data/ishang/hpa_u2os/")
label_file = Path("/home/ishang/cellmaps_toolkit/1.sc_embeddings/u2os_labels.csv")
df = pd.read_csv(label_file)
df["well_id"] = df["ID"].apply(lambda x: "_".join(x.split("_")[:2]))

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

def adjust_dimension(min_val, max_val, max_size):
    crop_size = max_val - min_val
    if crop_size > max_size:
        excess = crop_size - max_size
        min_val += excess // 2
        max_val -= excess // 2
    return min_val, max_val

def getCroppedImage(mask, image, imsize):
    bbox = np.argwhere(mask > 0)
    ymin, xmin = bbox.min(axis=0)
    ymax, xmax = bbox.max(axis=0)

    ymin, ymax = adjust_dimension(ymin, ymax, imsize)
    xmin, xmax = adjust_dimension(xmin, xmax, imsize)

    crop = image[:, ymin:ymax, xmin:xmax]
    desired_shape = (len(crop), imsize, imsize)
    padding = [(0, 0)]  # no padding for the channels dimension
    for current, desired in zip(crop.shape[1:], desired_shape[1:]):
        total_padding = desired - current
        padding_one_side = total_padding // 2
        padding.append((padding_one_side, total_padding - padding_one_side) if current < desired else (0, 0))
    crop_padded = (np.pad(crop, padding, mode='constant', constant_values=0) / crop.max().astype(np.float32)).astype(np.float32)
    return crop_padded[:, :imsize, :imsize]

def getPseudotimeMode(checkpoint_hash):
    log_dirs_home = Path("/data/ishang/pseudotime_pred/")
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

    return PseudoRegressorLit.load_from_checkpoint(checkpoint)

cell_indices, cls, percentiles = [], [], []
# iterate at the well-level to calculate percentiles for the input to the ps-time model
well_ids = df["well_id"].unique()
for well_id in tqdm(well_ids, total=len(well_ids)):
    well_folder = dataset_folder / well_id
    cls_cache_file = well_folder / 'cls_tensor.pt'
    percentiles_cache_file = well_folder / 'percentiles_tensor.pt'
    cell_indices_cache_file = well_folder / 'cell_indices.pkl'

    if cls_cache_file.exists() and percentiles_cache_file.exists() and cell_indices_cache_file.exists() and not overwrite:
        well_cls = torch.load(cls_cache_file)
        well_percentiles = torch.load(percentiles_cache_file)
        with open(cell_indices_cache_file, 'rb') as f:
            well_cell_indices = pkl.load(f)
        print("loading from cache")
        # print(well_cls.shape)
        # print(well_percentiles.shape)
        cls.append(well_cls)
        percentiles.append(well_percentiles)
        cell_indices.extend(well_cell_indices)
        continue

    cell_images = []
    well_samples = df[df["well_id"] == well_id]
    # need to get the cell-level images first
    # and to run the DINO model
    for image_id in well_samples["ID"].unique():
        # load four-channel images
        images_paths = [dataset_folder / well_id / (image_id + f"_{c}.png") for c in channels]
        images = [np.array(Image.open(p)) for p in images_paths]
        image = np.stack(images)

        # decode the masks and get cropped cell images
        image_samples = well_samples[well_samples["ID"] == image_id]
        rle_masks = image_samples["cellmask"]
        masks = decodeToBinaryMask(rle_masks, image.shape[1], image.shape[2])
        cell_images.extend([getCroppedImage(mask, image, imsize) for mask in masks])
        cell_indices.extend(image_samples.index.values.tolist())

    # get percentiles and DINO cls to get input embedding
    dataset = torch.Tensor(np.stack(cell_images))
    print(dataset, dataset.max())
    print(image.max())
    print(masks.max())
    sum_mask = np.zeros_like(masks[0])
    for i in range(len(masks)):
        sum_mask += masks[i]
    print((image * sum_mask).max())
    # dataset = dataset / dataset.max()
    values, percentiles_used = get_images_percentiles(dataset.numpy(), percentiles=eval_percentiles, non_zero=True)
    # print("VAL", values.shape)
    well_percentiles = torch.tensor(values)[:2].flatten()
    well_percentiles = torch.tile(well_percentiles, (len(dataset), 1))
    percentiles.append(well_percentiles)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for batch in iter(dataloader):
        cls.append(dino.predict_cls_ref_concat(batch[:, :2]).detach().cpu())
    
    well_cls = torch.cat(cls[-1 * len(dataloader):], dim=0)
    well_folder = dataset_folder / well_id
    # print(well_cls.shape, well_percentiles.shape)
    torch.save(well_cls, cls_cache_file)
    torch.save(well_percentiles, percentiles_cache_file)
    with open(cell_indices_cache_file, 'wb') as f:
        pkl.dump(cell_indices[-1 * len(dataset):], f)

cls = torch.cat(cls, dim=0)
percentiles = torch.Tensor(np.concatenate(percentiles))
cell_indices = np.array(cell_indices)
print("CLS", cls.shape)
print("PER", percentiles.shape)
print("IDX", len(cell_indices))
torch.save(cls, dataset_folder / 'cls_tensor.pt')
torch.save(percentiles, dataset_folder / 'percentiles_tensor.pt')
with open(dataset_folder / 'cell_indices.pkl', 'wb') as f:
    pkl.dump(cell_indices, f)

batch_size = 64
cls_to_pseudotime = getPseudotimeMode(checkpoint_hash)
cls = torch.cat([cls, percentiles], dim=1).to(cls_to_pseudotime.device)
pseudotime_dataloader = DataLoader(cls, batch_size=batch_size, shuffle=False)

pseudotimes = []
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')
with torch.no_grad():
    for batch in tqdm(pseudotime_dataloader, total=len(pseudotime_dataloader), desc="Calculating pseudotimes"):
        pseudotimes.append(cls_to_pseudotime(batch).detach().cpu())
pseudotimes = torch.cat(pseudotimes, dim=0)
pseudotimes = pseudotimes.remainder(2 * torch.pi) / (2 * torch.pi)

df.loc[cell_indices, 'pseudotime'] = pseudotimes.numpy()
df.to_csv(label_file, index=False)
print(df)

print(torch.sum(pseudotimes < 0) + torch.sum(pseudotimes > 1))

plt.hist(pseudotimes.numpy(), bins=50, range=(0, 1))
plt.title('Histogram of Pseudotimes')
plt.xlabel('Pseudotime')
plt.ylabel('Frequency')
plt.savefig('pseudotimes_histogram.png')
