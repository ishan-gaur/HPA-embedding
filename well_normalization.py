from tqdm import tqdm
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pipeline import load_index_paths
from utils import get_images_percentiles

from sklearn.decomposition import PCA
# from sklearn.decomposition import SparsePCA as PCA
from sklearn.preprocessing import StandardScaler
from data_viz import save_image_grid
import torch


import utils
utils.silent = True

data_folder = Path("/data/ishang/FUCCI-dataset-well/")
index_file = "index_clean_no_border_rm_1000_sharp_1250.csv"
index_path = data_folder / index_file

image_paths, mask_paths, _ = load_index_paths(index_path)

eval_percentiles = [0, 1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95, 99, 100]
channel_names = ["DAPI", "TUBL", "GMNN", "CDT1"]

well_percentiles = np.load(data_folder / "well_percentiles.npy")
normalized_well_percentiles = np.load(data_folder / "normalized_well_percentiles.npy")

def well_normalization_map(original_perc, transformed_perc):
    def normalize(well_images):
        # well images are B x C x H x W
        for c in range(original_perc.shape[0]):
            # bucket the image pixels into the percentile ranges
            perc_domain = np.concatenate([[0], original_perc[c]])
            perc_range = np.concatenate([[0], transformed_perc[c]])
            well_images_perc = np.digitize(well_images[:, c], perc_domain, right=False)
            assert (well_images_perc > 1).any(), "No pixels in the image are nonzero"

            # then normalize the image by the percentile range using a linear extrapolation within the bucket
            def normalize_pixel(x, v):
                if x != 1:
                    pass
                perc_below = perc_domain[x - 1]
                perc_above = perc_domain[x] if x < len(perc_domain) else perc_domain[-1]
                transformed_perc_below = perc_range[x - 1]
                transformed_perc_above = perc_range[x] if x < len(perc_range) else perc_range[-1]
                if perc_above == perc_below:
                    return np.random.uniform(transformed_perc_below, transformed_perc_above)
                bucket_pos = (v - perc_below) / (perc_above - perc_below)
                transformed_value = transformed_perc_below + bucket_pos * (transformed_perc_above - transformed_perc_below)
                return transformed_value
            
            normalize_pixel = np.vectorize(normalize_pixel)

            well_images[:, c] = normalize_pixel(well_images_perc.flatten(), well_images[:, c].flatten()).reshape(well_images.shape[0], well_images.shape[2], well_images.shape[3])

        return well_images
    return normalize


if not Path("test_norm_image.npy").exists() or True:
    for i, (image_path, mask_path) in tqdm(enumerate(zip(image_paths, mask_paths)), total=len(image_paths), desc="Calculating well percentiles"):
        if i != 2:
            continue
        normalization_function = well_normalization_map(well_percentiles[i], normalized_well_percentiles[i])
        images = np.load(image_path)
        masks_path = str(mask_path) + ".npy"
        masks = np.load(masks_path)[:, None, ...]
        images = images * (masks > 0)
        normalized_images = normalization_function(images[:4])
        np.save("test_norm_image", normalized_images)
else:
    normalized_images = np.load("test_norm_image.npy")
    print(image_paths[2])
    images = np.load(image_paths[2])
    masks = np.load(str(mask_paths[2]) + ".npy")[:, None, ...]
    images = images * (masks > 0)


for orig, norm in zip(images[:4], normalized_images):
    # normalize each channel of orig to [0, 1]
    print(norm.min(), norm.max())
    print(orig.min(), orig.max())
    print(np.percentile(norm[norm != 0], [0, 1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95, 99, 100]))
    print(np.percentile(orig[orig != 0], [0, 1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95, 99, 100]))
    # for c in range(orig.shape[0]):
        # orig[c] = (orig[c] - orig[c].min()) / (orig[c].max() - orig[c].min())
    orig = orig / orig.max()
    orig, norm = torch.from_numpy(orig.astype("float32")), torch.from_numpy(norm.astype("float32"))
    grid_images = torch.concatenate([orig, norm], dim=0)
    # grid_images = norm
    print(norm.min(), norm.max())
    print(orig.min(), orig.max())
    print(torch.quantile(norm[norm > 0], torch.tensor([0, 1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95, 99, 100]) / 100))
    print(torch.quantile(orig[orig > 0], torch.tensor([0, 1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95, 99, 100]) / 100))
    break
    
grid_images = grid_images[:, None, ...]
print(grid_images.shape)
save_image_grid(grid_images, "test_image.png", nrow=orig.shape[0], cmaps=None)