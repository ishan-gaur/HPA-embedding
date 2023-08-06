from pathlib import Path
import math
import torch
from kornia.geometry.transform import resize
from torchvision.utils import save_image
from data import CellImageDataset, ImageViewer, SimpleDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from tqdm import tqdm
import numpy as np

# DATA_DIR = Path("/home/ishang/HPA-embedding/dev-dataset-CCNB1/")
# COLOR_MAP = ["pure_blue", "pure_red", "pure_green"]
DATA_DIR = Path("/home/ishang/HPA-embedding/dev-dataset-FUCCI/")
# DATA_DIR = Path("/data/ishang/FUCCI-dataset/")
COLOR_MAP = ["pure_blue", "pure_yellow", "pure_red", "pure_green"]
device = "cuda:6"
dino_patch_size = 14
image_size = 768
# image_size = 512
reload = False

dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
dataset = CellImageDataset(DATA_DIR)


if SimpleDataset.has_cache_files(DATA_DIR / "rgb_dataset.pt") and reload:
    rgb_dataset = SimpleDataset(path=(DATA_DIR / "rgb_dataset.pt"))
else:
    dataset.set_channel_colors(COLOR_MAP)
    rgb_dataset = dataset.as_rgb()
    rgb_dataset.save(DATA_DIR / "rgb_dataset.pt")

# rgb_dataset = torch.cat([torch.load(DATA_DIR / f"rgb_dataset-{i * 5000}.pt") for i in range(0, 1)])
# rgb_dataset = SimpleDataset(tensor=rgb_dataset)


dino_embeddings = {
    "x_norm_clstoken": [],
    "x_norm_patchtokens": [],
}

dinov2_vitl14.to(device)
closest_multiple = math.floor(image_size / dino_patch_size)
extra_padding = image_size - closest_multiple * dino_patch_size
crop_slice = slice(extra_padding // 2, image_size - extra_padding // 2)
cropped_image_size = closest_multiple * dino_patch_size
assert cropped_image_size == image_size - extra_padding == crop_slice.stop - crop_slice.start, f"{cropped_image_size} {image_size} {extra_padding} {crop_slice.stop - crop_slice.start}"
with torch.no_grad():
    batch_size = 10
    for i in tqdm(range(0, len(rgb_dataset), batch_size), desc="Computing embeddings"):
        embedding = dinov2_vitl14.forward_features(rgb_dataset[i:i+batch_size, :, crop_slice, crop_slice].to(device))
        for key in dino_embeddings:
            if not dino_embeddings[key] is None:
                dino_embeddings[key].append(embedding[key])

for key in dino_embeddings:
    print(f"Processing {key}")
    if len(dino_embeddings[key]) == 0:
        print(f"No entries for {key}, skipping")
        continue
    print(len(dino_embeddings[key]))
    dino_embeddings[key] = torch.cat(dino_embeddings[key])
    print(f"{key} shape: {dino_embeddings[key].shape}")
    torch.save(dino_embeddings[key], DATA_DIR / f"{key}.pt")
    print(f"Saved {key}.pt")


num_plot = 50
print(f"Dino embeddings shape: {dino_embeddings['x_norm_clstoken'].shape}")
embeddings = dino_embeddings["x_norm_clstoken"].cpu().numpy()
embeddings_flat = embeddings[:num_plot].reshape(-1, embeddings.shape[-1])
print(f"embeddings shape: {embeddings.shape}")
pca = PCA(n_components=3)
scaler = StandardScaler()
embeddings_pca = pca.fit_transform(scaler.fit_transform(embeddings_flat))
print(f"embeddings_pca shape: {embeddings_pca.shape}")
intensities = dataset[:num_plot].sum(dim=(2, 3))
colors = (intensities[:, 2] - intensities[:, 3]) / (intensities[:, 2] + intensities[:, 3])
# colors = intensities[:, 2] / (intensities[:, 0] + intensities[:, 1])
colors = colors.cpu().numpy()
sns.scatterplot(x=embeddings_pca[:, 0], y=embeddings_pca[:, 1], hue=colors)
plt.savefig("dino_embeddings.png")

# plot an 5 images with their patch embeddings for each of the 3 FUCCI color classes: geminin, mixes, and cdt1
num_show = 10
n_comp = 3
patch_embeddings = dino_embeddings["x_norm_patchtokens"][:num_show].cpu().numpy()
print(f"patch_embeddings shape: {patch_embeddings.shape}")
patch_embeddings_flat = patch_embeddings.reshape(-1, patch_embeddings.shape[-1])
print(f"patch_embeddings_flat shape: {patch_embeddings_flat.shape}")
pca = NMF(n_components=n_comp)
sig = lambda x: 1 / (1 + np.exp(-x))
patch_embeddings_pca = pca.fit_transform(sig(scaler.fit_transform(patch_embeddings_flat)))
# pca = PCA(n_components=n_comp)
# patch_embeddings_pca = pca.fit_transform(scaler.fit_transform(patch_embeddings_flat))
patch_embeddings_pca = patch_embeddings_pca.reshape((num_show, -1, n_comp))[..., n_comp - 3:]
print(f"patch_embeddings_pca shape: {patch_embeddings_pca.shape}")
img_size = int(math.sqrt(patch_embeddings_pca.shape[1]))
assert int(patch_embeddings_pca.shape[1]) == int(img_size ** 2), f"{patch_embeddings_pca.shape[1]} {img_size}"
assert math.floor(cropped_image_size / img_size) == math.ceil(cropped_image_size / img_size), f"{cropped_image_size} {img_size}, {cropped_image_size / img_size}"
patch_pca_images = patch_embeddings_pca.reshape((num_show, img_size, img_size, 3))
patch_pca_images = torch.from_numpy(patch_pca_images)
patch_pca_images = patch_pca_images.permute(0, 3, 1, 2)
print(f"patch_pca_images shape: {patch_pca_images.shape}")
patch_pca_images = resize(patch_pca_images, (cropped_image_size, cropped_image_size), interpolation='nearest')
print(f"patch_pca_images shape post-resize: {patch_pca_images.shape}")
images = rgb_dataset[:num_show, :, crop_slice, crop_slice]
print(f"images shape: {images.shape}")
images = torch.cat([images, patch_pca_images])
grid = save_image(images, Path.cwd() / "pca_images.png", nrow=num_show)

# plot histograms of the patch embedding PCs
num_bins = 50
for i in range(3):
    plt.hist(patch_embeddings_pca[:, :, i].flatten(), bins=num_bins)
    plt.savefig(f"pca{i}.png")
    plt.clf()
    plt.close()