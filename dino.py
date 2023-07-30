from pathlib import Path
import torch
from data import CellImageDataset, ImageViewer, SimpleDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from tqdm import tqdm
import numpy as np

# DATA_DIR = Path("/home/ishang/HPA-embedding/dev-dataset-CCNB1/")
# COLOR_MAP = ["pure_blue", "pure_red", "pure_green"]
DATA_DIR = Path("/home/ishang/HPA-embedding/dev-dataset-FUCCI/")
# DATA_DIR = Path("/data/ishang/FUCCI-dataset/")
COLOR_MAP = ["pure_blue", "pure_yellow", "pure_red", "pure_green"]
device = "cuda:6"

dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
dataset = CellImageDataset(DATA_DIR)


if SimpleDataset.has_cache_files(DATA_DIR / "rgb_dataset.pt"):
    rgb_dataset = SimpleDataset(path=(DATA_DIR / "rgb_dataset.pt"))
else:
    rgb_dataset = dataset.as_rgb(num_workers=128)
    for i in range(rgb_dataset[0].shape[0]):
        plt.imshow(rgb_dataset[0][i, :, :].numpy())
        plt.show()
    rgb_dataset.save(DATA_DIR / "rgb_dataset.pt")

# rgb_dataset = torch.cat([torch.load(DATA_DIR / f"rgb_dataset-{i * 5000}.pt") for i in range(0, 1)])
# rgb_dataset = SimpleDataset(tensor=rgb_dataset)

dinov2_vitl14.to(device)
batch_size = 10
embeddings = []
with torch.no_grad():
    for i in tqdm(range(0, len(rgb_dataset), batch_size), desc="Computing embeddings"):
        embedding = dinov2_vitl14(rgb_dataset[i:i+batch_size, :, 2:-2, 2:-2].to(device))
        embeddings.append(embedding)
embeddings = torch.cat(embeddings)
print(embeddings.shape)

embeddings = embeddings.cpu().numpy()
np.save(DATA_DIR / "embeddings.npy", embeddings)
embeddings = np.load(DATA_DIR / "embeddings.npy")

num_plot = 5000
pca = PCA(n_components=2)
pca.fit(embeddings[:num_plot])
embeddings_pca = pca.transform(embeddings[:num_plot])

intensities = dataset[:num_plot].sum(dim=(2, 3))
colors = (intensities[:, 2] - intensities[:, 3]) / (intensities[:, 2] + intensities[:, 3])
colors = colors.cpu().numpy()
sns.scatterplot(x=embeddings_pca[:, 0], y=embeddings_pca[:, 1], hue=colors)
plt.savefig("dino_embeddings.png")