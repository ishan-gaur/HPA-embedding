import torch    
from torch.utils.data import Dataset, TensorDataset
from microfilm.microplot import microshow
from microfilm.colorify import multichannel_to_rgb
from pipeline import load_channel_names, load_dir_images
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
from glob import glob

class CellImageDataset(Dataset):
    # images are C x H x W
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.channel_names = load_channel_names(data_dir)
        self.images = torch.concat(load_dir_images(data_dir))
        self.channel_colors = None
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]

    def get_channel_names(self):
        return self.channel_names

    def set_channel_colors(self, channel_colors):
        self.channel_colors = channel_colors

    def get_channel_colors(self):
        return self.channel_colors

    def __set_default_channel_colors__(self):
        if len(self.channel_names) == 1:
            self.channel_colors = ["gray"]
        elif len(self.channel_names) == 2:
            self.channel_colors = ["blue", "red"]
        elif len(self.channel_names) == 3:
            self.channel_colors = ["blue", "green", "red"]
        else:
            raise ValueError(f"channel_colors not set and not suitable defaults for {len(self.channel_names)} channels.")

    def view(self, idx):
        image = self.images[idx].cpu().numpy()
        if self.channel_colors is None:
            self.__set_default_channel_colors__()
        microshow(image, cmaps=self.channel_colors)

    def convert_to_rgb(self, i):
        nans = torch.sum(torch.isnan(self.images[i]))
        if nans > 0:
            print(f"Warning: {nans} NaNs in image {i}")
        rgb_image, _, _, _ = multichannel_to_rgb(self.images[i].numpy(), cmaps=self.channel_colors,
                                                 limits=(np.nanmin(self.images[i]), np.nanmax(self.images[i])))
        return torch.Tensor(rgb_image)

    def as_rgb(self, channel_colors=None, num_workers=1):
        if self.channel_colors is None:
            if channel_colors is None:
                self.__set_default_channel_colors__()
            else:
                self.channel_colors = channel_colors
        device = self.images.device
        self.images.cpu()
        rgb_images = []
        for i in tqdm(range(self.__len__()), total=self.__len__(), desc="Converting to RGB"):
            rgb_image, _, _, _ = multichannel_to_rgb(self.images[i].numpy(), cmaps=self.channel_colors)
            rgb_images.append(torch.Tensor(rgb_image))
        rgb_images = torch.stack(rgb_images)
        rgb_images.to(device)
        rgb_images = rgb_images[..., :-1].permute(0, 3, 1, 2)
        # rgb_images = rgb_images.permute(0, 3, 1, 2)
        rgb_dataset = SimpleDataset(rgb_images)
        self.images.to(device)
        return rgb_dataset


class ImageViewer:
    def __init__(self, channel_colors, channel_names):
        assert len(channel_names) == len(channel_colors), "Number of channel colors and channel names must be equal"
        self.channel_colors = channel_colors
        self.channel_names = channel_names

    def view(self, image):
        if type(image) == torch.Tensor:
            image = image.cpu().numpy()
        microshow(image, cmaps=self.channel_colors)

class SimpleDataset(Dataset):
    def __init__(self, tensor=None, path=None) -> None:
        if path is not None:
            cache_files = list(path.parent.glob(f"{path.stem}-*.pt"))
            # print(cache_files)
            tensors = []
            for cache_file in tqdm(cache_files, desc="Loading SimpleDataset"):
                tensors.append(torch.load(cache_file))
            self.tensor = torch.cat(tensors)
        elif tensor is None:
            raise ValueError("Must provide either tensor or path")
        else:
            self.tensor = tensor
    
    def __getitem__(self, idx):
        return self.tensor[idx]

    def __len__(self):
        return self.tensor.size(0)

    def save(self, path, batch_size=5000):
        for i in tqdm(range(0, self.tensor.size(0), batch_size), total=self.tensor.size(0) // batch_size, desc="Saving SimpleDataset"):
            torch.save(self.tensor[i:min(self.tensor.size(0), i+batch_size)].clone(), path.with_stem(f"{path.stem}-{i}"))
            
    def has_cache_files(path):
        cache_files = list(path.parent.glob(f"{path.stem}-*.pt"))
        return len(cache_files) > 0