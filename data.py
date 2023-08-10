from pathlib import Path
import torch    
from lightning.pytorch import LightningDataModule
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
from microfilm.microplot import microshow
from microfilm.colorify import multichannel_to_rgb
from pipeline import load_channel_names, load_dir_images
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
from glob import glob

class CellImageDataset(Dataset):
    # images are C x H x W
    def __init__(self, index_file, channel_colors=None, channels=None):
        data_dir = Path(index_file).parent
        self.data_dir = data_dir
        self.channel_names = load_channel_names(data_dir)
        # self.images = torch.concat(load_dir_images(index_file))
        # image_tensors = load_dir_images(index_file)
        from pipeline import load_index_paths
        image_paths, _, _ = load_index_paths(index_file)
        batch_size = 500
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Loading dataset images"):
            image_tensors = []
            for image_path in image_paths[i:i+batch_size]:
                image_tensors.append(torch.load(Path(image_path)))
            image_tensors = torch.concat(image_tensors)
            if i == 0:
                self.images = image_tensors
            else:
                self.images = torch.concat((self.images, image_tensors))

        print(f"Loaded {len(self.images)} images from {len(image_paths)} files.")
        
        self.channel_colors = channel_colors
        self.channels = channels if channels is not None else list(range(len(self.channel_names)))
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.channels is None:
            return self.images[idx]
        else:
            return self.images[idx, self.channels]

    def get_channel_names(self):
        if self.channels is None:
            return self.channel_names
        else:
            return [self.channel_names[i] for i in self.channels]

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
        image = self.__getitem__(idx).cpu().numpy()
        if self.channel_colors is None:
            self.__set_default_channel_colors__()
        microshow(image, cmaps=self.channel_colors)

    def convert_to_rgb(self, i):
        nans = torch.sum(torch.isnan(self.__getitem__(i)))
        if nans > 0:
            print(f"Warning: {nans} NaNs in image {i}")
        rgb_image, _, _, _ = multichannel_to_rgb(self.__getitem__(i).numpy(), cmaps=self.channel_colors,
                                                 limits=(np.nanmin(self.__getitem__(i)), np.nanmax(self.__getitem__(i))))
        return torch.Tensor(rgb_image)

    def as_rgb(self, channel_colors=None, num_workers=1):
        if self.channel_colors is None:
            if channel_colors is not None:
                self.channel_colors = channel_colors
            else:
                self.__set_default_channel_colors__()
        assert len(self.channels) == len(self.channel_colors), "Number of channel colors and channels must be equal"
        device = self.images.device
        self.images.cpu()
        rgb_images = []
        for i in tqdm(range(self.__len__()), total=self.__len__(), desc="Converting to RGB"):
            rgb_image, _, _, _ = multichannel_to_rgb(self.__getitem__(i).numpy(), cmaps=self.channel_colors)
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

class DinoRefToCC(Dataset):
    def __init__(self, data_dir, data_name, ward, num_classes):
        print(f"Loading ref_embeddings_{data_name}.pt")
        self.X = torch.load(data_dir / f"ref_embeddings_{data_name}.pt")
        if ward:
            file_path = Path(data_dir) / f"ward_{num_classes}_probs_{data_name}.pt"
            if not file_path.exists():
                raise ValueError(f"File {str(file_path)} does not exist")
            print(f"Loading {str(file_path)}")
            self.Y = torch.load(file_path)
            self.Y = torch.nn.functional.one_hot(self.Y).float()
            print(self.Y.shape)
        else:
            self.Y = torch.load(data_dir / f"gmm_probs_{data_name}.pt")

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
    def __len__(self):
        return self.X.size(0)

class RefToCC(Dataset):
    def __init__(self, data_dir, data_name, ward, num_classes):
        self.X = CellImageDataset(data_dir / f"index_{data_name}.csv", channels=[0, 1])
        if ward:
            file_path = Path(data_dir) / f"ward_{num_classes}_probs_{data_name}.pt"
            if not file_path.exists():
                raise ValueError(f"File {str(file_path)} does not exist")
            print(f"Loading {str(file_path)}")
            self.Y = torch.load(file_path)
            self.Y = torch.nn.functional.one_hot(self.Y).float()
        else:
            self.Y = torch.load(data_dir / f"gmm_probs_{data_name}.pt")

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def __len__(self):
        return len(self.X)

class CellCycleModule(LightningDataModule):
    def __init__(self, data_dir, data_name, batch_size, num_workers, split, ward, num_classes):
        super().__init__()
        self.data_dir = data_dir
        self.data_name = data_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split = split
        self.ward = ward
        self.num_classes = num_classes

        dataset = DinoRefToCC(self.data_dir, self.data_name, self.ward, self.num_classes)
        generator = torch.Generator().manual_seed(420)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, self.split, generator=generator)

    def __shared_dataloader(self, dataset, shuffle=False):
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, shuffle=shuffle)

    def train_dataloader(self):
        return self.__shared_dataloader(self.train_dataset, shuffle=True)
    
    def val_dataloader(self):
        return self.__shared_dataloader(self.val_dataset)
    
    def test_dataloader(self):
        return self.__shared_dataloader(self.test_dataset)

class RefChannelCellCycle(LightningDataModule):
    def __init__(self, data_dir, data_name, batch_size, num_workers, split, ward, num_classes):
        super().__init__()
        self.data_dir = data_dir
        self.data_name = data_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split = split
        self.ward = ward
        self.num_classes = num_classes

        dataset = RefToCC(self.data_dir, self.data_name, self.ward, self.num_classes)
        generator = torch.Generator().manual_seed(420)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, self.split, generator=generator)

    def __shared_dataloader(self, dataset, shuffle=False):
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, shuffle=shuffle)

    def train_dataloader(self):
        return self.__shared_dataloader(self.train_dataset, shuffle=True)
    
    def val_dataloader(self):
        return self.__shared_dataloader(self.val_dataset)
    
    def test_dataloader(self):
        return self.__shared_dataloader(self.test_dataset)