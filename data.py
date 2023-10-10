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
    def __init__(self, index_file, channel_colors=None, channels=None, batch_size=500):
        data_dir = Path(index_file).parent
        self.data_dir = data_dir
        self.channel_names = load_channel_names(data_dir)
        # self.images = torch.concat(load_dir_images(index_file))
        # image_tensors = load_dir_images(index_file)
        from pipeline import load_index_paths
        image_paths, _, _ = load_index_paths(index_file)
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
    """
    Data module for training a classifier on top of DINO embeddings of DAPI+TUBL reference channels
    Trying to match labels from a GMM or Ward cluster labeling algorithm of the FUCCI channel intensities
    """
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


class RefChannelIntensity(Dataset):
    def __init__(self, data_dir, data_name):
        self.X = torch.load(data_dir / f"ref_embeddings_{data_name}.pt")
        self.Y = torch.tensor(np.load(data_dir / f"FUCCI_log_intensity_labels_{data_name}.npy"))

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def __len__(self):
        return len(self.X)

class RefChannelIntensityDM(LightningDataModule):
    """
    Data module for training a classifier on top of DINO embeddings of DAPI+TUBL reference channels
    Trying to match labels from a GMM or Ward cluster labeling algorithm of the FUCCI channel intensities
    """
    def __init__(self, data_dir, data_name, batch_size, num_workers, split):
        super().__init__()
        self.data_dir = data_dir
        self.data_name = data_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split = split

        dataset = RefChannelIntensity(self.data_dir, self.data_name)
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

class RefChannelPseudo(Dataset):
    def __init__(self, data_dir, data_name, HPA=False, dataset=None, concat_well_stats=False):
        if HPA:
            if not dataset is None and not concat_well_stats:
                if type(dataset) == str:
                    dataset = (dataset,)
                assert type(dataset) == tuple, "Must provide tuple of datasets"
                self.X, self.Y = [], []
                for src_dataset in dataset:
                    print(f"Loading {src_dataset}_HPA_DINO_cls_concat_tokens.pt")
                    print(f"Loading {src_dataset}_pseudotime.npy")
                    self.X.append(torch.load(data_dir / f"{src_dataset}_HPA_DINO_cls_concat_tokens.pt"))
                    self.Y.append(np.load(data_dir / f"{src_dataset}_pseudotime.npy"))
                self.X = torch.cat(self.X)
                self.Y = torch.tensor(np.concatenate(self.Y))
            # elif concat_well_stats and not dataset is None and type(dataset) == str:
            #     self.X = torch.load(data_dir / f"{dataset}_HPA_DINO_cls_concat_tokens.pt")
            #     print(f"Loaded {dataset}_HPA_DINO_cls_concat_tokens.pt")
            #     self.Y = torch.tensor(np.load(data_dir / f"FUCCI_pseudotime_normalized.npy"))
            #     print(f"Loaded FUCCI_pseudotime_normalized.npy")
            #     self.well_stats = torch.load(data_dir / f"image_well_percentiles_64.pt")
            #     print(f"Loaded image_well_percentiles_64.pt")
            #     print(type(self.well_stats[0]))
            #     print(len(self.well_stats[0]))
            #     self.well_stats = self.well_stats[:, :2, :] # N x C x P
            #     self.well_stats = self.well_stats.reshape(self.well_stats.size(0), -1)
            #     self.X = torch.cat((self.X, self.well_stats), dim=1)
            #     print(f"Concatenated well stats to X, new shape: {self.X.shape}")
            elif concat_well_stats and not dataset is None:
                self.X, self.well_stats, self.Y = [], [], []
                if type(dataset) == str:
                    dataset = (dataset,)
                    if dataset[0] == "fucci" and len(dataset) == 1:
                        dataset = ("fucci_cham", "fucci_tile", "fucci_over")
                datasets = dataset
                for src_dataset in datasets:
                    print(f"Loading {src_dataset}_HPA_DINO_cls_concat_tokens.pt")
                    print(f"Loading {src_dataset}_pseudotime_normalized.npy")
                    self.X.append(torch.load(data_dir / f"{src_dataset}_HPA_DINO_cls_concat_tokens.pt"))
                    self.well_stats.append(np.load(data_dir / f"{src_dataset}_intensity_percentiles.npy"))
                    self.Y.append(np.load(data_dir / f"{src_dataset}_pseudotime_normalized.npy"))
                self.X = torch.cat(self.X)
                self.Y = torch.tensor(np.concatenate(self.Y))
                self.well_stats = np.concatenate(self.well_stats)
                self.well_stats = torch.tensor(self.well_stats).float()
                self.X = torch.cat((self.X, self.well_stats), dim=1)

            else:
                self.X = torch.load(data_dir / f"HPA_DINO_cls_{data_name}.pt")
                print(f"Loaded HPA_DINO_cls_{data_name}.pt")
                self.Y = torch.tensor(np.load(data_dir / f"FUCCI_pseudotime_{data_name}.npy")).flatten()
        else:
            self.X = torch.load(data_dir / f"ref_embeddings_{data_name}.pt")
            print(f"Loaded ref_embeddings_{data_name}.pt")
            self.Y = torch.tensor(np.load(data_dir / f"FUCCI_pseudotime_{data_name}.npy")).flatten()
        self.X, self.Y = self.X.float(), self.Y.float()

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def __len__(self):
        return len(self.X)

class RefChannelPseudoDM(LightningDataModule):
    """
    Data module for training a classifier on top of DINO embeddings of DAPI+TUBL reference channels
    Trying to match labels from a GMM or Ward cluster labeling algorithm of the FUCCI channel intensities
    """
    def __init__(self, data_dir, data_name, batch_size, num_workers, split, HPA=False, dataset=None, concat_well_stats=False):
        super().__init__()
        self.data_dir = data_dir
        self.data_name = data_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split = split

        if type(dataset) is str:
            self.dataset = RefChannelPseudo(self.data_dir, self.data_name, HPA=HPA, dataset=dataset, concat_well_stats=concat_well_stats)
            generator = torch.Generator().manual_seed(420)
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, self.split, generator=generator)
            self.split_indices = {"train": self.train_dataset.indices, "val": self.val_dataset.indices, "test": self.test_dataset.indices}
        else:
            assert type(dataset) == tuple, "Must provide tuple of datasets"
            assert len(dataset) == 3, "Must provide tuple of datasets of length 3, one for each split"
            print("Loading Training Dataset")
            self.train_dataset = RefChannelPseudo(self.data_dir, self.data_name, HPA=HPA, dataset=dataset[0], concat_well_stats=concat_well_stats)
            print("Loading Validation Dataset")
            self.val_dataset = RefChannelPseudo(self.data_dir, self.data_name, HPA=HPA, dataset=dataset[1], concat_well_stats=concat_well_stats)
            print("Loading Testing Dataset")
            self.test_dataset = RefChannelPseudo(self.data_dir, self.data_name, HPA=HPA, dataset=dataset[2], concat_well_stats=concat_well_stats)
            self.split_indices = {"train": torch.arange(len(self.train_dataset)), "val": torch.arange(len(self.val_dataset)), "test": torch.arange(len(self.test_dataset))}

    def __shared_dataloader(self, dataset, shuffle=False):
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, shuffle=shuffle)

    def train_dataloader(self):
        return self.__shared_dataloader(self.train_dataset, shuffle=True)
    
    def val_dataloader(self):
        return self.__shared_dataloader(self.val_dataset)
    
    def test_dataloader(self):
        return self.__shared_dataloader(self.test_dataset)