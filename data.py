import torch    
from torch.utils.data import Dataset
from microfilm.microplot import microshow
from pipeline import load_channel_names, load_dir_images

class CellImageDataset(Dataset):
    # images are C x H x W
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.channel_names = load_channel_names(data_dir)
        self.images = torch.concat(load_dir_images(data_dir))
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]

    def get_channel_names(self):
        return self.channel_names

class ImageViewer:
    def __init__(self, channel_colors, channel_names):
        assert len(channel_names) == len(channel_colors), "Number of channel colors and channel names must be equal"
        self.channel_colors = channel_colors
        self.channel_names = channel_names

    def view(self, image):
        if type(image) == torch.Tensor:
            image = image.cpu().numpy()
        microshow(image, cmaps=self.channel_colors)
