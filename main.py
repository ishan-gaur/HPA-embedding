import torch
from tqdm import tqdm
import numpy as np
from pathlib import Path
from pipeline import create_image_paths_file, image_paths_from_folders, create_data_path_index, save_channel_names
from pipeline import segmentator_setup, get_masks, clean_and_save_masks, crop_images, resize
from utils import min_max_normalization
utils.silent = True

# cache paths are written at the mask stage: images and masks as np
# image paths are OVERwritten at the normalizing stage: images as np
# mask paths are OVERwritten at the cleaning stage: masks as np
# cache paths are written at the cropping stage: segmented images and masks as np
# cache paths are written at the resizing stage: images and masks as pt
# these final paths are written to the index file and saved with the channel names

# DATA_DIR = Path("/data/ishang/CCNB1-dataset/") # NEEDS TO BE ABSOLUTE PATH
# DATA_DIR = Path("/home/ishang/HPA-embedding/dev-dataset-CCNB1/")
# CHANNEL_NAMES = ["nuclei", "microtubule", "cyclinb1"]
# DAPI, TUBL, CALB2 = 0, 1, None
# output_image_size = 768 # final sidelength of the image in pixels
# cutoff = output_image_size # pixel square bounding box
# nuc_margin = 50

DATA_DIR = Path("/data/ishang/FUCCI-dataset/") # NEEDS TO BE ABSOLUTE PATH
# DATA_DIR = Path("/home/ishang/HPA-embedding/dev-dataset-FUCCI/")
CHANNEL_NAMES = ["nuclei", "microtubule", "Geminin", "CDT1"]
DAPI, TUBL, CALB2 = 0, 1, None
output_image_size = 512 # final sidelength of the image in pixels
cutoff = output_image_size # pixel square bounding box
nuc_margin = 50

rebuild = False
device_num = 7

data_paths_file, num_paths = create_image_paths_file(DATA_DIR, overwrite=True)
image_paths = image_paths_from_folders(data_paths_file)
assert len(image_paths) == num_paths

device = f"cuda:{device_num}" if torch.cuda.is_available() else "cpu"
multi_channel_model = True if CALB2 is not None else False

segmentator = segmentator_setup(multi_channel_model, device)
image_paths, nuclei_mask_paths, cell_mask_paths = get_masks(segmentator, image_paths, CHANNEL_NAMES, DAPI, TUBL, CALB2, rebuild=rebuild)
if rebuild:
    assert len(image_paths) == len(nuclei_mask_paths) == len(cell_mask_paths) == num_paths, f"Number of images, masks, and target paths are not equal with rebuild={rebuild}"

for i in tqdm(range(0, len(image_paths), 100), desc="Normalizing images"):
    batch_paths = image_paths[i:min(i+100, len(image_paths))]
    images = []
    for path in batch_paths:
        images.append(np.load(path))
    images = np.concatenate(images, axis=0)
    images = min_max_normalization(images, stats=False)
    for j, path in enumerate(batch_paths):
        np.save(path, images[j])

num_original, num_removed = clean_and_save_masks(cell_mask_paths, nuclei_mask_paths, rm_border=False, remove_size=1000)
print("Fraction removed:", num_removed / num_original)
print("Total cells removed:", num_removed)
print("Total cells remaining:", num_original - num_removed)

seg_image_paths, seg_cell_mask_paths, seg_nuclei_mask_paths = crop_images(image_paths, cell_mask_paths, nuclei_mask_paths, cutoff, nuc_margin)
final_image_paths, final_cell_mask_paths, final_nuclei_mask_paths = resize(seg_image_paths, seg_cell_mask_paths, seg_nuclei_mask_paths, output_image_size)
create_data_path_index(final_image_paths, final_cell_mask_paths, final_nuclei_mask_paths, DATA_DIR / "index.csv", overwrite=True)
save_channel_names(DATA_DIR, CHANNEL_NAMES)

assert len(final_image_paths) == len(final_cell_mask_paths) == len(final_nuclei_mask_paths) == num_paths, f"Number of images, masks, and target paths are not equal: {len(final_image_paths)}, {len(final_cell_mask_paths)}, {len(final_nuclei_mask_paths)}, {num_paths}"