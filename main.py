import torch
from pathlib import Path
from pipeline import create_image_paths_file, image_paths_from_folders, create_data_path_index, save_channel_names
from pipeline import segmentator_setup, get_masks, clean_and_save_masks, crop_images, resize_and_normalize

# DATA_DIR = Path("/data/ishang/CCNB1-dataset/") # NEEDS TO BE ABSOLUTE PATH
DATA_DIR = Path("/home/ishang/HPA-embedding/dev-dataset-CCNB1/")
CHANNEL_NAMES = ["nuclei", "microtubule", "cyclinb1"]
DAPI, TUBL, CALB2 = 0, 1, None
device_num = 7
rebuild = True
output_image_size = 256 # final sidelength of the image in pixels
cutoff = output_image_size * 3 # pixel square bounding box
nuc_margin = 50

# DATA_DIR = Path("/home/ishang/HPA-embedding/dev-dataset-FUCCI/")
# CHANNEL_NAMES = ["nuclei", "microtubule", "Geminin", "CDT1"]
# DAPI, TUBL, CALB2 = 0, 1, None
# device_num = 6
# rebuild = True
# output_image_size = 256 # final sidelength of the image in pixels
# cutoff = output_image_size * 2 # pixel square bounding box
# nuc_margin = 50

data_paths_file, num_paths = create_image_paths_file(DATA_DIR, overwrite=True)
image_paths = image_paths_from_folders(data_paths_file)
assert len(image_paths) == num_paths

device = f"cuda:{device_num}" if torch.cuda.is_available() else "cpu"
multi_channel_model = True if CALB2 is not None else False

segmentator = segmentator_setup(multi_channel_model, device)
image_paths, nuclei_mask_paths, cell_mask_paths = get_masks(segmentator, image_paths, CHANNEL_NAMES, DAPI, TUBL, CALB2, rebuild=rebuild)
if rebuild:
    assert len(image_paths) == len(nuclei_mask_paths) == len(cell_mask_paths) == num_paths, f"Number of images, masks, and target paths are not equal with rebuild={rebuild}"

num_original, num_removed = clean_and_save_masks(cell_mask_paths, nuclei_mask_paths, rm_border=False, remove_size=1000)
print("Fraction removed:", num_removed / num_original)
print("Total cells removed:", num_removed)
print("Total cells remaining:", num_original - num_removed)

seg_image_paths, seg_cell_mask_paths, seg_nuclei_mask_paths = crop_images(image_paths, cell_mask_paths, nuclei_mask_paths, cutoff, nuc_margin)
final_image_paths, final_cell_mask_paths, final_nuclei_mask_paths = resize_and_normalize(seg_image_paths, seg_cell_mask_paths, seg_nuclei_mask_paths, output_image_size)
create_data_path_index(final_image_paths, final_cell_mask_paths, final_nuclei_mask_paths, DATA_DIR / "index.csv", overwrite=True)
save_channel_names(DATA_DIR, CHANNEL_NAMES)

assert len(final_image_paths) == len(final_cell_mask_paths) == len(final_nuclei_mask_paths) == num_paths, f"Number of images, masks, and target paths are not equal: {len(final_image_paths)}, {len(final_cell_mask_paths)}, {len(final_nuclei_mask_paths)}, {num_paths}"