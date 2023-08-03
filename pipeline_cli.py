import inspect
import argparse
import config
from pathlib import Path

import utils
import pipeline
from pipeline import create_image_paths_file, image_paths_from_folders, create_data_path_index, load_index_paths, load_channel_names, save_channel_names
from pipeline import segmentator_setup, get_masks, normalize_images, clean_and_save_masks, crop_images, resize
from stats import pixel_range_info, normalization_dry_run, image_by_level_set

import torch
import numpy as np

utils.silent = True
pipeline.suppress_warnings = True

stats_opt = ['norm', 'pix_range', 'int_img']

parser = argparse.ArgumentParser(description='Dataset preprocessing pipline')
parser.add_argument('--data_dir', type=str, help='Path to dataset, should be absolute path', required=True)
parser.add_argument('--output_dir', type=str , help='Path to output directory, should be absolute path')
parser.add_argument('--name', type=str, help='Name of dataset')
parser.add_argument('--stats', type=str, help=f"Image stats to show, options include: {stats_opt}", choices=stats_opt)
parser.add_argument('--viz_num', type=int, default=5, help='Number of samples to show')
parser.add_argument('--calc_num', type=int, default=30, help='Number of samples to use for calculating image stats')
parser.add_argument('--all', action='store_true', help='Run all steps')
parser.add_argument('--image_mask_cache', action='store_true', help='Save images')
parser.add_argument('--normalize', action='store_true', help='Normalize images')
parser.add_argument('--clean_masks', action='store_true', help='Clean masks: remove small objects and join cells without nuclei, etc.')
parser.add_argument('--single_cell', action='store_true', help='Crop and save single cell images')
parser.add_argument('--device', type=int, default=7, help='GPU device number')
parser.add_argument('--rebuild', action='store_true', help='Rebuild specifed steps even if files exist')

args = parser.parse_args()

DATA_DIR = Path(args.data_dir)
OUTPUT_DIR = Path(args.output_dir) if args.output_dir is not None else Path.cwd() / "output"
if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True)
    print(f"Created output directory {OUTPUT_DIR}")

BASE_INDEX = DATA_DIR / "index.csv"
NORM_SUFFIX = f"_{config.norm_strategy}{f'_{config.norm_min}_{config.norm_max}' if config.norm_strategy in ['threshold', 'percentile'] else ''}"
NORM_INDEX = DATA_DIR / f"index{NORM_SUFFIX}.csv"
NAME_INDEX = DATA_DIR / f"index_{args.name}.csv"

NORM, PIX_RANGE, INT_IMG = 0, 1, 2

CHANNELS = load_channel_names(DATA_DIR) if config.channels is None else config.channels
if config.channels is None:
    save_channel_names(DATA_DIR, CHANNELS)
DAPI, TUBL, CALB2 = config.dapi, config.tubl, config.calb2

device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
print(f"Using device {device}")


if args.stats is not None:
    data_paths_file, num_paths = create_image_paths_file(DATA_DIR)
    image_paths = image_paths_from_folders(data_paths_file)
    if args.stats == stats_opt[PIX_RANGE]:
        pixel_range_info(args, image_paths, CHANNELS, OUTPUT_DIR)
    if args.stats == stats_opt[NORM]:
        normalization_dry_run(args, config, image_paths, CHANNELS, OUTPUT_DIR, device)
    if args.stats == stats_opt[INT_IMG]:
        image_by_level_set(args, image_paths, CHANNELS, OUTPUT_DIR)

if args.image_mask_cache or args.all:
    print("Caching composite images and getting segmentation masks")
    data_paths_file, num_paths = create_image_paths_file(DATA_DIR)
    image_paths = image_paths_from_folders(data_paths_file)
    if BASE_INDEX.exists() and not args.rebuild:
        print("Index file already exists, skipping. Set --rebuild to overwrite.")
    multi_channel_model = True if CALB2 is not None else False
    segmentator = segmentator_setup(multi_channel_model, device)
    image_paths, nuclei_mask_paths, cell_mask_paths = get_masks(segmentator, image_paths, CHANNELS, DAPI, TUBL, CALB2, rebuild=args.rebuild)
    create_data_path_index(image_paths, cell_mask_paths, nuclei_mask_paths, BASE_INDEX, overwrite=True)

if args.normalize or args.all:
    print("Normalizing images")
    if NORM_INDEX.exists() and not args.rebuild:
        print("Index file already exists, skipping. Set --rebuild to overwrite.")
    assert BASE_INDEX.exists(), "Index file does not exist, run --image_mask_cache first"
    assert config.norm_strategy is not None, "Normalization strategy not set in config"
    image_paths, _, _ = load_index_paths(BASE_INDEX)
    norm_paths = normalize_images(image_paths, config.norm_strategy, NORM_SUFFIX)
    create_data_path_index(norm_paths, cell_mask_paths, nuclei_mask_paths, NORM_INDEX, overwrite=True)

if args.clean_masks or args.all:
    print("Cleaning masks")
    assert BASE_INDEX.exists(), "Index file does not exist, run --image_mask_cache first"
    num_original, num_removed = clean_and_save_masks(cell_mask_paths, nuclei_mask_paths, rm_border=config.rm_border, remove_size=config.remove_size)
    print("Fraction removed:", num_removed / num_original)
    print("Total cells removed:", num_removed)
    print("Total cells remaining:", num_original - num_removed)

if args.single_cell or args.all:
    print("Cropping single cell images")
    assert NORM_INDEX.exists(), "Index file for normalized images does not exist, run --normalize first"
    image_paths, cell_mask_paths, nuclei_mask_paths = load_index_paths(NORM_INDEX)
    seg_image_paths, seg_cell_mask_paths, seg_nuclei_mask_paths = crop_images(image_paths, cell_mask_paths, nuclei_mask_paths, config.cutoff, config.nuc_margin)
    final_image_paths, final_cell_mask_paths, final_nuclei_mask_paths = resize(seg_image_paths, seg_cell_mask_paths, seg_nuclei_mask_paths, config.output_image_size)
    create_data_path_index(final_image_paths, final_cell_mask_paths, final_nuclei_mask_paths, NAME_INDEX, overwrite=True)

    # save the source of the config module to the data directory with name args.name + '.py'
    # this will allow us to reproduce the results later
    with open(DATA_DIR / f"{args.name}.py", "w") as f:
        f.write("\n\n# Source of config module:\n")
        f.write(f"\n\n# Using normalized images from {NORM_INDEX}:\n")
        f.write(inspect.getsource(config))