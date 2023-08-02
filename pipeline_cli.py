import argparse
import config
from pathlib import Path

from pipeline import create_image_paths_file, image_paths_from_folders, create_data_path_index, load_channel_names, save_channel_names
from pipeline import composite_images_from_paths
from pipeline import segmentator_setup, get_masks, clean_and_save_masks, crop_images, resize_and_normalize
from data_viz import plot_intensities, barplot_percentiles, histplot_percentiles, cdf_percentiles
from data_viz import save_image_grid, color_image_by_intensity
from utils import min_max_normalization, rescale_normalization, threshold_normalization, percentile_normalization
from utils import get_dataset_percentiles, get_image_percentiles


import torch
import numpy as np

parser = argparse.ArgumentParser(description='Dataset preprocessing pipline')
parser.add_argument('--data_dir', type=str, help='Path to dataset, should be absolute path', required=True)
parser.add_argument('--output_dir', type=str , help='Path to output directory, should be absolute path')
NORM, PIX_RANGE, INT_IMG = 0, 1, 2
stats_opt = ['norm', 'pix_range', 'int_img']
parser.add_argument('--stats', type=str, help=f"Image stats to show, options include: {stats_opt}", choices=stats_opt)
parser.add_argument('--viznum', type=int, default=5, help='Number of samples to show')
parser.add_argument('--calcnum', type=int, default=30, help='Number of samples to use for calculating image stats')
# parser.add_argument('--images', action='store_true', help='Save images')
parser.add_argument('--device', type=int, default=7, help='GPU device number')
parser.add_argument('--config', type=str, default='config.py', help='Path to config file')


args = parser.parse_args()

DATA_DIR = Path(args.data_dir)
OUTPUT_DIR = Path(args.output_dir) if args.output_dir is not None else Path.cwd() / "output"
if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True)
    print(f"Created output directory {OUTPUT_DIR}")

data_paths_file, num_paths = create_image_paths_file(DATA_DIR)
image_paths = image_paths_from_folders(data_paths_file)

CHANNELS = load_channel_names(DATA_DIR) if config.channels is None else config.channels
if config.channels is None:
    save_channel_names(DATA_DIR, CHANNELS)

device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

if args.stats is not None:
    if args.stats == stats_opt[PIX_RANGE]:
        # if args.calcnum < 500:
        #     print("Warning: using less than 500 images to calculate pixel range may result in inaccurate pixel range")
        image_sample_paths = np.random.choice(image_paths, args.calcnum)
        image_sample = composite_images_from_paths(image_sample_paths, CHANNELS)

        values, percentiles = get_dataset_percentiles(image_sample, non_zero=False)
        thresholded_values, thresholded_percentiles = get_dataset_percentiles(image_sample)

        barplot_percentiles(percentiles, values, CHANNELS, OUTPUT_DIR / 'pixel_percentiles.png')
        barplot_percentiles(thresholded_percentiles, thresholded_values, CHANNELS, OUTPUT_DIR / 'pixel_percentiles_non_zero.png')

        image_values, image_percentiles = get_image_percentiles(image_sample)
        # histplot_percentiles(image_percentiles, image_values, CHANNELS, OUTPUT_DIR / 'image_percentiles.png')
        cdf_percentiles(image_percentiles, image_values, CHANNELS, OUTPUT_DIR / 'image_percentiles_cdf.png')
        
if args.stats == stats_opt[NORM]:
        image_sample_paths = np.random.choice(image_paths, args.calcnum)
        image_sample = composite_images_from_paths(image_sample_paths, CHANNELS)

        if config.norm_strategy == 'min_max':
            norm_images, mins, maxes, intensities = min_max_normalization(image_sample)
        if config.norm_strategy == 'rescale':
            norm_images, mins, maxes, intensities = rescale_normalization(image_sample)
        if config.norm_strategy == 'threshold':
            assert config.norm_min is not None and config.norm_max is not None, "Must specify norm_min and norm_max in config for threshold normalization"
            norm_images, mins, maxes, intensities = threshold_normalization(image_sample, config.norm_min, config.norm_max)
        if config.norm_strategy == 'percentile':
            assert config.norm_min is not None and config.norm_max is not None, "Must specify norm_min and norm_max in config for percentile normalization"
            norm_images, mins, maxes, intensities = percentile_normalization(image_sample, config.norm_min, config.norm_max)

        if config.norm_strategy == 'threshold':
            strategy = f"{config.norm_strategy}_{config.norm_min}_{config.norm_max}"
        elif config.norm_strategy == 'percentile':
            strategy = f"{config.norm_strategy}_{config.norm_min}_{config.norm_max}"
        else:
            strategy = config.norm_strategy 
            
        intensity_hist_file = OUTPUT_DIR / 'image_intensity_histogram.png'
        print("Plotting image intensity histogram")
        plot_intensities(intensities, CHANNELS, intensity_hist_file, log=True)
        print(f"Saved image intensity histogram to {intensity_hist_file}")

        norm_intensities = norm_images.transpose(1, 0, 2, 3).reshape(len(CHANNELS), -1)
        norm_intensity_hist_file = OUTPUT_DIR / f'norm_{strategy}_image_intensity_histogram.png'
        print("Plotting normalized image intensity histogram")
        plot_intensities(norm_intensities, CHANNELS, norm_intensity_hist_file, log=True)
        print(f"Saved normalized image intensity histogram to {norm_intensity_hist_file}")

        mins, maxes = mins.squeeze(), maxes.squeeze()
        # print the 0, 25, 50, 75, 100 percentiles
        for channel in range(len(CHANNELS)):
            print(f"Channel {CHANNELS[channel]}")
            print("\tImage min and max percentiles:")
            print(f"\t{np.percentile(mins[channel], [0, 25, 50, 75, 100])}")
            print(f"\t{np.percentile(maxes[channel], [0, 25, 50, 75, 100])}")

        # save a grid of images before and after
        print("Saving image samples")
        norm_image_file = OUTPUT_DIR / f'normalized_images_{strategy}.png'
        image_sample = rescale_normalization(image_sample[:args.viznum], stats=False)
        image_sample = torch.from_numpy(image_sample).to(device)
        norm_images = torch.from_numpy(norm_images[:args.viznum]).to(device)
        images = torch.cat([image_sample, norm_images], dim=0)
        print(images.shape)
        save_image_grid(images[:, 2:], norm_image_file, args.viznum, config.cmaps[2:])

if args.stats == stats_opt[INT_IMG]:
    image_sample_paths = np.random.choice(image_paths, args.viznum)
    image_sample = composite_images_from_paths(image_sample_paths, CHANNELS)
    image_sample = rescale_normalization(image_sample, stats=False)
    color_image_by_intensity(image_sample, OUTPUT_DIR / 'intensity_colored_images.png')