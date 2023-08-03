import torch
import numpy as np
from pipeline import composite_images_from_paths
from utils import get_dataset_percentiles, get_image_percentiles
from utils import min_max_normalization, rescale_normalization, threshold_normalization, percentile_normalization
from data_viz import plot_intensities, barplot_percentiles, cdf_percentiles, histplot_percentiles
from data_viz import save_image_grid, color_image_by_intensity


def pixel_range_info(args, image_paths, CHANNELS, OUTPUT_DIR):
        # if args.calc_num < 500:
        #     print("Warning: using less than 500 images to calculate pixel range may result in inaccurate pixel range")
        image_sample_paths = np.random.choice(image_paths, args.calc_num)
        image_sample = composite_images_from_paths(image_sample_paths, CHANNELS)

        values, percentiles = get_dataset_percentiles(image_sample, non_zero=False)
        thresholded_values, thresholded_percentiles = get_dataset_percentiles(image_sample)

        barplot_percentiles(percentiles, values, CHANNELS, OUTPUT_DIR / 'pixel_percentiles.png')
        barplot_percentiles(thresholded_percentiles, thresholded_values, CHANNELS, OUTPUT_DIR / 'pixel_percentiles_non_zero.png')

        image_values, image_percentiles = get_image_percentiles(image_sample)
        # histplot_percentiles(image_percentiles, image_values, CHANNELS, OUTPUT_DIR / 'image_percentiles.png')
        cdf_percentiles(image_percentiles, image_values, CHANNELS, OUTPUT_DIR / 'image_percentiles_cdf.png')
        
def normalization_dry_run(args, config, image_paths, CHANNELS, OUTPUT_DIR, device, plot_channel=None):
    image_sample_paths = np.random.choice(image_paths, args.calc_num)
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
    image_sample = rescale_normalization(image_sample[:args.viz_num], stats=False)
    image_sample = torch.from_numpy(image_sample).to(device)
    norm_images = torch.from_numpy(norm_images[:args.viz_num]).to(device)
    images = torch.cat([image_sample, norm_images], dim=0)
    print(images.shape)
    if channel_slice is None:
        channel_slice = slice(0, len(CHANNELS))
    save_image_grid(images[:, channel_slice], norm_image_file, args.viz_num, config.cmaps[channel_slice])

def image_by_level_set(args, image_paths, CHANNELS, OUTPUT_DIR):
    image_sample_paths = np.random.choice(image_paths, args.viz_num)
    image_sample = composite_images_from_paths(image_sample_paths, CHANNELS)
    image_sample = rescale_normalization(image_sample, stats=False)
    color_image_by_intensity(image_sample, OUTPUT_DIR / 'intensity_colored_images.png')

def sample_sharpness(images, kernel_size=3):
    from kornia.filters import laplacian, sobel
    # laplacian = Laplacian(3)
    # image_sharpness = laplacian_images.mean(dim=(1,2,3))
    # image_sharpness = laplacian_images.sum(dim=(1,2,3))
    # laplacian_images = laplacian(images, kernel_size=kernel_size)
    # image_sharpness = laplacian_images.std(dim=(1,2,3))
    image_sharpness = sobel(images)
    image_sharpness = image_sharpness.std(dim=(1,2,3))
    return image_sharpness