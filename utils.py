import numpy as np

silent = False

def get_dataset_percentiles(images, percentiles=[0, 25, 50, 90, 99, 99.99], non_zero=True):
    num_channels = images.shape[1]
    images = images.transpose(1, 0, 2, 3).reshape(num_channels, -1)
    if not silent: print("Calculating dataset pixel percentiles")
    if non_zero:
        images = [channel_pixels[channel_pixels > 0] for channel_pixels in images]
    values = np.array([np.percentile(channel_pixels, percentiles) for channel_pixels in images])
    return values, percentiles

def get_image_percentiles(images, percentiles=[90, 99, 99.99], non_zero=True):
    # returns a list of percentiles for each image by channel: C x B x P
    num_channels, num_images = images.shape[1], images.shape[0]
    channel_images = images.transpose(1, 0, 2, 3).reshape(num_channels, num_images, -1)
    if not silent: print("Calculating image pixel percentiles")
    if non_zero:
        channel_images = [[image[image > 0] for image in images] for images in channel_images]
    values = np.array([[np.percentile(image, percentiles) for image in images] for images in channel_images])
    return values, percentiles

def get_min_max_int(images):
    num_channels = images.shape[1]
    if not silent: print("Calculating image min and max")
    mins = images.min(axis=(2, 3), keepdims=True)
    maxes = images.max(axis=(2, 3), keepdims=True)
    if not silent: print("Reshaping images to calculate intensities")
    intensities = images.transpose(1, 0, 2, 3).reshape(num_channels, -1)
    return mins, maxes, intensities

def min_max_normalization(images, stats=True):
    mins, maxes, intensities = get_min_max_int(images)
    if not silent: print("Normalizing images")
    norm_images = (images - mins) / (maxes - mins)
    norm_images = norm_images.astype(np.float32)
    if not stats:
        return norm_images
    return norm_images, mins, maxes, intensities

def rescale_normalization(images, stats=True):
    dtype_max = np.iinfo(images.dtype).max
    if not silent: print("Normalizing images")
    norm_images = images / dtype_max
    norm_images = norm_images.astype(np.float32)
    if not stats:
        return norm_images
    mins, maxes, intensities = get_min_max_int(norm_images)
    return norm_images, mins, maxes, intensities

def threshold_normalization(images, min_int, max_int, stats=True):
    norm_images = np.clip(images, min_int, max_int)
    norm_images = (norm_images - min_int) / (max_int - min_int)
    norm_images = norm_images.astype(np.float32)
    if not stats:
        return norm_images
    mins, maxes, intensities = get_min_max_int(images)
    return norm_images, mins, maxes, intensities

def percentile_normalization(images, perc_min, perc_max, stats=True):
    if not silent: print("Calculating image percentiles")
    percentiles, _ = get_image_percentiles(images, percentiles=[perc_min, perc_max])
    percentiles = percentiles.transpose(1, 0, 2)
    min_int, max_int = percentiles[..., 0], percentiles[..., 1]
    return threshold_normalization(images, min_int[..., None, None], max_int[..., None, None], stats=stats)