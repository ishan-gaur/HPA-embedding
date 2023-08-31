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
    # images are now C x B x H x W
    num_channels = images.shape[0]
    if not silent: print("Calculating image min and max")
    mins = images.min(axis=(1, 2, 3), keepdims=True)
    maxes = images.max(axis=(1, 2, 3), keepdims=True)
    intensities = images.reshape(num_channels, -1)
    return mins, maxes, intensities

def min_max_normalization(images, stats=True, non_zero=True):
    # images come in as B x C x H x W
    images = images.transpose(1, 0, 2, 3)
    mins, maxes, intensities = get_min_max_int(images)
    if non_zero:
        min_nonzero = [intensities[channel][intensities[channel] > 0].min() for channel in range(len(intensities))]
        min_nonzero = np.array(min_nonzero).reshape(mins.shape)
        mins = min_nonzero
    if not silent: print("Normalizing images")
    images = np.clip(images, mins, maxes)
    norm_images = (images - mins) / (maxes - mins)
    # convert norm images back to B x C x H x W
    norm_images = norm_images.transpose(1, 0, 2, 3)
    norm_images = norm_images.astype(np.float32)
    if not stats:
        return norm_images
    # intensities for all images for a given channel
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