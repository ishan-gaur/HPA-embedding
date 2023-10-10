import torch
import numpy as np
from tqdm import tqdm

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

def get_images_percentiles(images, percentiles=[90, 99, 99.99], non_zero=True):
    # returns a list of percentiles for each batch by channel: C x P
    num_channels = images.shape[1]
    channel_images = images.transpose(1, 0, 2, 3).reshape(num_channels, -1)
    if not silent: print("Calculating well pixel percentiles")
    if non_zero:
        channel_images = [channel_pixels[channel_pixels > 0] for channel_pixels in channel_images]
    values = np.array([np.percentile(channel_pixels, percentiles) for channel_pixels in channel_images])
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
    percentiles, _ = get_images_percentiles(images, percentiles=[perc_min, perc_max]) # C x P
    print(percentiles)
    percentiles = percentiles[None, ...] # add batch dimension
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

def image_cells_sharpness(image, cell_mask):
    from kornia.filters import sobel
    if len(image.shape) != 3 or len(cell_mask.shape) != 2:
        raise ValueError(f"This method only takes single images. Input image must be of shape C x H x W and cell_mask must be of shape H x W.\
                         image {image.shape} and mask {cell_mask.shape} were given.")
    image = image[None, ...] # kornia expects a batch dimension
    image_sharpness = sobel(image)
    # filling in None in case the cell masks aren't consecutive
    cell_mask = cell_mask.astype(int)
    sharpness_levels = [None for _ in range(cell_mask.max() + 1)]
    for cell in np.unique(cell_mask):
        mask_tile = list(image.shape)
        mask_tile[-cell_mask.ndim:] = [1] * cell_mask.ndim
        image_mask = torch.tensor(cell_mask == cell).tile(mask_tile)
        cell_sharpness = image_sharpness[image_mask]
        if len(cell_sharpness) == 0:
            continue
        sharpness_levels[cell] = cell_sharpness.std()
    sharpness_levels[0] = None
    assert len(sharpness_levels) == cell_mask.max().astype(int) + 1
    return sharpness_levels


def two_sig_fig_floor(x):
    return torch.floor(x / torch.pow(10.0, torch.floor(torch.log10(x)) - 1)) * torch.pow(10.0, torch.floor(torch.log10(x)) - 1)


def get_intensity_metrics(images, cell_masks, nuclei_masks, batch_size=32):
    # Gather intensity and non-zero pixel counts
    intensity_sums, nuclear_intensity_sums, cell_pixel_ct, nucleus_pixel_ct = [], [], [], []
    for i in tqdm(range(0, len(images), batch_size), desc="Computing mean intensities"):
        batch = images[i:i+batch_size]
        cell_mask_batch = cell_masks[i:i+batch_size, None]
        nuclei_mask_batch = nuclei_masks[i:i+batch_size, None]

        intensity_sums.append(torch.sum(batch, dim=(2, 3)))
        nuclear_intensity_sums.append(torch.sum(batch * (nuclei_mask_batch > 0), dim=(2, 3)))
        cell_pixel_ct.append(torch.sum(cell_mask_batch > 0, dim=(2, 3)))
        nucleus_pixel_ct.append(torch.sum(nuclei_mask_batch > 0, dim=(2, 3)))

    intensity_sums = torch.cat(intensity_sums, dim=0)
    nuclear_intensity_sums = torch.cat(nuclear_intensity_sums, dim=0)
    cell_pixel_ct = torch.cat(cell_pixel_ct, dim=0)
    nucleus_pixel_ct = torch.cat(nucleus_pixel_ct, dim=0)

    # calculate epsilon for each term
    non_zero_cells = intensity_sums > 0
    non_zero_nucs = nuclear_intensity_sums > 0
    epsilon_int, epsilon_nuc_int, epsilon_cell_ct, epsilon_nuc_ct = [], [], [], []
    for i in range(intensity_sums.shape[1]):
        epsilon_int.append(two_sig_fig_floor(torch.min(intensity_sums[non_zero_cells[:, i], i])))
        epsilon_nuc_int.append(two_sig_fig_floor(torch.min(nuclear_intensity_sums[non_zero_nucs[:, i], i])))
    # epsilon_cell_ct.append(two_sig_fig_floor(torch.min(cell_pixel_ct[non_zero_cells[:, 0], 0])))
    # epsilon_nuc_ct.append(two_sig_fig_floor(torch.min(nucleus_pixel_ct[non_zero_nucs[:, 0], 0])))

    epsilon_int = torch.stack(epsilon_int, dim=0)
    epsilon_nuc_int = torch.stack(epsilon_nuc_int, dim=0)
    # epsilon_cell_ct = torch.stack(epsilon_cell_ct, dim=0)
    # epsilon_nuc_ct = torch.stack(epsilon_nuc_ct, dim=0)

    intensity_sums = intensity_sums + (epsilon_int * (~non_zero_cells))
    nuclear_intensity_sums = nuclear_intensity_sums + (epsilon_nuc_int * (~non_zero_nucs))
    # cell_pixel_ct = cell_pixel_ct + (epsilon_cell_ct * (~non_zero_cells))
    # nucleus_pixel_ct = nucleus_pixel_ct + (epsilon_nuc_ct * (~non_zero_nucs))

    return intensity_sums, nuclear_intensity_sums, cell_pixel_ct, nucleus_pixel_ct


def get_intensity_metrics_sc(images, batch_size=32):
    # Gather intensity and non-zero pixel counts
    intensity_sums, pixel_ct = [], []
    for i in tqdm(range(0, len(images), batch_size), desc="Computing mean intensities"):
        batch = images[i:min(i+batch_size, len(images))]
        intensity_sums.append(torch.sum(batch, dim=(2, 3)))
        pixel_ct.append(torch.sum(batch > 0, dim=(2, 3)))

    intensity_sums = torch.cat(intensity_sums, dim=0)
    pixel_ct = torch.cat(pixel_ct, dim=0)

    # calculate epsilon for each term
    non_zero_cells = intensity_sums > 0
    non_empty_cells = pixel_ct > 0
    epsilon_int, epsilon_ct = [], []
    for i in range(intensity_sums.shape[1]):
        epsilon_int.append(two_sig_fig_floor(torch.min(intensity_sums[non_zero_cells[:, i], i])))
        epsilon_ct.append(two_sig_fig_floor(torch.min(pixel_ct[non_empty_cells[:, i], i])))

    epsilon_int = torch.stack(epsilon_int, dim=0)
    epsilon_ct = torch.stack(epsilon_ct, dim=0)
    intensity_sums = intensity_sums + (epsilon_int * (~non_zero_cells))
    pixel_ct = pixel_ct + (epsilon_ct * (~non_empty_cells))

    return intensity_sums, pixel_ct


from cc_pseudotime import f_2
from scipy.optimize import least_squares
def intensities_to_polar_pseudotime(log_intensities, center=None):
    if center is None:
        center_estimate = np.mean(log_intensities, axis=0)
        center_est2 = least_squares(f_2, center_estimate, args=(log_intensities[:, 0], log_intensities[:, 1]))
        center = center_est2.x
    centered_intensities = log_intensities - center
    r = np.sqrt(np.sum(centered_intensities ** 2, axis=1))
    theta = np.arctan2(centered_intensities[:, 1], centered_intensities[:, 0])
    polar = np.stack([r, theta], axis=-1)
    fucci_time = calculate_pseudotime(polar.T, centered_intensities)
    return fucci_time

from cc_pseudotime import stretch_time
import matplotlib.pyplot as plt

def calculate_pseudotime(pol_data, centered_data, save_dir=""):
    pol_sort_inds = np.argsort(pol_data[1])
    pol_sort_rho = pol_data[0][pol_sort_inds]
    pol_sort_phi = pol_data[1][pol_sort_inds]
    centered_data_sort0 = centered_data[pol_sort_inds, 0]
    centered_data_sort1 = centered_data[pol_sort_inds, 1]

    # Rezero to minimum --resoning, cells disappear during mitosis, so we should have the fewest detected cells there
    bins = plt.hist(pol_sort_phi, 1000)
    plt.close()
    start_phi = bins[1][np.argmin(bins[0])]

    # Move those points to the other side
    more_than_start = np.greater(pol_sort_phi, start_phi)
    less_than_start = np.less_equal(pol_sort_phi, start_phi)
    pol_sort_rho_reorder = np.concatenate(
        (pol_sort_rho[more_than_start], pol_sort_rho[less_than_start])
    )
    pol_sort_inds_reorder = np.concatenate(
        (pol_sort_inds[more_than_start], pol_sort_inds[less_than_start])
    )
    pol_sort_phi_reorder = np.concatenate(
        (pol_sort_phi[more_than_start], pol_sort_phi[less_than_start] + np.pi * 2)
    )
    pol_sort_centered_data0 = np.concatenate(
        (centered_data_sort0[more_than_start], centered_data_sort0[less_than_start])
    )
    pol_sort_centered_data1 = np.concatenate(
        (centered_data_sort1[more_than_start], centered_data_sort1[less_than_start])
    )
    pol_sort_shift = pol_sort_phi_reorder + np.abs(np.min(pol_sort_phi_reorder))

    # Shift and re-scale "time"
    # reverse "time" since the cycle goes counter-clockwise wrt the fucci plot
    pol_sort_norm = pol_sort_shift / np.max(pol_sort_shift)
    pol_sort_norm_rev = 1 - pol_sort_norm
    pol_sort_norm_rev = stretch_time(pol_sort_norm_rev)
    pol_unsort = np.argsort(pol_sort_inds_reorder)
    fucci_time = pol_sort_norm_rev[pol_unsort]
    return fucci_time
