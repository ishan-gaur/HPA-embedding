import numpy as np


def all_equal(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == x for x in iterator)


def standardize_image(image, percentile=99):
    min = np.percentile(image, 100 - percentile)
    max = np.percentile(image, percentile)
    image = (image - min) / (max - min)
    return np.clip(image, 0, 1)


def safe_crop(image, bbox):
    x1, y1, x2, y2 = bbox
    img_w, img_h = image.shape[:2]
    is_single_channel = len(image.shape) == 2
    if x1 < 0:
        pad_x1 = 0 - x1
        new_x1 = 0
    else:
        pad_x1 = 0
        new_x1 = x1
    if y1 < 0:
        pad_y1 = 0 - y1
        new_y1 = 0
    else:
        pad_y1 = 0
        new_y1 = y1
    if x2 > img_w - 1:
        pad_x2 = x2 - (img_w - 1)
        new_x2 = img_w - 1
    else:
        pad_x2 = 0
        new_x2 = x2
    if y2 > img_h - 1:
        pad_y2 = y2 - (img_h - 1)
        new_y2 = img_h - 1
    else:
        pad_y2 = 0
        new_y2 = y2

    patch = image[new_x1:new_x2, new_y1:new_y2]
    patch = (
        np.pad(
            patch,
            ((pad_x1, pad_x2), (pad_y1, pad_y2)),
            mode="constant",
            constant_values=0,
        )
        if is_single_channel
        else np.pad(
            patch,
            ((pad_x1, pad_x2), (pad_y1, pad_y2), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    )
    return patch


def get_stats(well_images):
    well_mean = np.mean(
        well_images, axis=(0, 1, 2) if len(well_images.shape) == 4 else (0, 1)
    )
    well_std = np.std(
        well_images, axis=(0, 1, 2) if len(well_images.shape) == 4 else (0, 1)
    )
    well_min_1 = np.percentile(
        well_images, 1, axis=(0, 1, 2) if len(well_images.shape) == 4 else (0, 1)
    )
    well_min_0 = np.percentile(
        well_images, 0, axis=(0, 1, 2) if len(well_images.shape) == 4 else (0, 1)
    )
    well_max_99 = np.percentile(
        well_images, 99, axis=(0, 1, 2) if len(well_images.shape) == 4 else (0, 1)
    )
    well_max_100 = np.percentile(
        well_images, 100, axis=(0, 1, 2) if len(well_images.shape) == 4 else (0, 1)
    )

    return well_mean, well_std, well_min_1, well_max_99, well_min_0, well_max_100


def get_overlap(bboxes_x, bboxes_gt):
    # determine the coordinates of the intersection rectangle
    x_left = np.max([bboxes_x[:, 0], bboxes_gt[:, 0]], axis=0)
    y_top = np.max([bboxes_x[:, 1], bboxes_gt[:, 1]], axis=0)
    x_right = np.min([bboxes_x[:, 2], bboxes_gt[:, 2]], axis=0)
    y_bottom = np.min([bboxes_x[:, 3], bboxes_gt[:, 3]], axis=0)

    intersection = (x_right - x_left) * (y_bottom - y_top)
    # intersection[intersection < 0] = 0

    bbox_gt_area = (bboxes_gt[:, 2] - bboxes_gt[:, 0]) * (
        bboxes_gt[:, 3] - bboxes_gt[:, 1]
    )
    bbox_x_area = (bboxes_x[:, 2] - bboxes_x[:, 0]) * (bboxes_x[:, 3] - bboxes_x[:, 1])

    overlap_gt = intersection / bbox_gt_area
    overlap_x = intersection / bbox_x_area
    return overlap_gt, overlap_x
