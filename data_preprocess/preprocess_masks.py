import numpy as np
from skimage import measure, segmentation, morphology
import cv2


def get_single_cell_mask(
    cell_mask,
    nuclei_mask,
    final_size=None,
    closing_radius=7,
    rm_border=True,
    remove_size=1000,
):
    if rm_border:
        nuclei_mask = segmentation.clear_border(nuclei_mask)
        keep_value = np.unique(nuclei_mask)
        borderedcellmask = np.array(
            [[x_ in keep_value for x_ in x] for x in cell_mask]
        ).astype("uint8")
        cell_mask = cell_mask * borderedcellmask

    if final_size is not None:
        nuclei_mask = cv2.resize(
            nuclei_mask, final_size, interpolation=cv2.INTER_NEAREST
        )
        cell_mask = cv2.resize(cell_mask, final_size, interpolation=cv2.INTER_NEAREST)

    ### see if nuclei are touching and merge them
    bin_nuc_mask = nuclei_mask > 0
    cls_nuc = morphology.closing(bin_nuc_mask, morphology.disk(closing_radius))
    # get the labels of touching nuclei
    new_label_map = morphology.label(cls_nuc)
    new_label_idx = np.unique(new_label_map)[1:]

    new_cell_mask = np.zeros_like(cell_mask)
    new_nuc_mask = np.zeros_like(nuclei_mask)
    for new_label in new_label_idx:
        # get the label of the touching nuclei
        old_labels = np.unique(nuclei_mask[new_label_map == new_label])
        old_labels = old_labels[old_labels != 0]

        new_nuc_mask[np.isin(nuclei_mask, old_labels)] = new_label
        new_cell_mask[np.isin(cell_mask, old_labels)] = new_label

    # assert set(np.unique(new_nuc_mask)) == set(np.unique(new_cell_mask))

    region_props_cell = measure.regionprops(
        new_cell_mask, intensity_image=(new_cell_mask > 0).astype(np.uint8)
    )
    region_props_nuc = measure.regionprops(
        new_nuc_mask, intensity_image=(new_nuc_mask > 0).astype(np.uint8)
    )

    region_props = [
        region_props_cell[i]
        for (i, x) in enumerate(region_props_nuc)
        if x.area > remove_size
    ]
    if len(region_props) == 0:
        return new_cell_mask, new_nuc_mask, None, None, None
    else:
        bbox_array = np.array([x.bbox for x in region_props])
        ## convert x1,y1,x2,y2 to x,y,w,h
        bbox_array[:, 2] = bbox_array[:, 2] - bbox_array[:, 0]
        bbox_array[:, 3] = bbox_array[:, 3] - bbox_array[:, 1]

        com_array = np.array([x.weighted_centroid for x in region_props])
        bbox_label = np.array([x.label for x in region_props])
        return new_cell_mask, new_nuc_mask, bbox_array, com_array, bbox_label
