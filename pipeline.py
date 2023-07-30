import os
import csv
import subprocess
from tqdm import tqdm
from glob import glob
from pathlib import Path 

import torch
import numpy as np

import cv2
from scipy import ndimage
from microfilm.microplot import microshow
import hpacellseg.cellsegmentator as cellsegmentator
from hpacellseg.utils import label_cell, label_nuclei
from skimage import measure, segmentation, morphology, transform

import matplotlib.pyplot as plt

def save_channel_names(data_dir, channel_names):
    with open(data_dir / "channel_names.txt", "w+") as f:
        f.write("\n".join(channel_names))

def load_channel_names(data_dir):
    with open(data_dir / "channel_names.txt", "r") as f:
        channel_names = f.read().splitlines()
    return channel_names

def create_image_paths_file(data_dir, exists_ok=True, overwrite=False):
    if type(data_dir) == Path:
        data_dir = str(data_dir)
    data_paths_file = "data-folder.txt"

    if os.path.exists(data_paths_file):
        if not exists_ok:
            raise Exception(f"Image path index already exists at: {data_paths_file}")
        if overwrite:
            print(f"Overwriting image path index at: {data_paths_file}")
            os.remove(data_paths_file)
        else:
            print(f"Image path index found at: {data_paths_file}")
    if not os.path.exists(data_paths_file) or overwrite:
        bash_create_index = f"find \"{data_dir}\" -type d > \"{data_paths_file}\""
        bash_remove_top_line = f"sed -i '1d' \"{data_paths_file}\""
        subprocess.run(bash_create_index, shell=True)
        subprocess.run(bash_remove_top_line, shell=True)
        print("Image path index created at:", data_paths_file)
    print("Number of target paths found:", end=" ")
    p = subprocess.run(f"cat {data_paths_file} | wc -l", shell=True, capture_output=True)
    num_paths = int(p.stdout.decode("utf-8").strip())
    return data_paths_file, num_paths

def image_paths_from_folders(folders_file):
    image_paths = list(open(folders_file, "r"))
    image_paths = [Path(x.strip()) for x in image_paths]
    return image_paths

def segmentator_setup(multi_channel_model, device):
    pwd = Path(os.getcwd())
    NUC_MODEL = pwd / "HPA-Cell-Segmentation" / "nuclei-model.pth"
    CELL_MODEL = pwd / "HPA-Cell-Segmentation" / "cell-model.pth"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    segmentator = cellsegmentator.CellSegmentator(
        str(NUC_MODEL), str(CELL_MODEL), device=device, padding=True, multi_channel_model=multi_channel_model
    )
    return segmentator

def segmentations_exist(image_path):
    if (image_path / "images.npy").exists():
        if (image_path / "cell_masks.npy").exists():
            if (image_path / "nuclei_masks.npy").exists():
                return True
    return False

def clean_segmentations(image_paths): 
    for image_path in image_paths:
            if (image_path / "images.npy").exists():
                os.remove(image_path / "images.npy") 
            if (image_path / "cell_masks.npy").exists():
                os.remove(image_path / "cell_masks.npy")
            if (image_path / "nuclei_masks.npy").exists():
                os.remove(image_path / "nuclei_masks.npy")

def merge_extraneous_elements(targets, elements):
    extraneous_elements = np.unique(elements[~np.isin(elements, targets)])
    extraneous_elements = extraneous_elements[extraneous_elements != 0] # remove background
    target_masks = np.stack([targets == target for target in np.unique(targets)])
    target_coms = np.array([ndimage.center_of_mass(mask) for mask in target_masks])
    for element in extraneous_elements:
        element_com = ndimage.center_of_mass(elements == element)
        target_distances = np.linalg.norm(target_coms - element_com, axis=1)
        closest_target = np.unique(targets)[np.argmin(target_distances)]
        elements[elements == element] = closest_target
    return elements

def need_rebuild(file, rebuild):
    if not file.exists():
        return True
    elif rebuild:
        os.remove(file)
        return True
    else:
        return False


def get_masks(segmentator, image_paths, channel_names, dapi, tubl, calb2, merge_missing=True, rebuild=False, display=False):
    """
    Get the masks for the images in image_paths using HPA-Cell-Segmentation
    """
    images_paths = []
    nuclei_mask_paths = []
    cell_mask_paths = []
    for image_path in tqdm(image_paths, desc="Getting masks"):
        if ((not need_rebuild(image_path / "images.npy", rebuild)) and
            (not need_rebuild(image_path / "cell_masks.npy", rebuild)) and
            (not need_rebuild(image_path / "nuclei_masks.npy", rebuild))):
            images_paths.append(image_path / "images.npy")
            nuclei_mask_paths.append(image_path / "nuclei_masks.npy")
            cell_mask_paths.append(image_path / "cell_masks.npy")
            continue

        glob_channel_images = lambda image_path, c: list(glob(f"{str(image_path)}/**/*{channel_names[c]}.png", recursive=True))
        dapi_paths = sorted(glob_channel_images(image_path, dapi))
        tubl_paths = sorted(glob_channel_images(image_path, tubl))
        calb2_paths = sorted(glob_channel_images(image_path, calb2)) if calb2 is not None else None

        if len(dapi_paths) == 0 or len(tubl_paths) == 0:
            print(f"Missing DAPI or TUBULIN image in {image_path}")
            print("\t", os.listdir(image_path))
            continue
        
        for dapi_path, tubl_path in zip(dapi_paths, tubl_paths):
            assert str(dapi_path).split(channel_names[dapi])[0] == str(tubl_path).split(channel_names[tubl])[0], f"File mismatch for {dapi_path} and {tubl_path}"
        if calb2 is not None and calb2_paths is not None:
            for dapi_path, anln_path in zip(dapi_paths, calb2_paths):
                assert str(dapi_path).split(channel_names[dapi])[0] == str(anln_path).split(channel_names[calb2])[0], f"File mismatch for {dapi_path} and {anln_path}"

        load_image = lambda path_list: [cv2.imread(str(x), cv2.IMREAD_UNCHANGED) for x in path_list]
        dapi_images = load_image(dapi_paths)
        tubl_images = load_image(tubl_paths)
        calb2_images = load_image(calb2_paths) if calb2_paths is not None else None

        ref_images = [tubl_images, calb2_images, dapi_images]
        nuc_segmentation = segmentator.pred_nuclei(ref_images[2])
        cell_segmentation = segmentator.pred_cells(ref_images)

        # post-processing
        nuclei_masks, cell_masks = [], []
        for i in range(len(ref_images[2])): # 2 because DAPI will always be present and we set the order manually when defining ref_images
            nuclei_mask, cell_mask = label_cell(
                nuc_segmentation[i], cell_segmentation[i]
            )
            nuclei_masks.append(nuclei_mask)
            cell_masks.append(cell_mask)
        nuclei_masks = np.stack(nuclei_masks, axis=0)
        cell_masks = np.stack(cell_masks, axis=0)

        # apply preprocessing mask if the user want to merge nuclei
        images = []
        for c, channel in enumerate(channel_names):
            channel_paths = sorted(glob_channel_images(image_path, c))
            channel_images = load_image(channel_paths)
            channel_images = np.stack(channel_images, axis=0)
            images.append(channel_images)
        images = np.stack(images, axis=1)

        for i in [0, -2, -1]:
            assert images.shape[i] == nuclei_masks.shape[i] == cell_masks.shape[i], f"Shape mismatch for images and masks in {image_path}, at index {i}, images has shape {images.shape}, nuclei_masks has shape {nuclei_masks.shape}, cell_masks has shape {cell_masks.shape}"

        image_idx = 0
        for image, nuclei_mask, cell_mask in zip(images, nuclei_masks, cell_masks):
            if set(np.unique(nuclei_mask)) != set(np.unique(cell_mask)): 
                print(f"Mask mismatch for {image_path}, nuclei: {np.unique(nuclei_masks)}, cell: {np.unique(cell_masks)}")
                if display:
                    microshow(image[(dapi, tubl),], cmaps=["pure_blue", "pure_red"], label_text=f"Image: {Path(image_path).name}[{image_idx}] ")

                # show cells without nuclei if any
                missing_nuclei = np.asarray(list(set(np.unique(cell_mask)) - set(np.unique(nuclei_mask))))
                if len(missing_nuclei) > 0 and display:
                    microshow(image[tubl] * np.isin(cell_mask, missing_nuclei), cmaps=["pure_red"], label_text=f"Cells missing nuclei: {missing_nuclei}")

                # show nuclei without cells if any
                missing_cells = np.asarray(list(set(np.unique(nuclei_mask)) - set(np.unique(cell_mask))))
                if len(missing_cells) > 0 and display:
                    microshow(image[dapi] * np.isin(nuclei_mask, missing_cells), cmaps=["pure_blue"], label_text=f"Nuclei without cells: {missing_cells}")

                # show cell masks and merge missing cells based on nuclei
                if display:
                    microshow(cell_mask, label_text=f"Cell mask: {np.unique(cell_mask)}")
                if len(missing_nuclei) > 0 and merge_missing:
                    cell_mask = merge_extraneous_elements(nuclei_mask, cell_mask)
                    if display:
                        microshow(cell_mask, label_text=f"Cell mask after merging: {np.unique(cell_mask)}")

                # show nuclei masks and merge missing nuclei based on cells
                microshow(nuclei_mask, label_text=f"Nuclei mask: {np.unique(nuclei_mask)}")
                if len(missing_cells) > 0 and merge_missing:
                    nuclei_mask = merge_extraneous_elements(cell_mask, nuclei_mask)
                    if display:
                        microshow(nuclei_mask, label_text=f"Nuclei mask after merging: {np.unique(nuclei_mask)}")

        assert np.max(nuclei_masks) > 0 and np.max(cell_masks) > 0, f"No nuclei or cell mask found for {image_path}"
        assert set(np.unique(nuclei_masks)) == set(np.unique(cell_masks)), f"Mask mismatch for {image_path}, nuclei: {np.unique(nuclei_masks)}, cell: {np.unique(cell_masks)}"

        np.save(image_path / "images.npy", images)
        np.save(image_path / "nuclei_masks.npy", nuclei_masks)
        np.save(image_path / "cell_masks.npy", cell_masks)

        images_paths.append(image_path / "images.npy")
        nuclei_mask_paths.append(image_path / "nuclei_masks.npy")
        cell_mask_paths.append(image_path / "cell_masks.npy")

    return images_paths, nuclei_mask_paths, cell_mask_paths

def clear_border(cell_mask, nuclei_mask):
    # inside clear_border they make borders by dimension. So if you have a 2D image, 
    # they make a 2D border, if you have a 3D image, they make a 3D border, which means the
    # first and last channels are fully borders. So we need to squeeze the image to 2D beforehand
    cell_mask, nuclei_mask = np.squeeze(cell_mask), np.squeeze(nuclei_mask)
    assert set(np.unique(nuclei_mask)) == set(np.unique(cell_mask)), f"Mask mismatch before clearing border, nuclei: {np.unique(nuclei_mask)}, cell: {np.unique(cell_mask)}"

    num_removed = 0
    cleared_nuclei_mask = segmentation.clear_border(nuclei_mask)
    keep_value = np.unique(cleared_nuclei_mask)
    bordering_cells = np.isin(cell_mask, keep_value)
    cleared_cell_mask = cell_mask * bordering_cells
    num_removed = len(np.unique(nuclei_mask)) - len(keep_value)
    if num_removed == np.max(nuclei_mask):
        assert np.max(keep_value) == 0, f"Something went wrong with clearing the border, num_removed is the same as the highest index mask in nuclei mask, but the keep_value {np.max(keep_value)} != 0"
    nuclei_mask = cleared_nuclei_mask
    cell_mask = cleared_cell_mask

    assert set(np.unique(nuclei_mask)) == set(np.unique(cell_mask)), f"Mask mismatch after clearing border, nuclei: {np.unique(nuclei_mask)}, cell: {np.unique(cell_mask)}"
    return cell_mask, nuclei_mask, num_removed

def remove_small_objects(cell_mask, nuclei_mask, min_size=1000):
    num_removed = 0

    cell_mask = morphology.remove_small_objects(cell_mask, min_size=min_size)
    num_removed = len(np.unique(nuclei_mask)) - len(np.unique(cell_mask))
    nuclei_mask = nuclei_mask * (cell_mask > 0)
    assert set(np.unique(nuclei_mask)) == set(np.unique(cell_mask)), f"Mask mismatch after removing small objects, nuclei: {np.unique(nuclei_mask)}, cell: {np.unique(cell_mask)}"

    nuclei_mask = morphology.remove_small_objects(nuclei_mask, min_size=min_size)
    num_removed += len(np.unique(cell_mask)) - len(np.unique(nuclei_mask))
    cell_mask = cell_mask * np.isin(cell_mask, np.unique(nuclei_mask))
    assert set(np.unique(nuclei_mask)) == set(np.unique(cell_mask)), f"Mask mismatch after removing small objects, nuclei: {np.unique(nuclei_mask)}, cell: {np.unique(cell_mask)}"

    return cell_mask, nuclei_mask, num_removed

def merge_nuclei(nuclei_mask, cell_mask, dialation_radius=20):
    bin_nuc_mask = (nuclei_mask > 0).astype(np.int8)
    cls_nuc = morphology.closing(bin_nuc_mask, morphology.disk(dialation_radius))
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

        # for old_label in old_labels:
        #     new_cell_mask[cell_mask == old_label] = new_label
        #     new_nuc_mask[nuclei_mask == old_label] = new_label
    return new_nuc_mask, new_cell_mask


def clean_cell_masks(
    cell_mask,
    nuclei_mask,
    remove_size=0, # remove cells smaller than remove_size, based on the area of the bounding box, honestly could be higher, mb 2500. Make 0 to turn off.
    dialation_radius=0, # this is for 2048x2048 images adjust as needed. Make 0 to turn off.
):
    num_removed = 0
    ### see if nuclei are touching and merge them
    if dialation_radius > 0:
        nuclei_mask, cell_mask = merge_nuclei(nuclei_mask, cell_mask, dialation_radius=dialation_radius)
        assert set(np.unique(nuclei_mask)) == set(np.unique(cell_mask))
    else:
        cell_mask = cell_mask
        nuclei_mask = nuclei_mask

    region_props = measure.regionprops(cell_mask, (cell_mask > 0).astype(np.uint8))
    pre_size = len(region_props)
    if remove_size > 0:
        region_props = [x for x in region_props if x.area > remove_size]
        num_removed += pre_size - len(region_props)
    bbox_array = np.array([x.bbox for x in region_props])

    # convert x1,y1,x2,y2 to x,y,w,h
    bbox_array[:, 2] = bbox_array[:, 2] - bbox_array[:, 0]
    bbox_array[:, 3] = bbox_array[:, 3] - bbox_array[:, 1]

    # com_array = np.array([x.weighted_centroid for x in region_props])

    return cell_mask, nuclei_mask, bbox_array, num_removed

def relabel_masks(cell_mask, nuclei_mask):
    # relabel the masks so that the labels are consecutive
    # this is necessary because the masks may have been modified
    # and may have missing labels
    new_nuclei_mask = np.zeros_like(nuclei_mask)
    new_cell_mask = np.zeros_like(cell_mask)
    for i, label in enumerate(np.unique(nuclei_mask)):
        new_nuclei_mask[nuclei_mask == label] = i
        new_cell_mask[cell_mask == label] = i
    return new_cell_mask, new_nuclei_mask

def clean_and_save_masks(
    cell_mask_paths,
    nuclei_mask_paths,
    rm_border=True, # removes cells with nuclei touching the border
    remove_size=2500, # remove cells smaller than remove_size, based on the area of the bounding box, honestly could be higher, mb 2500. Make 0 to turn off.
    # dialation_radius=0, # this is for 2048x2048 images adjust as needed. Make 0 to turn off.
):
    num_original = 0
    num_removed = 0
    for (cell_mask_path, nuclei_mask_path) in tqdm(list(zip(cell_mask_paths, nuclei_mask_paths)), desc="Cleaning masks"):
        cell_mask = np.load(cell_mask_path).squeeze()
        nuclei_mask = np.load(nuclei_mask_path).squeeze()
        assert set(np.unique(nuclei_mask)) == set(np.unique(cell_mask)), f"Mask mismatch before cleaning, nuclei: {np.unique(nuclei_mask)}, cell: {np.unique(cell_mask)}"
        num_original += len(np.unique(nuclei_mask))
        if rm_border:
            cell_mask, nuclei_mask, n_removed = clear_border(cell_mask, nuclei_mask)
            num_removed += n_removed
        if remove_size > 0:
            cell_mask, nuclei_mask, n_removed = remove_small_objects(cell_mask, nuclei_mask, min_size=remove_size)
            num_removed += n_removed
        assert set(np.unique(nuclei_mask)) == set(np.unique(cell_mask)), f"Mask mismatch after cleaning, nuclei: {np.unique(nuclei_mask)}, cell: {np.unique(cell_mask)}"
        cell_mask, nuclei_mask = relabel_masks(cell_mask, nuclei_mask)
        np.save(cell_mask_path, cell_mask)
        np.save(nuclei_mask_path, nuclei_mask)
    return num_original, num_removed


def crop_images(image_paths, cell_mask_paths, nuclei_mask_paths, crop_size, nuc_margin=50):
    # images need to be C x H x W
    seg_image_paths, seg_cell_mask_paths, seg_nuclei_mask_paths = [], [], []
    for (image_path, cell_mask_path, nuclei_mask_path) in tqdm(list(zip(image_paths, cell_mask_paths, nuclei_mask_paths)), desc="Cropping images"):
        image = np.load(image_path).squeeze()
        cell_masks = np.load(cell_mask_path).squeeze()
        nuclei_masks = np.load(nuclei_mask_path).squeeze()

        region_props = measure.regionprops(cell_masks)
        nuclear_regions = measure.regionprops(nuclei_masks)
        bboxes = np.array([x.bbox for x in region_props])
        bbox_widths = bboxes[:, 2] - bboxes[:, 0]
        bbox_heights = bboxes[:, 3] - bboxes[:, 1]
        nuc_bboxes = np.array([x.bbox for x in nuclear_regions])

        cell_mask_list = []
        nuclei_mask_list = []
        image_list = []
        for i, (bbox, width, height, nbox) in enumerate(zip(bboxes, bbox_widths, bbox_heights, nuc_bboxes)):
            center = np.array([np.mean([bbox[0], bbox[2]]), np.mean([bbox[1], bbox[3]])])
            if width > crop_size:
                left_edge = int(center[0] - crop_size / 2)
                right_edge = int(center[0] + crop_size / 2)
                if nbox[0] < left_edge and nbox[2] > right_edge:
                    print(f"nucleus {i} is wider than crop region, bbox: {bbox}, nbox: {nbox}, center: {center}, crop_size: {crop_size}")
                    continue
                if nbox[0] < left_edge:
                    new_left = nbox[0] - nuc_margin
                    displacement = new_left - left_edge
                    left_edge = new_left
                    right_edge = right_edge + displacement
                elif nbox[2] > right_edge:
                    new_right = nbox[2] + nuc_margin
                    displacement = new_right - right_edge
                    right_edge = new_right
                    left_edge = left_edge + displacement
                slice_x = slice(max(0, left_edge), min(cell_masks.shape[0], right_edge)) 
                padding_x = (max(0 - left_edge, 0), max(right_edge - cell_masks.shape[0], 0))
            else:
                slice_x = slice(bbox[0], bbox[2])
                padding_x = crop_size - width
                padding_x = (padding_x // 2, padding_x - padding_x // 2)
            if height > crop_size:
                top_edge = int(center[1] - crop_size / 2)
                bottom_edge = int(center[1] + crop_size / 2)
                if nbox[1] < top_edge and nbox[3] > bottom_edge:
                    print(f"nucleus {i} is taller than crop region, bbox: {bbox}, nbox: {nbox}, center: {center}, crop_size: {crop_size}")
                    continue
                if nbox[1] < top_edge:
                    new_top = nbox[1] - nuc_margin
                    displacement = top_edge - new_top
                    top_edge = new_top
                    bottom_edge = bottom_edge - displacement
                elif nbox[3] > bottom_edge:
                    new_bottom = nbox[3] + nuc_margin
                    displacement = new_bottom - bottom_edge
                    bottom_edge = new_bottom
                    top_edge = top_edge + displacement
                slice_y = slice(max(0, top_edge), min(cell_masks.shape[1], bottom_edge))
                padding_y = (max(0 - top_edge, 0), max(bottom_edge - cell_masks.shape[1], 0))
            else:
                slice_y = slice(bbox[1], bbox[3])
                padding_y = crop_size - height
                padding_y = (padding_y // 2, padding_y - padding_y // 2)

            cell_mask = ((cell_masks[slice_x, slice_y] * np.isin(cell_masks[slice_x, slice_y], i + 1)) > 0)
            nuclei_mask = ((nuclei_masks[slice_x, slice_y] * np.isin(nuclei_masks[slice_x, slice_y], i + 1)) > 0)
            cell_image = image[:, slice_x, slice_y] * cell_mask
            
            cell_mask = np.pad(cell_mask, (padding_x, padding_y), mode="constant", constant_values=0)
            nuclei_mask = np.pad(nuclei_mask, (padding_x, padding_y), mode="constant", constant_values=0)
            cell_image = np.pad(cell_image, ((0, 0), padding_x, padding_y), mode="constant", constant_values=0)

            cell_mask_list.append(cell_mask)
            nuclei_mask_list.append(nuclei_mask)
            image_list.append(cell_image)

        cell_masks = np.stack(cell_mask_list, axis=0)
        nuclei_masks = np.stack(nuclei_mask_list, axis=0)
        images = np.stack(image_list, axis=0)

        seg_cell_mask_paths.append(cell_mask_path.parent /"seg_cell_masks.npy")
        seg_nuclei_mask_paths.append(nuclei_mask_path.parent / "seg_nuclei_masks.npy")
        seg_image_paths.append(image_path.parent / "seg_images.npy")
        np.save(seg_cell_mask_paths[-1], cell_masks)
        np.save(seg_nuclei_mask_paths[-1], nuclei_masks)
        np.save(seg_image_paths[-1], images)
    return seg_image_paths, seg_cell_mask_paths, seg_nuclei_mask_paths

channel_min = lambda x: torch.min(torch.min(x, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]
channel_max = lambda x: torch.max(torch.max(x, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]

def resize_and_normalize(seg_image_paths, seg_cell_mask_paths, seg_nuclei_mask_paths, target_dim, normalize=True, resize_type=cv2.INTER_LANCZOS4):
    target_dim = (target_dim, target_dim)
    final_image_paths, final_cell_mask_paths, final_nuclei_mask_paths = [], [], []
    for (seg_image_path, seg_cell_mask_path, seg_nuclei_mask_path) in tqdm(list(zip(seg_image_paths, seg_cell_mask_paths, seg_nuclei_mask_paths)), desc="Resizing images"):
        images = np.load(seg_image_path).squeeze() # B x C x H x W
        images = np.transpose(images, (0, 2, 3, 1)) # B x H x W x C
        cell_masks = np.load(seg_cell_mask_path).squeeze() # B x H x W
        nuclei_masks = np.load(seg_nuclei_mask_path).squeeze() # B x H x W
        
        resized_images, resized_cell_masks, resized_nuclei_masks = [], [], []
        for image, cell_mask, nuclei_mask in zip(images, cell_masks, nuclei_masks):
            resized_images.append(cv2.resize(image, dsize=target_dim, interpolation=resize_type))
            assert np.max(np.unique(cell_mask)) <= 1, f"Cell mask has more than 1 unique value, {np.unique(cell_mask)}"
            resized_cell_masks.append(cv2.resize(cell_mask.astype("float32"), dsize=target_dim, interpolation=cv2.INTER_NEAREST))
            assert np.max(np.unique(nuclei_mask)) <= 1, f"Nuclei mask has more than 1 unique value, {np.unique(nuclei_mask)}"
            resized_nuclei_masks.append(cv2.resize(nuclei_mask.astype("float32"), dsize=target_dim, interpolation=cv2.INTER_NEAREST))

        resized_images = np.stack(resized_images, axis=0)
        resized_images = np.transpose(resized_images, (0, 3, 1, 2)) # B x C x H x W
        resized_images = torch.Tensor(resized_images.astype("float32"))
        # normalized_images = (resized_images - torch.min(resized_images, )) / (torch.max(resized_images) - torch.min(resized_images))
        if normalize:
            normalized_images = (resized_images - channel_min(resized_images)) / (channel_max(resized_images) - channel_min(resized_images))
        else:
            normalized_images = resized_images

        torch.save(normalized_images, seg_image_path.parent / "images.pt")
        final_image_paths.append(seg_image_path.parent / "images.pt")
        torch.save(torch.Tensor(resized_cell_masks), seg_cell_mask_path.parent / "cell_masks.pt")
        final_cell_mask_paths.append(seg_cell_mask_path.parent / "cell_masks.pt")
        torch.save(torch.Tensor(resized_nuclei_masks), seg_nuclei_mask_path.parent / "nuclei_masks.pt")
        final_nuclei_mask_paths.append(seg_nuclei_mask_path.parent / "nuclei_masks.pt")

    return final_image_paths, final_cell_mask_paths, final_nuclei_mask_paths

def create_data_path_index(image_paths, cell_mask_paths, nuclei_mask_paths, index_file, overwrite=False):
    if os.path.exists(index_file) and not overwrite:
        print("Index file already exists, not overwriting")
        return

    sample_paths = []
    for image_path, cell_mask_path, nuclei_mask_path in zip(image_paths, cell_mask_paths, nuclei_mask_paths):
        sample_paths.append({
            "sample_name": image_path.parent,
            "image_path": str(image_path),
            "cell_mask_path": str(cell_mask_path),
            "nuclei_mask_path": str(nuclei_mask_path)
        })

    with open(index_file, "w") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_name", "image_path", "cell_mask_path", "nuclei_mask_path"])
        writer.writeheader()
        for sample_path in sample_paths:
            writer.writerow(sample_path)

def load_data_path_index(data_dir):
    index_file = data_dir / "index.csv"
    if not index_file.exists():
        raise ValueError(f"Index file {index_file} does not exist")
    with open(index_file, "r") as f:
        reader = csv.DictReader(f)
        sample_paths = []
        for row in reader:
            sample_paths.append(row)
    return sample_paths

def load_dir_images(data_dir):
    sample_paths = load_data_path_index(data_dir)
    images = []
    for sample_path in tqdm(sample_paths, desc="Loading images"):
        images.append(torch.load(Path(sample_path["image_path"])))
    return images