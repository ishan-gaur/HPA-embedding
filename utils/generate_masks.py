import os
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import cv2
import hpacellseg.cellsegmentator as cellsegmentator
import matplotlib.pyplot as plt
import pandas as pd
from hpacellseg.utils import label_cell, label_nuclei
import numpy as np
from tqdm import tqdm


def get_masks(segmentator, image_path, mask_path, no_mask_df):
    for idx, row in tqdm(no_mask_df.iterrows(), total=len(no_mask_df)):
        plate_id = row["if_plate_id"]
        position = row["position"]
        sample = row["sample"]
        name_str = str(plate_id) + "_" + str(position) + "_" + str(sample)

        if not os.path.exists(f"{mask_path}/{name_str}_cellmask.png"):
            image_mt = cv2.imread(f"{image_path}/{plate_id}/{name_str}_red.png", -1)
            image_er = cv2.imread(f"{image_path}/{plate_id}/{name_str}_yellow.png", -1)
            image_nuc = cv2.imread(f"{image_path}/{plate_id}/{name_str}_blue.png", -1)

            if image_mt is None or image_er is None or image_nuc is None:
                print(f"Image not found, path:{name_str}")
                continue

            image = [[image_mt], [image_er], [image_nuc]]

            nuc_segmentation = segmentator.pred_nuclei(image[2])
            cell_segmentation = segmentator.pred_cells(image)

            # post-processing
            nuclei_mask = label_nuclei(nuc_segmentation[0])
            nuclei_mask, cell_mask = label_cell(
                nuc_segmentation[0], cell_segmentation[0]
            )

            # assert set(np.unique(nuclei_mask)) == set(
            #     np.unique(cell_mask)
            # ), f"Mask mismatch for {name_str}, nuclei: {np.unique(nuclei_mask)}, cell: {np.unique(cell_mask)}"
            assert (
                np.max(nuclei_mask) > 0 and np.max(cell_mask) > 0
            ), f"No nuclei or cell mask found for {name_str}"

            cv2.imwrite(f"{mask_path}/{name_str}_nucleimask.png", nuclei_mask)
            cv2.imwrite(f"{mask_path}/{name_str}_cellmask.png", cell_mask)


def get_mask_not_available_df(df):
    no_mask_df = pd.DataFrame(columns=df.columns)
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        plate_id = row["if_plate_id"]
        position = row["position"]
        sample = row["sample"]
        name_str = str(plate_id) + "_" + str(position) + "_" + str(sample)
        if not os.path.exists(f"{mask_path}/{name_str}_cellmask.png"):
            no_mask_df = no_mask_df.append(row).reset_index(drop=True)
            no_mask_df.to_csv("annotations/IF-image-U2OS_no_mask.csv")


NUC_MODEL = "./nuclei-model.pth"
CELL_MODEL = "./cell-model.pth"
segmentator = cellsegmentator.CellSegmentator(
    NUC_MODEL, CELL_MODEL, device="cuda", padding=True, multi_channel_model=True
)

image_path = "/data/HPA-IF-images"
# mask_path = "/data/kaggle-dataset/PUBLICHPA/mask/test"
mask_path = "/data/ankit/new_masks"

# no_mask_df = get_mask_not_available_df(df)


no_mask_df = pd.read_csv("annotations/IF-image-U2OS_no_mask.csv")
get_masks(segmentator, image_path, mask_path, no_mask_df)
