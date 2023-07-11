import asyncio
import glob
import json
import multiprocessing as mp
import os
import pickle
import warnings
from concurrent.futures import ThreadPoolExecutor
from time import monotonic

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde
from skimage.measure import regionprops
from tqdm import tqdm

from preprocess_masks import get_single_cell_mask

import pandas as pd
from .utils import get_overlap


def check_paths(df, image_path, mask_path):
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        plate_id = row["if_plate_id"]
        position = row["position"]
        sample = row["sample"]
        name_str = str(plate_id) + "_" + str(position) + "_" + str(sample)
        if (
            not os.path.exists(f"{image_path}/{plate_id}/{name_str}_red.png")
            or not os.path.exists(f"{image_path}/{plate_id}/{name_str}_yellow.png")
            or not os.path.exists(f"{image_path}/{plate_id}/{name_str}_blue.png")
            or not os.path.exists(f"{image_path}/{plate_id}/{name_str}_green.png")
            or not os.path.exists(f"{mask_path}/{name_str}_cellmask.png")
            or not os.path.exists(f"{mask_path}/{name_str}_nucleimask.png")
        ):
            df.drop(idx, inplace=True)
    return df.reset_index(drop=True)


def get_size_df_mp(df, unique_ids, num_cores):
    with joblib.Parallel(n_jobs=num_cores) as parallel:
        parallel(
            joblib.delayed(get_size_df)(df, unique_id)
            for unique_id in tqdm(unique_ids, total=len(unique_ids))
        )


def get_size_df(df, unique_id):
    df_temp = df[df.plate_position_sample == unique_id].reset_index(drop=True)

    plate_id = df_temp["if_plate_id"].values[0]
    position = df_temp["position"].values[0]
    sample = df_temp["sample"].values[0]
    name_str = str(plate_id) + "_" + str(position) + "_" + str(sample)

    cell_masks = cv2.imread(f"{mask_path}/{name_str}_cellmask.png", -1)
    nuc_masks = cv2.imread(f"{mask_path}/{name_str}_nucleimask.png", -1)

    # assert cell_masks is not None and nuc_masks is not None, f"{name_str} masks not found"

    cell_masks, nuc_masks, cell_bboxes, coms, bbox_labels = get_single_cell_mask(
        cell_masks, nuc_masks, IMG_SIZE, closing_radius=10, rm_border=True
    )
    cv2.imwrite(f"{mask_save_path}/{name_str}_cellmask.png", cell_masks)

    if cell_bboxes is not None:
        x1, y1, w, h = (
            cell_bboxes[:, 0],
            cell_bboxes[:, 1],
            cell_bboxes[:, 2],
            cell_bboxes[:, 3],
        )
        x_c, y_c = coms[:, 0].astype(np.int16), coms[:, 1].astype(np.int16)

        annotation_df = pd.DataFrame(
            {
                "unique_id": [unique_id] * len(x1),
                "plate_id": [plate_id] * len(x1),
                "position": [position] * len(x1),
                "sample": [sample] * len(x1),
                "mask_label": bbox_labels,
                "x1": x1,
                "y1": y1,
                "w": w,
                "h": h,
                "x_c": x_c,
                "y_c": y_c,
            }
        )

    else:
        annotation_df = pd.DataFrame(
            {
                "unique_id": [unique_id],
                "plate_id": [plate_id],
                "position": [position],
                "sample": [sample],
                "mask_label": bbox_labels,
                "x1": [np.nan],
                "y1": [np.nan],
                "w": [np.nan],
                "h": [np.nan],
                "x_c": [np.nan],
                "y_c": [np.nan],
            }
        )

    annotation_df.to_csv(
        f"annotations/bbox_annotations/IF-image-U2OS-bboxes-{unique_id}.csv"
    )

    # return annotation_df


#### get a unique column with combination of "if_plate_id" and "position"
def get_cell_bboxes():
    proper_done_files = []
    # done_files = glob.glob("annotations/bbox_annotations/IF-image-U2OS-bboxes-*.csv")
    # for file in done_files:
    #     try:
    #         pd.read_csv(file, index_col=0)
    #         # proper_done_files.append(file)
    #     except:
    #         print(file)
    # proper_done_files = [x.split("-")[-1].split(".")[0] for x in proper_done_files]
    df = pd.read_csv("annotations/IF-image-U2OS-filtered.csv", index_col=0)
    df["plate_position_sample"] = (
        df["if_plate_id"].astype(str)
        + "_"
        + df["position"].astype(str)
        + "_"
        + df["sample"].astype(str)
    )
    unique_plate_position_sample = df["plate_position_sample"].unique().tolist()
    unique_plate_position_sample = [
        x for x in unique_plate_position_sample if x not in proper_done_files
    ]
    # unique_plate_position_sample = sorted(unique_plate_position_sample)
    get_size_df_mp(df, unique_plate_position_sample, num_cores=10)


#### combine all the files
def filter_and_comb_bboxes():
    files = sorted(glob.glob("annotations/bbox_annotations/IF-image-U2OS-bboxes-*.csv"))
    done_files = []
    size_df = pd.DataFrame()
    for file in files:
        try:
            x = pd.read_csv(file, index_col=0)
            done_files.append(file)
            size_df = pd.concat([size_df, x])
        except:
            print(file)
    size_df = size_df.sort_values(by=["plate_id", "position", "sample"]).reset_index(
        drop=True
    )

    ## remove nan rows
    size_df = size_df[~size_df.isna().any(axis=1)]
    size_df.to_csv("annotations/IF-image-U2OS-bboxes.csv")


def get_plots(size_df):
    orig_bboxes = size_df[["x1", "y1", "w", "h"]].values
    orig_bboxes[:, 2] = orig_bboxes[:, 0] + orig_bboxes[:, 2]
    orig_bboxes[:, 3] = orig_bboxes[:, 1] + orig_bboxes[:, 3]

    center_bboxes = np.zeros_like(orig_bboxes)
    center_bboxes[:, 0] = ((size_df["x1"] + size_df["w"] / 2) - 256).values
    center_bboxes[:, 1] = ((size_df["y1"] + size_df["h"] / 2) - 256).values
    center_bboxes[:, 2] = center_bboxes[:, 0] + 512
    center_bboxes[:, 3] = center_bboxes[:, 1] + 512

    com_bboxes = np.zeros_like(orig_bboxes)
    com_bboxes[:, 0] = size_df["x_c"].values - 256
    com_bboxes[:, 1] = size_df["y_c"].values - 256
    com_bboxes[:, 2] = com_bboxes[:, 0] + 512
    com_bboxes[:, 3] = com_bboxes[:, 1] + 512

    centerbbox_gt_overlap, centerbbox_overlap = get_overlap(center_bboxes, orig_bboxes)
    combbox_gt_overlap, combbox_overlap = get_overlap(com_bboxes, orig_bboxes)

    plt.figure()
    plt.hist(
        centerbbox_gt_overlap,
        bins=10,
        alpha=0.5,
        label=f"center_gt_{centerbbox_gt_overlap.mean():.3f}",
    )
    plt.hist(
        combbox_gt_overlap,
        bins=10,
        alpha=0.5,
        label=f"com_gt_{combbox_gt_overlap.mean():.3f}",
    )
    plt.legend()
    plt.savefig("data_exploration/centerbbox_overlap_hist.png", dpi=300)
    plt.close()
    plt.figure()
    plt.hist(
        centerbbox_overlap,
        bins=10,
        alpha=0.5,
        label=f"center_{centerbbox_overlap.mean():.3f}",
    )
    plt.hist(
        combbox_overlap, bins=10, alpha=0.5, label=f"com_{combbox_overlap.mean():.3f}"
    )
    plt.legend()
    plt.savefig("data_exploration/combbox_overlap_hist.png", dpi=300)
    plt.close()

    ax = sns.histplot(
        x=size_df["w"],
        y=size_df["h"],
        bins=50,
        thresh=0.0,
        cbar=True,
        stat="probability",
    )
    ax.hlines(512, 0, 512, colors="r", linestyles="dashed")
    ax.vlines(512, 0, 512, colors="r", linestyles="dashed")
    plt.grid()
    plt.savefig("data_exploration/cell_size_hist.png", dpi=300)
    plt.close()

    ax = sns.histplot(
        x=size_df["w"],
        y=size_df["h"],
        bins=50,
        thresh=0.0,
        cbar=True,
        stat="probability",
        cumulative=True,
    )
    plt.grid()
    ax.hlines(512, 0, 512, colors="r", linestyles="dashed")
    ax.vlines(512, 0, 512, colors="r", linestyles="dashed")
    plt.savefig("data_exploration/cell_size_hist_cum.png", dpi=300)
    plt.close()

    kde_data = gaussian_kde([size_df["w"], size_df["h"]])
    all_cdfs = []
    for i in range(0, 1024, 16):
        all_cdfs.append(kde_data.integrate_box([0, 0], [i, i]))
    ax = sns.lineplot(x=range(0, 1024, 16), y=all_cdfs)
    ax.hlines(0.9, 0, 1024, colors="r", linestyles="dashed")
    ax.vlines(512, 0, 1, colors="r", linestyles="dashed")
    plt.grid()
    plt.savefig("data_exploration/cell_size_cdf.png", dpi=300)
    plt.close()

    sidelength = np.sqrt(size_df["w"] * size_df["h"]).values
    sns.histplot(sidelength, bins=50, stat="probability")
    ax.hlines(0.9, 0, 1024, colors="r", linestyles="dashed")
    ax.vlines(512 * np.sqrt(2), 0, 1, colors="r", linestyles="dashed")
    plt.savefig("data_exploration/cell_size_sidelength_hist.png", dpi=300)
    plt.grid()
    plt.close()

    ax = sns.histplot(sidelength, bins=50, stat="probability", cumulative=True)
    ax.hlines(0.9, 0, 1024, colors="r", linestyles="dashed")
    ax.vlines(512 * np.sqrt(2), 0, 1, colors="r", linestyles="dashed")
    plt.savefig("data_exploration/cell_size_sidelength_hist_cum.png", dpi=300)
    plt.grid()
    plt.close()


if __name__ == "__main__":
    image_path = "/data/HPA-IF-images"
    mask_path = "/data/kaggle-dataset/PUBLICHPA/mask/test"
    save_path = "/data/ankit"
    mask_save_path = "/data/ankit/processed_masks"
    display_save_path = "/data/ankit/display_images"
    processing_save_path = "/data/ankit/process_images"

    IMG_SIZE = (1024, 1024)
    colors = ["red", "yellow", "blue", "green"]

    # df = pd.read_csv("annotations/IF-image-U2OS.csv", index_col=0)
    # df = check_paths(df, image_path, mask_path)
    # df.to_csv("annotations/IF-image-U2OS-filtered.csv")

    get_cell_bboxes()
    filter_and_comb_bboxes()

    size_df = pd.read_csv("annotations/IF-image-U2OS-bboxes.csv", index_col=0)
    get_plots(size_df)
