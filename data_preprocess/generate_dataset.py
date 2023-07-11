import glob

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from .utils import all_equal, safe_crop


def get_annotation_df_mp(df, bbox_df, unique_ids, num_cores):
    with joblib.Parallel(n_jobs=num_cores) as parallel:
        parallel(
            joblib.delayed(get_annotation_df)(df, bbox_df, unique_id)
            for unique_id in tqdm(unique_ids, total=len(unique_ids))
        )


def get_annotation_df(df, bbox_df, unique_id):
    df_temp = df[df.unique_plate_position == unique_id].reset_index(drop=True)
    bbox_df_temp = bbox_df[bbox_df.unique_plate_position == unique_id].reset_index(
        drop=True
    )

    ensemble_id = df_temp["ensembl_ids"].to_list()
    assert all_equal(ensemble_id)

    well_bboxes = []
    well_mask_bboxes = []
    well_images = []
    for idx, row in df_temp.iterrows():
        plate_id = row["if_plate_id"]
        position = row["position"]
        sample = row["sample"]
        name_str = str(plate_id) + "_" + str(position) + "_" + str(sample)

        image_mt = cv2.imread(f"{image_path}/{plate_id}/{name_str}_red.png", -1)
        image_er = cv2.imread(f"{image_path}/{plate_id}/{name_str}_yellow.png", -1)
        image_nuc = cv2.imread(f"{image_path}/{plate_id}/{name_str}_blue.png", -1)
        image_pr = cv2.imread(f"{image_path}/{plate_id}/{name_str}_green.png", -1)

        image = cv2.resize(
            np.dstack((image_mt, image_er, image_nuc, image_pr)),
            IMG_SIZE,
            cv2.INTER_LINEAR,
        )
        well_images.append(image)

        image_cell_mask = cv2.imread(f"{mask_path}/{name_str}_cellmask.png", -1)

        bbox_df_temp_site = bbox_df_temp[bbox_df_temp.unique_id == name_str]
        for bbox_idx, bbox_row in bbox_df_temp_site.iterrows():
            x1 = int(bbox_row["x1"] + (bbox_row["w"] / 2) - (CROP_SIZE[0] / 2))
            y1 = int(bbox_row["y1"] + (bbox_row["h"] / 2) - (CROP_SIZE[1] / 2))
            x2 = int(x1 + CROP_SIZE[0])
            y2 = int(y1 + CROP_SIZE[1])

            cell_image = safe_crop(image, [x1, y1, x2, y2])
            cell_image = cv2.resize(cell_image, CELL_SIZE, cv2.INTER_LINEAR)
            well_bboxes.append(cell_image)

            cell_mask = safe_crop(image_cell_mask, [x1, y1, x2, y2])
            cell_mask = cell_mask == bbox_row["mask_label"]
            cell_mask = cell_mask.astype(np.uint8) * 255
            cell_mask = cv2.resize(cell_mask, CELL_SIZE, cv2.INTER_NEAREST)
            well_mask_bboxes.append(cell_mask)

            cv2.imwrite(
                f"{display_save_path}/{name_str}_{bbox_idx+1}_cellimage.png",
                cv2.cvtColor(cell_image[:, :, :3], cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                f"{display_save_path}/{name_str}_{bbox_idx+1}_protienimage.png",
                cell_image[:, :, 3],
            )
            cv2.imwrite(
                f"{display_save_path}/{name_str}_{bbox_idx+1}_cellmask.png",
                cell_mask,
            )

    well_bboxes = np.array(well_bboxes)
    well_mask_bboxes = np.array(well_mask_bboxes)
    well_images = np.array(well_images)

    np.save(f"{processing_save_path}/{unique_id}_cells.npy", well_bboxes[:, :, :, :3])
    np.save(f"{processing_save_path}/{unique_id}_proteins.npy", well_bboxes[:, :, :, 3])
    np.save(f"{processing_save_path}/{unique_id}_cellmask.npy", well_mask_bboxes)

    image_mean = np.mean(
        well_images, axis=(1, 2) if len(well_images.shape) == 4 else (0, 1)
    )
    image_std = np.std(
        well_images, axis=(1, 2) if len(well_images.shape) == 4 else (0, 1)
    )
    image_min_1 = np.percentile(
        well_images, 1, axis=(1, 2) if len(well_images.shape) == 4 else (0, 1)
    )
    image_max_99 = np.percentile(
        well_images, 99, axis=(1, 2) if len(well_images.shape) == 4 else (0, 1)
    )
    image_min = np.min(
        well_images, axis=(1, 2) if len(well_images.shape) == 4 else (0, 1)
    )
    image_max = np.max(
        well_images, axis=(1, 2) if len(well_images.shape) == 4 else (0, 1)
    )
    for color_idx, color in enumerate(colors):
        df_temp[f"mean_{color}"] = image_mean[:, color_idx]
        df_temp[f"std_{color}"] = image_std[:, color_idx]
        df_temp[f"min_1_{color}"] = image_min_1[:, color_idx]
        df_temp[f"max_99_{color}"] = image_max_99[:, color_idx]
        df_temp[f"min_{color}"] = image_min[:, color_idx]
        df_temp[f"max_{color}"] = image_max[:, color_idx]
    df_temp["cell_path"] = f"{processing_save_path}/{unique_id}_cellimage.npy"
    df_temp["prot_path"] = f"{processing_save_path}/{unique_id}_proteinimage.npy"
    df_temp["n_cells"] = len(well_bboxes)
    df_temp.to_csv(f"annotations/image_annotations/{unique_id}_image.csv", index=False)

    mean = np.mean(well_bboxes, axis=(1, 2) if len(well_bboxes.shape) == 4 else (0, 1))
    std = np.std(well_bboxes, axis=(1, 2) if len(well_bboxes.shape) == 4 else (0, 1))
    min_1 = np.percentile(
        well_bboxes, 1, axis=(1, 2) if len(well_bboxes.shape) == 4 else (0, 1)
    )
    max_99 = np.percentile(
        well_bboxes, 99, axis=(1, 2) if len(well_bboxes.shape) == 4 else (0, 1)
    )
    min = np.min(well_bboxes, axis=(1, 2) if len(well_bboxes.shape) == 4 else (0, 1))
    max = np.max(well_bboxes, axis=(1, 2) if len(well_bboxes.shape) == 4 else (0, 1))
    for color_idx, color in enumerate(colors):
        bbox_df_temp[f"mean_{color}"] = mean[:, color_idx]
        bbox_df_temp[f"std_{color}"] = std[:, color_idx]
        bbox_df_temp[f"min_1_{color}"] = min_1[:, color_idx]
        bbox_df_temp[f"max_99_{color}"] = max_99[:, color_idx]
        bbox_df_temp[f"min_{color}"] = min[:, color_idx]
        bbox_df_temp[f"max_{color}"] = max[:, color_idx]
    bbox_df_temp.to_csv(
        f"annotations/image_annotations/{unique_id}_stats.csv", index=False
    )


def get_cell_images(get_annotation_df_mp, df, bbox_df):
    df["unique_plate_position"] = (
        df["if_plate_id"].astype(str) + "_" + df["position"].astype(str)
    )
    bbox_df["unique_plate_position"] = (
        bbox_df["plate_id"].astype(str) + "_" + bbox_df["position"].astype(str)
    )

    df = df[df["unique_plate_position"].isin(bbox_df["unique_plate_position"].unique())]
    bbox_df = bbox_df[
        bbox_df["unique_plate_position"].isin(df["unique_plate_position"])
    ]

    assert set(bbox_df["unique_plate_position"].unique()) == set(
        df["unique_plate_position"].unique()
    ), "bbox_df and df should have same unique_ids"

    unique_plate_id_position = df["unique_plate_position"].unique().tolist()
    get_annotation_df_mp(df, bbox_df, unique_plate_id_position, num_cores=10)


def plot_stats(colors, stats_df, level):
    plt.figure()
    for color in colors:
        sns.histplot(
            x=stats_df[f"mean_{color}"] / stats_df[f"max_overall"],
            bins=50,
            stat="probability",
            label=f"mean_{color}/max_{color}",
            color=color,
            alpha=0.5,
        )
    plt.legend()
    plt.savefig(f"data_exploration/{level}_mean_max.png")
    plt.close()

    plt.figure()
    for color in colors:
        sns.histplot(
            x=stats_df[f"max_99_{color}"] / stats_df[f"max_overall"],
            bins=50,
            stat="probability",
            label=f"max_99_{color}/max_{color}",
            color=color,
            alpha=0.5,
        )
    plt.legend()
    plt.savefig(f"data_exploration/{level}_max99_max.png")
    plt.close()


def analyse_protein_intensity(imgstats_df):
    green_df = imgstats_df[
        [x for x in imgstats_df.columns if "green" in x]
        + ["if_plate_id", "position", "locations"]
    ]
    green_df = green_df[~green_df["locations"].isna()]
    green_df = green_df.sort_values("mean_green")
    g_low, g_low_cnt = np.unique(green_df["locations"][:2000], return_counts=True)
    g_high, g_high_cnt = np.unique(green_df["locations"][-2000:], return_counts=True)

    g_low = [x.split(",") if "," in x else x for x in g_low]
    g_low_fin = []
    for x in g_low:
        g_low_fin.extend(x) if type(x) == list else g_low_fin.append(x)
    top5low = np.unique(g_low_fin, return_counts=True)[0][
        np.argsort(np.unique(g_low_fin, return_counts=True)[1])[-5:]
    ]

    g_high = [x.split(",") if "," in x else x for x in g_high]
    g_high_fin = []
    for x in g_high:
        g_high_fin.extend(x) if type(x) == list else g_high_fin.append(x)
    top5high = np.unique(g_high_fin, return_counts=True)[0][
        np.argsort(np.unique(g_high_fin, return_counts=True)[1])[-5:]
    ]

    plt.figure()
    plt.bar(g_low, g_low_cnt, label="low green")
    plt.bar(g_high, g_high_cnt, label="high green")


if __name__ == "__main__":
    image_path = "/data/HPA-IF-images"
    mask_path = "/data/ankit/processed_masks"
    save_path = "/data/ankit"
    display_save_path = "/data/ankit/display_images"
    processing_save_path = "/data/ankit/process_images"

    IMG_SIZE = (1024, 1024)
    CROP_SIZE = (512, 512)
    CELL_SIZE = (256, 256)
    colors = ["red", "yellow", "blue", "green"]

    # df = pd.read_csv("annotations/IF-image-U2OS-filtered.csv", index_col=0)
    # bbox_df = pd.read_csv("annotations/IF-image-U2OS-bboxes.csv", index_col=0)
    # get_cell_images(get_annotation_df_mp, df, bbox_df)

    imgstats_list = glob.glob("annotations/image_annotations/*_image.csv")
    imgstats_df = pd.concat(
        [pd.read_csv(stats_file) for stats_file in imgstats_list]
    ).reset_index(drop=True)
    imgstats_df["max_overall"] = np.max(
        imgstats_df[["max_red", "max_yellow", "max_blue", "max_green"]], axis=1
    )
    imgstats_df["max_overall"] = imgstats_df["max_overall"].apply(
        lambda x: 255 if x <= 255 else 65535
    )
    plot_stats(colors, imgstats_df, "image")

    # analyse_protein_intensity(imgstats_df)

    cellstats_list = glob.glob("annotations/image_annotations/*_stats.csv")
    cellstats_df = pd.concat(
        [pd.read_csv(stats_file) for stats_file in cellstats_list]
    ).reset_index(drop=True)
    cellstats_df["max_overall"] = np.max(
        cellstats_df[["max_red", "max_yellow", "max_blue", "max_green"]], axis=1
    )
    cellstats_df["max_overall"] = cellstats_df["max_overall"].apply(
        lambda x: 255 if x <= 255 else 65535
    )
    plot_stats(colors, cellstats_df, "cell")

    ## save image stats
    save_cols = [
        "if_plate_id",
        "position",
        "locations",
        "ensembl_ids",
        "cell_path",
        "prot_path",
        "n_cells",
    ]
    stats_cols = (
        [f"mean_{color}" for color in colors]
        + [f"std_{color}" for color in colors]
        + [f"min_1_{color}" for color in colors]
        + [f"max_99_{color}" for color in colors]
        + [f"min_{color}" for color in colors]
        + [f"max_{color}" for color in colors]
    )
    metadata_df = imgstats_df[save_cols].drop_duplicates()
    stats_save_df = imgstats_df[["if_plate_id", "position"] + stats_cols]
    stats_save_df = (
        stats_save_df.groupby(["if_plate_id", "position"]).mean().reset_index()
    )
    save_df = (
        metadata_df.merge(stats_save_df, on=["if_plate_id", "position"], how="left")
        .sort_values(["if_plate_id", "position"])
        .reset_index(drop=True)
    )
    save_df.to_csv("annotations/IF-image-U2OS-image-data.csv")

    plt.figure()
    sns.histplot(x="n_cells", bins=50, stat="probability", data=save_df)
    plt.savefig("data_exploration/n_cells_hist.png")
    plt.close()
    plt.figure()
    sns.histplot(
        x="n_cells",
        hue="locations",
        bins=50,
        stat="probability",
        data=save_df,
        legend=False,
    )
    plt.savefig("data_exploration/n_cells_loc_hist.png")
    plt.close()
    ## get nans in locations
    loc_nans = save_df[save_df["locations"].isna()]
    loc_nans.to_csv("annotations/IF-image-U2OS-image-data-loc-nans.csv")

    ## save cell stats
    cellstats_df = cellstats_df.rename(columns={"plate_id": "if_plate_id"})
    cellstats_df = (
        cellstats_df.merge(
            imgstats_df[
                ["if_plate_id", "position", "locations", "sample", "ensembl_ids"]
            ],
            on=["if_plate_id", "position", "sample"],
        )
        .sort_values(["if_plate_id", "position"])
        .reset_index(drop=True)
    )
    cellstats_df.to_csv("annotations/IF-image-U2OS-cell-data.csv")
