import pandas as pd
import cv2
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
import json

data_folder = "/data/HPA-IF-images"
# x = np.load(os.path.join(data_folder, "uniprot_sequence_deepgoplus_embeddings_8192.pkl"), allow_pickle=True)

df = pd.read_csv(os.path.join(data_folder, "IF-image.csv"))
df_U2OS = df[df.atlas_name == "U2OS"].reset_index(drop=True)
df_U2OS = df_U2OS[df_U2OS.latest_version == 23].reset_index(drop=True)
df_U2OS = df_U2OS[df_U2OS.ensembl_ids.notna()].reset_index(drop=True)
df_U2OS.to_csv("IF-image-U2OS.csv")


def get_image_names(mask_path, df):
    colors = ["red", "yellow", "blue", "green"]
    with open("image_path.txt", "w") as f:
        for row in df.itertuples():
            plate_id = row.if_plate_id
            position = row.position
            sample = row.sample
            name_str = str(plate_id) + "_" + str(position) + "_" + str(sample)

            for color in colors:
                f.write(f"HPA-IF-images/{plate_id}/{name_str}_{color}.png\n")

            if not os.path.exists(f"{mask_path}/{name_str}_cellmask.png"):
                print(f"cell mask not found, path:{name_str}")

            os.path.join(mask_path, name_str + "_cellmask.png")


# get_image_names(mask_path, df)
