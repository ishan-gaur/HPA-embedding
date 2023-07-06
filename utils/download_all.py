import requests
import getpass
from requests.auth import HTTPBasicAuth
import pandas as pd
import gzip
import io
import numpy as np
import imageio
import os
from multiprocessing.pool import Pool
from tqdm import tqdm
import pandas as pd

base_url = "https://if.proteinatlas.org"
colors = ["blue", "red", "green", "yellow"]
data_dir = "/data/HPA-IF-images/"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)


def download_images(data_dir, img_list, downloaded_images, pid, sp, ep):
    log = open("./log-" + pid + ".txt", "w")
    print(img_list.shape, img_list.head())
    for filename in tqdm(
        img_list["filename"][sp:ep], postfix=pid
    ):  # df.head().itertuples():
        if not filename or filename in downloaded_images:
            continue
        try:
            file_path = filename.replace("/archive/", "")
            subfolder = file_path.split("/")[0]
            name = file_path.split("/")[1]
            if not os.path.exists(os.path.join(data_dir, subfolder)):
                os.makedirs(os.path.join(data_dir, subfolder))
            for color in colors:
                print(name, color)
                fname = f"{name}{color}.tif.gz"
                url = f"{base_url}/{subfolder}/{fname}"
                target_dir = os.path.join(data_dir, subfolder)
                target_path = os.path.join(target_dir, f"{name}{color}.png")
                if os.path.exists(target_path):
                    print(target_path)
                    if os.path.getsize(target_path) > 100000:
                        continue
                    try:
                        img = imageio.imread(target_path)
                        assert img.ndim == 2 or img.ndim == 3
                    except:
                        print(
                            "file broken: ",
                            target_path,
                            ", will try to download again.",
                        )
                    else:
                        continue
                print(f"Downloading {url}")
                r = requests.get(url, auth=HTTPBasicAuth("trang", password))
                if r.status_code != 200:
                    fname = f"{name}{color}.tif"
                    url = f"{base_url}/{subfolder}/{fname}"
                    r = requests.get(url, auth=HTTPBasicAuth("trang", password))
                    tf = r.content
                    print(r.status_code, url)
                else:
                    f = io.BytesIO(r.content)
                    tf = gzip.open(f).read()
                img = imageio.imread(tf, "tiff")
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                imageio.imwrite(target_path, img)
                # img2 = imageio.imread(target_path)
                # assert(np.all(np.equal(img2, img)))
        except:
            log.write("failed to download: " + fname + "\n")
    log.close()


def run_proc(data_dir, img_list, downloaded_images, name, sp, ep):
    print("Run child process %s (%s) sp:%d ep: %d" % (name, os.getpid(), sp, ep))
    download_images(data_dir, img_list, downloaded_images, name, sp, ep)
    print("Run child process %s done" % (name))


def start_download(data_dir, img_list, downloaded_images, process_num=10):
    os.makedirs(data_dir, exist_ok=True)
    print("Parent process %s." % os.getpid())
    list_len = len(img_list["filename"])
    print(img_list.head())
    p = Pool(process_num)
    for i in range(process_num):
        p.apply_async(
            run_proc,
            args=(
                data_dir,
                img_list,
                downloaded_images,
                str(i),
                int(i * list_len / process_num),
                int((i + 1) * list_len / process_num),
            ),
        )
    print("Waiting for all subprocesses done...")
    p.close()
    p.join()
    print("All subprocesses done.")


if __name__ == "__main__":
    password = getpass.getpass("What is the your password?")
    base_url = "https://if.proteinatlas.org"
    colors = ["blue", "red", "green", "yellow"]
    data_dir = "/data/HPA-IF-images/"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    r = requests.get(base_url + "/IF-image.csv", auth=HTTPBasicAuth("trang", password))
    if not r.status_code == 200:
        raise Exception("Failed to download IF-image.csv")
    with open(os.path.join(data_dir, "IF-image.csv"), "wb") as f:
        f.write(r.content)
    df = pd.read_csv(os.path.join(data_dir, "IF-image.csv"))
    df = df[df.latest_version == 23]
    print(f"Checking and download missing for {df.shape[0]}")
    record_file = os.path.join(data_dir, "download_record.csv")
    # if os.path.exists(record_file):
    #     rd = pd.read_csv(record_file)
    #     downloaded_images = rd['filename'].tolist()
    # else:
    downloaded_images = []
    start_download(data_dir, df, downloaded_images, process_num=2)
    with open(os.path.join(data_dir, "IF-image-finished.csv"), "wb") as f:
        f.write(r.content)
    # df.to_csv(record_file)
