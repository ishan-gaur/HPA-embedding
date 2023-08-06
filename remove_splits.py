from pathlib import Path
import shutil
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", help="Path to the dataset directory", required=True)
args = parser.parse_args()
data_dir = Path(args.dir)

for split in data_dir.iterdir():
    if split.is_dir():
        to_rm = False
        for well in split.iterdir():
            if well.is_dir():
                if data_dir / well.name in data_dir.iterdir():
                    print(f"Folder {well.name} already exists in {data_dir}")
                    shutil.rmtree(str(well))
                else:
                    shutil.move(str(well), str(data_dir))
                to_rm = True
        if to_rm:
            shutil.rmtree(str(split))

