import os
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Reorganize FUCCI dataset into wells')
parser.add_argument('--data', type=str, default='/data/ishang/FUCCI-dataset/')
parser.add_argument('--output', type=str, default='/data/ishang/FUCCI-dataset-well/')
parser.add_argument('--copy', action='store_true', default=False)

image_files = ['Geminin.png', 'microtubule.png', 'nuclei.png', 'CDT1.png']

args = parser.parse_args()

if not Path(args.data).is_absolute():
    raise ValueError("Data directory should be an absolute path")

if not Path(args.output).is_absolute():
    raise ValueError("Output directory should be an absolute path")
if not Path(args.output).exists():
    os.mkdir(args.output)

args.data = Path(args.data)
args.output = Path(args.output)

if args.copy:
    import shutil
    print(f'Copying all folders\' images from {str(args.data)} to {str(args.output)}')

    for folder in args.data.iterdir():
        if not folder.is_dir():
            continue

        skip = False
        for file_name in image_files:
            if not (folder / file_name).exists():
                print(f'Folder {folder.name} is missing {file_name}')
                skip = True
                break
        
        if skip: 
            continue

        os.mkdir(args.output / folder.name)
        for file_name in image_files:
            shutil.copy(src=(folder / file_name), dst=(args.output / folder.name))