import os
import time
from glob import glob
from pathlib import Path
import argparse

import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from models import RegressorLit, DINO
from data import RefChannelIntensityDM


##########################################################################################
# Set up environment and parser
##########################################################################################

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
torch.set_float32_matmul_precision('medium')

parser = argparse.ArgumentParser(description="Train a model to align the FUCCI dataset reference channels with the FUCCI channels",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--data", required=True, help="path to dataset folder")
parser.add_argument("-n", "--name", required=True, help="dataset version name")
parser.add_argument("-e", "--epochs", type=int, default=100, help="maximum number of epochs to train for")
parser.add_argument("-c", "--checkpoint", help="path to checkpoint to load from")
parser.add_argument("-r", "--run", default=time.strftime('%Y_%m_%d_%H_%M'), help="Name to help lookup the run's logging directory")

args = parser.parse_args()

##########################################################################################
# Experiment parameters and logging
##########################################################################################
config = {
    # "batch_size": 32,
    "batch_size": 64,
    "devices": [4, 5, 6, 7],
    # "devices": [4, 5, 6, 7],
    # "devices": [0],
    # "devices": [1, 2, 3],
    "num_workers": 1,
    "split": (0.64, 0.16, 0.2),
    "conv": False,
    "lr": 1e-4,
    "epochs": args.epochs,
    "nf": 16,
    # "n_hidden": 3,
    "n_hidden": 1,
    # "d_hidden": DINO.CLS_DIM * 12,
    "d_hidden": DINO.CLS_DIM * 8,
    "dropout": True,
    "batchnorm": True,
    "num_classes": 2
}

NUM_CHANNELS, NUM_CLASSES = 2, config["num_classes"]

def print_with_time(msg):
    print(f"[{time.strftime('%m/%d/%Y @ %H:%M')}] {msg}")

fucci_path = Path(args.data)
project_name = f"intensity_conv_regressor" if config["conv"] else f"intensity_dino_regressor"
log_dirs_home = Path("/data/ishang/pseudotime_pred/")
log_folder = log_dirs_home / f"{project_name}_{args.run}"
if not log_folder.exists():
    os.makedirs(log_folder, exist_ok=True)
lightning_dir = log_folder / "lightning_logs"
wandb_dir = log_folder

wandb_logger = WandbLogger(
    project=project_name,
    log_model=True,
    save_dir=wandb_dir,
    config=config
)

val_checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="validate/loss",
    mode="min",
    dirpath=lightning_dir,
    filename="{epoch:02d}-{validate/loss:.2f}",
    auto_insert_metric_name=False,
)

latest_checkpoint_callback = ModelCheckpoint(dirpath=lightning_dir, save_last=True)

##########################################################################################
# Set up data, model, and trainer
##########################################################################################

if args.checkpoint is not None:
    chkpt_dir_pattern = f"{log_dirs_home}/*/*/*-{args.checkpoint}/"
    checkpoint_folder = glob(chkpt_dir_pattern)
    if len(checkpoint_folder) > 1:
        raise ValueError(f"Multiple possible checkpoints found: {checkpoint_folder}")
    if len(checkpoint_folder) == 0:
        raise ValueError(f"No checkpoint found for glob pattern: {chkpt_dir_pattern}")
    args.checkpoint = Path(checkpoint_folder[0]).parent.parent / "lightning_logs" / "last.ckpt"
    if not args.checkpoint.exists():
        raise ValueError(f"Checkpoint path {args.checkpoint} does not exist")

print_with_time("Setting up model and data module...")
if not config["conv"]:
    dm = RefChannelIntensityDM(fucci_path, args.name, config["batch_size"], config["num_workers"], config["split"])
    if args.checkpoint is None:
        print("Training from scratch")
        model = RegressorLit(d_input=DINO.CLS_DIM, d_output=NUM_CLASSES, d_hidden=config["d_hidden"], n_hidden=config["n_hidden"], 
                            dropout=config["dropout"], batchnorm=["batchnorm"], lr=config["lr"])
    else:
        print(f"Loading checkpoint from {args.checkpoint}")
        model = RegressorLit.load_from_checkpoint(args.checkpoint)
else:
    raise NotImplementedError("Convolutional regressor not implemented yet")
    # dm = RefChannelCellCycle(Path(args.data), args.name, config["batch_size"], config["num_workers"], config["split"],
    #                          ward=config["ward"], num_classes=NUM_CLASSES)
    # args.data = Path(args.data)
    # if not args.data.is_absolute():
    #     args.data = Path.cwd() / args.data
    # config_file = args.data / args.name
    # sys.path.append(str(config_file.parent))
    # dataset_config = import_module(str(config_file.stem))
    # model = ClassifierLit(conv=True, imsize=dataset_config.output_image_size, nc=NUM_CHANNELS, nf=config["nf"], d_output=NUM_CLASSES,
    #                       dropout=config["dropout"], lr=config["lr"], soft=config["soft"])

wandb_logger.watch(model, log="all", log_freq=10)

print_with_time("Setting up trainer...")

trainer = pl.Trainer(
    default_root_dir=lightning_dir,
    accelerator="gpu",
    devices=config["devices"],
    # strategy=DDPStrategy(find_unused_parameters=True),
    logger=wandb_logger,
    max_epochs=config["epochs"],
    gradient_clip_val=5e5,
    callbacks=[
        val_checkpoint_callback,
        latest_checkpoint_callback,
    ]
)

##########################################################################################
# Train and test model
##########################################################################################

print_with_time("Training model...")
trainer.fit(model, dm)

# print_with_time("Testing model...")
# trainer = pl.Trainer(
#     default_root_dir=lightning_dir,
#     accelerator="gpu",
#     devices=config["devices"][-1:],
#     logger=wandb_logger,
# )
# trainer.test(model, dm)