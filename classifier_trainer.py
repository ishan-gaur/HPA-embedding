import sys
import os
import time
from pathlib import Path
import argparse
from importlib import import_module

import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy

from models import ClassifierLit, DINO
from data import CellCycleModule


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

if args.checkpoint is not None:
    if not Path(args.checkpoint).exists():
        raise ValueError("Checkpoint path does not exist.")
    print("WARNING: Checkpoint loading not implemented, ignoring...")

##########################################################################################
# Experiment parameters and logging
##########################################################################################
config = {
    "batch_size": 64,
    "devices": [6],
    # "devices": list(range(4, torch.cuda.device_count())),
    # "devices": list(range(0, torch.cuda.device_count())),
    "num_workers": 1,
    # "num_workers": 4,
    # "num_workers": 8,
    "split": (0.64, 0.16, 0.2),
    "lr": 5e-5,
    "epochs": args.epochs,
    "latent_dim": 512,
    "lambda": 5e6,
}

def print_with_time(msg):
    print(f"[{time.strftime('%m/%d/%Y @ %H:%M')}] {msg}")

fucci_path = Path(args.data)
project_name = f"FUCCI_dino_classifier"
log_folder = Path(f"/data/ishang/fucci_vae/{project_name}_{args.run}")
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

print_with_time("Setting up data module...")
dm = CellCycleModule(Path(args.data), args.name, config["batch_size"], config["num_workers"], config["split"])
# if not Path(args.data).is_absolute():
#     args.data = str(Path.cwd() / args.data)
# sys.path.append(args.data)
# dataset_config = import_module(args.name)
model = ClassifierLit(d_input=DINO.CLS_DIM, d_output=3, lr=config["lr"]) # 3 components in the GMM output

wandb_logger.watch(model, log="all", log_freq=10)

print_with_time("Setting up trainer...")

trainer = pl.Trainer(
    default_root_dir=lightning_dir,
    accelerator="gpu",
    devices=config["devices"],
    strategy=DDPStrategy(find_unused_parameters=True),
    logger=wandb_logger,
    max_epochs=config["epochs"],
    gradient_clip_val=5e5,
    callbacks=[
        val_checkpoint_callback,
        latest_checkpoint_callback,
        # LearningRateMonitor(logging_interval='step'),
        # ReconstructionVisualization(channels=None if args.model == "total" else dm.get_channels(), mode=reconstuction_mode),
        # EmbeddingLogger(every_n_epochs=1, mode=args.model, channels=dm.get_channels() if args.model == "all" else None),
    ]
)

##########################################################################################
# Train and test model
##########################################################################################

print_with_time("Training model...")
trainer.fit(model, dm)

print_with_time("Testing model...")
trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    num_nodes=1,
)
trainer.test(model, dm)