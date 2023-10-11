import os
import time
from glob import glob
from pathlib import Path
import argparse

import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from models import PseudoRegressorLit, DINO
from data import RefChannelPseudoDM


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
parser.add_argument("--best", action="store_true", help="load best checkpoint instead of last checkpoint")
parser.add_argument("-r", "--run", default=time.strftime('%Y_%m_%d_%H_%M'), help="Name to help lookup the run's logging directory")

args = parser.parse_args()

##########################################################################################
# Experiment parameters and logging
##########################################################################################
HPA = True
# dataset = (("fucci_cham", "fucci_tile"), "fucci_over", "fucci_over")
# dataset = (("fucci_cham", "fucci_tile"), "fucci_cham", "fucci_over")
# dataset = (("fucci_cham", "fucci_tile"), "fucci_tile", "fucci_over")
# dataset = ("fucci_cham", ("fucci_tile", "fucci_over"), "fucci_over")
# dataset = ["fucci_cham", "fucci_tile"]
# dataset = "fucci_cham"
dataset = "fucci_tile"
concat_well_stats = True
CART = False
if not HPA:
    DINO_INPUT = DINO.CLS_DIM
else:
    DINO_INPUT = 768 if dataset is None else 2 * 768
if concat_well_stats:
    DINO_INPUT += 2 * 64
config = {
    "HPA": HPA,
    "dataset": dataset,
    "concat_well_stats": concat_well_stats,
    "loss_type": "cart" if CART else "arc",
    "reweight_loss": True,
    "bins": 6,
    # "batch_size": 32,
    "batch_size": 64,
    "devices": [0, 1, 2, 3, 4, 5, 6, 7],
    # "devices": [0, 1, 2, 3],
    # "devices": [1, 2, 3, 4],
    # "devices": [4, 5, 6, 7],
    # "devices": [0],
    # "devices": [7],
    # "devices": [0, 1, 2, 3, 4],
    "num_workers": 1,
    # "split": (0.64, 0.16, 0.2),
    "split": (0.8, 0.2, 0.0),
    "conv": False,
    "lr": 1e-4,
    # "lr": 0,
    "epochs": args.epochs,
    "nf": 16,
    "n_hidden": 3,
    # "n_hidden": 1,
    # "d_hidden": DINO_INPUT * 12,
    "d_hidden": DINO_INPUT * 2,
    "dropout": False,
    "batchnorm": True,
    "num_classes": 1
}

NUM_CHANNELS, NUM_CLASSES = 2, config["num_classes"]

def print_with_time(msg):
    print(f"[{time.strftime('%m/%d/%Y @ %H:%M')}] {msg}")

fucci_path = Path(args.data)
project_name = f"pseudo_conv_regressor" if config["conv"] else f"pseudo_dino_regressor"
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
    models_folder = Path(checkpoint_folder[0]).parent.parent / "lightning_logs"
    if args.best:
        models_list = list(models_folder.iterdir())
        models_list.sort()
        # the elements should be ###-##.ckpt, 'epoch=###.ckpt', and 'last.ckpt'
        args.checkpoint = models_list[0]
    else:
        args.checkpoint = models_folder / "last.ckpt"
    if not args.checkpoint.exists():
        raise ValueError(f"Checkpoint path {args.checkpoint} does not exist")

print_with_time("Setting up model and data module...")
if not config["conv"]:
    dm = RefChannelPseudoDM(fucci_path, args.name, config["batch_size"], config["num_workers"], config["split"], HPA=config["HPA"], 
                            dataset=config["dataset"], concat_well_stats=config["concat_well_stats"])
    if args.checkpoint is None:
        print("Training from scratch")
        model = PseudoRegressorLit(d_input=DINO_INPUT, d_output=NUM_CLASSES, d_hidden=config["d_hidden"], n_hidden=config["n_hidden"], 
                            dropout=config["dropout"], batchnorm=["batchnorm"], lr=config["lr"], loss_type=config["loss_type"],
                            reweight_loss=config["reweight_loss"], bins=config["bins"])
    else:
        print(f"Loading checkpoint from {args.checkpoint}")
        model = PseudoRegressorLit.load_from_checkpoint(args.checkpoint)
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

model.lr = config["lr"]
model.loss_type = config["loss_type"]
model.reweight_loss = config["reweight_loss"]
model.bins = config["bins"]

wandb_logger.watch(model, log="all", log_freq=10)

print_with_time("Setting up trainer...")

trainer = pl.Trainer(
    default_root_dir=lightning_dir,
    # accelerator="cpu",
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