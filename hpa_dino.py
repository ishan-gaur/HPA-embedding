import os
import yaml
from tqdm import tqdm
from pathlib import Path
from functools import partial

import torch
import torch.nn as nn
from torch.nn import DataParallel

from DINO4Cells_code.archs import vision_transformer as vits
from DINO4Cells_code.archs.vision_transformer import DINOHead
from DINO4Cells_code.archs import xresnet as cell_models  # (!)
from DINO4Cells_code.utils import utils

config_file = Path("/home/ishang/HPA-embedding/dino_config.yaml")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = yaml.safe_load(open(config_file, "r"))
class HPA_DINO:
    def __init__(self, imsize, batch_size=100, config=config, device=device) -> None:
        if config["model"]["arch"] in vits.__dict__.keys():
            print("config['model']['arch'] {} in vits".format(config["model"]["arch"]))
            # model = vits.__dict__[config['model']['arch']](img_size=[112], patch_size=config['model']['patch_size'], num_classes=0, in_chans=config['model']['num_channels'])
            # model = vits.__dict__[config['model']['arch']](img_size=[512], patch_size=config['model']['patch_size'], num_classes=0, in_chans=config['model']['num_channels'])
            model = vits.__dict__[config["model"]["arch"]](
                img_size=[224],
                patch_size=config["model"]["patch_size"],
                num_classes=0,
                in_chans=config["model"]["num_channels"],
            )
            # model = vits.__dict__[config['model']['arch']](img_size=[224], patch_size=config['model']['patch_size'], num_classes=0, in_chans=config['model']['num_channels'])
            embed_dim = model.embed_dim
        elif config["model"]["arch"] in cell_models.__dict__.keys():
            print(f"config['model']['arch'] {config['model']['arch']} in cell_models")
            model = partial(
                cell_models.__dict__[config["model"]["arch"]],
                c_in=config["model"]["num_channels"],
            )(False)
            embed_dim = model[-1].in_features
            model[-1] = nn.Identity()

        if config["embedding"]["HEAD"] == True:
            model = utils.MultiCropWrapper(
                model,
                DINOHead(
                    embed_dim,
                    config["model"]["out_dim"],
                    config["model"]["use_bn_in_head"],
                ),
            )

        for p in model.parameters():
            p.requires_grad = False

        model.eval()
        model.to(device)
        pretrained_weights = config["embedding"]["pretrained_weights"]
        print(f'loaded {config["embedding"]["pretrained_weights"]}')

        if os.path.isfile(pretrained_weights):
            state_dict = torch.load(pretrained_weights, map_location="cpu")
            if "teacher" in state_dict:
                teacher = state_dict["teacher"]
                if not config["embedding"]["HEAD"] == True:
                    teacher = {k.replace("module.", ""): v for k, v in teacher.items()}
                    teacher = {
                        k.replace("backbone.", ""): v for k, v in teacher.items()
                    }
                msg = model.load_state_dict(teacher, strict=False)
            else:
                student = state_dict
                if not config["embedding"]["HEAD"] == True:
                    student = {k.replace("module.", ""): v for k, v in student.items()}
                    student = {
                        k.replace("backbone.", ""): v for k, v in student.items()
                    }
                student = {k.replace("0.", ""): v for k, v in student.items()}
                msg = model.load_state_dict(student, strict=False)

            for p in model.parameters():
                p.requires_grad = False
            model = model.cuda()
            model = model.eval()
            model = DataParallel(model)
            print(
                "Pretrained weights found at {} and loaded with msg: {}".format(
                    pretrained_weights, msg
                )
            )
        else:
            print(
                "Checkpoint file not found at {}. Please check and retry".format(
                    pretrained_weights
                )
            )

        self.model = model
        self.device = device
        self.config = config
        self.imsize = imsize
        self.batch_size = batch_size
        
        self.model.to(self.device)

    def clear_sample(self, x):
        x[:, 2:] = 0
        while x.shape[1] < 4:
            x = torch.cat([x, torch.zeros_like(x[:, 0:1])], dim=1)
        x.to(self.device)
        return x

    def predict_cls(self, dataset):
        cls_tokens = []
        for i in tqdm(range(0, len(dataset), self.batch_size)):
            sample = dataset[i:i+self.batch_size]
            cleared_sample = self.clear_sample(sample)
            cls = self.model(cleared_sample)
            cls_tokens.append(cls)
        cls_tokens = torch.cat(cls_tokens, dim=0).cpu()
        return cls_tokens