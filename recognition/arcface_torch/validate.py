import argparse
import logging
import os
from datetime import datetime

import numpy as np
import torch
from backbones import get_model
from dataset import get_dataloader
from losses import CombinedMarginLoss
from lr_scheduler import PolynomialLRWarmup
from partial_fc_v2 import PartialFC_V2
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_distributed_sampler import setup_seed
from utils.utils_logging import AverageMeter, init_logging
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook

import sys

sys.path.append("/home/c_samarth/insightface/recognition/arcface_torch")
from backbones.mobilefacenet import MobileFaceNet

assert (
    torch.__version__ >= "1.12.0"
), "In order to enjoy the features of the new torch, \
we have upgraded the torch to 1.12.0. torch before than 1.12.0 may not work in the future."

try:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    local_rank = 2
    world_size = int(os.environ["WORLD_SIZE"])
    distributed.init_process_group("nccl")
except KeyError:
    rank = 0
    local_rank = 1 # Sets the GPU index
    world_size = 1
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12585",
        rank=rank,
        world_size=world_size,
    )


def main(args):
    # get config
    cfg = get_config(args.config)
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)
    print(cfg.rank_id)
    print("Setting device ", local_rank)
    torch.cuda.set_device(local_rank)

    os.makedirs(cfg.output, exist_ok=True)

    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
        if rank == 0
        else None
    )

    wandb_logger = None

    start_epoch = 0
    global_step = 0
    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets,
        rec_prefix=cfg.rec,
        summary_writer=summary_writer,
        wandb_logger=wandb_logger,
    )
    callback_verification.ver_test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch"
    )
    parser.add_argument("config", type=str, help="py config file")
    main(parser.parse_args())
