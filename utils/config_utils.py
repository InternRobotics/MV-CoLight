import argparse
from datetime import datetime
import json
import logging
import os
from glob import glob
import random
import subprocess
from typing import Tuple
from datetime import timedelta

from mmengine.config import Config
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist

# os.environ['NCCL_BLOCKING_WAIT'] = '0'  # not to enforce timeout

def create_experiment_workspace(cfg):
    """
    This function creates a folder for experiment tracking.

    Args:
        args: The parsed arguments.

    Returns:
        exp_dir: The path to the experiment folder.
    """
    # Make outputs folder (holds all experiment subfolders)
    os.makedirs(cfg.outputs, exist_ok=True)
    experiment_index = len(glob(f"{cfg.outputs}/*"))
    dist.barrier()
    # Create an experiment folder
    exp_name = f"{cfg.exp_name}"
    exp_dir = f"{cfg.outputs}/{exp_name}"
    os.makedirs(exp_dir, exist_ok=True)
    return exp_name, exp_dir


def save_training_config(cfg, experiment_dir):
    with open(f"{experiment_dir}/config.json", "w") as f:
        json.dump(cfg, f, indent=4)


def create_tensorboard_writer(exp_dir):
    tensorboard_dir = f"{exp_dir}/tensorboard"
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(tensorboard_dir)
    return writer

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected!")

def setup_logger(log_dir, experiment_name=None, is_master=True):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if experiment_name:
        exp_dir = os.path.join(log_dir, f'{experiment_name}', f'{timestamp}')
        log_file = os.path.join(log_dir, f'{experiment_name}', f'{timestamp}', 'outputs.log')
    else:
        exp_dir = os.path.join(log_dir, f'training_{timestamp}')
        log_file = os.path.join(log_dir, f'training_{timestamp}.log')

    os.makedirs(exp_dir, exist_ok=True)

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        if is_master:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

    return logger, exp_dir

def to_torch_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        return dtype
    elif isinstance(dtype, str):
        dtype_mapping = {
            "float64": torch.float64,
            "float32": torch.float32,
            "float16": torch.float16,
            "fp32": torch.float32,
            "fp16": torch.float16,
            "half": torch.float16,
            "bf16": torch.bfloat16,
        }
        if dtype not in dtype_mapping:
            raise ValueError
        dtype = dtype_mapping[dtype]
        return dtype
    else:
        raise ValueError

def format_numel_str(numel: int) -> str:
    B = 1024**3
    M = 1024**2
    K = 1024
    if numel >= B:
        return f"{numel / B:.2f} B"
    elif numel >= M:
        return f"{numel / M:.2f} M"
    elif numel >= K:
        return f"{numel / K:.2f} K"
    else:
        return f"{numel}"

def get_model_numel(model: torch.nn.Module) -> Tuple[int, int]:
    num_params = 0
    num_params_trainable = 0
    for p in model.parameters():
        num_params += p.numel()
        if p.requires_grad:
            num_params_trainable += p.numel()
    return num_params, num_params_trainable

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        # here we don't set seed for all gpus.
        # Each process set the seed for its gpu respectively
        
# initialization for distributed training
def init_dist(args):
    port = args.get("port", 29453)
    # timeout=timedelta(seconds=7200000)
    
    if 'LOCAL_RANK' in os.environ:
    # Environment variables set by torch.distributed.launch or torchrun
        # local_rank = int(os.environ['LOCAL_RANK']) % torch.cuda.device_count()
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        world_rank = int(os.environ['RANK'])
    elif 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
    # Environment variables set by mpirun
        local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        world_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        world_rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        local_rank = world_rank % num_gpus
        addr = subprocess.getoutput(
            f'scontrol show hostname {node_list} | head -n1')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(world_rank)
    else:
        raise NotImplementedError
    torch.cuda.set_device(local_rank)
    if 'SLURM_PROCID' in os.environ:
        while True:
            try:
                port = os.environ.get('PORT', port)
                os.environ['MASTER_PORT'] = str(port)
                #dist.init_process_group(backend="nccl", init_method="env://", rank=world_rank, world_size=world_size, timeout=timeout)
                dist.init_process_group(backend="nccl", init_method="env://", rank=world_rank, world_size=world_size)
                break
            except Exception as e:
                port += 1
    else:
        #dist.init_process_group(backend="nccl", init_method="env://", rank=world_rank, world_size=world_size, timeout=timeout)
        dist.init_process_group(backend="nccl", init_method="env://", rank=world_rank, world_size=world_size)
    
    dist.barrier()
    # record distributed configurations
    args.local_rank = local_rank
    args.world_size = world_size
    args.rank = world_rank
    args.addr = os.environ['MASTER_ADDR']
    args.port = os.environ['MASTER_PORT']

    return args