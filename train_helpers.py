import logging
import os
import pathlib
import shutil
from mmengine.config import Config
import imageio
import numpy as np
import torch
import torchvision
import wandb
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from einops import rearrange
from torchmetrics.functional import (
    peak_signal_noise_ratio as psnr,
    structural_similarity_index_measure as fused_ssim,
)
from tqdm import tqdm

amp_dtype_mapping = {"fp16": torch.float16, "bf16": torch.bfloat16}

class EMA:
    def __init__(self, beta: float) -> None:
        """Initialize EMA with decay rate beta.
        
        Args:
            beta: Decay rate between 0 and 1
        """
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_average(self, old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def update(self, old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        if old is None:
            return new
        return self.update_average(old, new)

def calculate_grad_norm(model):
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


def get_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        return f"Alloc: {allocated:.0f}MB, Reserved: {reserved:.0f}MB"
    return "GPU not available"


def save_and_log_images(
    input_imgs,
    target_imgs,
    predictions,
    eval_dir,
    step,
    num_images=10,
    split="train",
):
    B, _, _, H, W = predictions.shape
    num_images = min(num_images, B)
    os.makedirs(eval_dir, exist_ok=True)

    # save img (input+pred) / (input+target)
    input_pred_target_list = []
    input_pred_target_imgs = torch.cat([input_imgs[:, :, :3, :, :], predictions, target_imgs], dim=1).detach().cpu() # (B, 2+6, 3, H, W)
    for i in range(num_images):
        input_pred_target_image = input_pred_target_imgs[i].permute(1, 2, 0, 3).flatten(2, 3) # (3, H, V*W)
        input_pred_target_list.append(input_pred_target_image)
    input_pred_target_imgs = torch.cat(input_pred_target_list, dim=1)
    torchvision.utils.save_image(input_pred_target_imgs, os.path.join(eval_dir, f"input_pred_target_{split}_{step}.png"))

def save_and_log_images(
    input_imgs,
    target_imgs,
    predictions,
    eval_dir,
    step,
    num_images=10,
    split="train",
):
    B, _, _, H, W = predictions.shape
    num_images = min(num_images, B)
    os.makedirs(eval_dir, exist_ok=True)
    
    # save img (input+pred) / (input+target)
    image_list = []
    if input_imgs==None:
        images = torch.cat([predictions, target_imgs], dim=1).detach().cpu() # (B, 2+6, 3, H, W)
    else:
        images = torch.cat([input_imgs[:, :, :3, :, :], predictions, target_imgs], dim=1).detach().cpu() # (B, 2+6, 3, H, W)
    for i in range(num_images):
        image = images[i].permute(1, 2, 0, 3).flatten(2, 3) # (3, H, V*W)
        image_list.append(image)
    images = torch.cat(image_list, dim=1)
    torchvision.utils.save_image(images, os.path.join(eval_dir, f"img_{split}_{step}.png"))

def save_and_log_rgb(
    input_rgb,
    target_rgb,
    pred_rgb,
    eval_dir,
    step,
    num_images=10,
    split="train",
):
    B, _, _, H, W = pred_rgb.shape
    num_images = min(num_images, B)
    os.makedirs(eval_dir, exist_ok=True)
    
    # save img (input+pred) / (input+target)
    rgb_list = []
    rgbs = torch.cat([input_rgb, pred_rgb, target_rgb], dim=1).detach().cpu() # (B, 2+6, 3, H, W)
    for i in range(num_images):
        rgb = rgbs[i].permute(1, 2, 0, 3).flatten(2, 3) # (3, H, V*W)
        rgb_list.append(rgb)
    rgbs = torch.cat(rgb_list, dim=1)
    torchvision.utils.save_image(rgbs, os.path.join(eval_dir, f"rgb_{split}_{step}.png"))

def saveRuntimeCode(dst: str) -> None:
    additionalIgnorePatterns = [".git", ".gitignore"]
    ignorePatterns = set()
    ROOT = "."
    gitignore_path = os.path.join(ROOT, ".gitignore")
    if os.path.exists(gitignore_path):
        with open(gitignore_path) as gitIgnoreFile:
            for line in gitIgnoreFile:
                line = line.strip()
                if not line.startswith("#") and line != "":
                    if line.endswith("/"):
                        line = line[:-1]
                    ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)

    log_dir = pathlib.Path(__file__).parent.resolve()
    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))
    print("Backup Finished!")

def create_optimizer(model, weight_decay, learning_rate, betas) -> torch.optim.AdamW:
    decay_params, nodecay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'layernorm' in name.lower():  
            nodecay_params.append(param)
        else:
            decay_params.append(param)
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
    return optimizer
    
def create_scheduler(optimizer, total_train_steps, warm_up_steps, scheduler_type='cosine'):
    if scheduler_type == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer, warm_up_steps, total_train_steps)
    elif scheduler_type == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(optimizer, warm_up_steps, total_train_steps)
    elif scheduler_type == 'constant':
        scheduler = get_constant_schedule_with_warmup(optimizer, warm_up_steps)
    else:
        raise ValueError(f'Invalid scheduler type: {scheduler_type}')
    return scheduler

def compute_plucker_rays_gpu(intrinsics, extrinsics, H, W, device):
    B = intrinsics.shape[0]
    num_views = intrinsics.shape[1]

    # Prepare pixel grid
    i, j = torch.meshgrid(
        torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"
    )
    # i = (i.float() + 0.5).reshape(-1)  # Center align
    # j = (j.float() + 0.5).reshape(-1)
    i = (i.float() + 0.5).reshape(-1) / (W - 1)
    j = (j.float() + 0.5).reshape(-1) / (H - 1)
    grid = torch.stack([j, i, torch.ones_like(i)], dim=1)  # [H*W, 3]

    fx = intrinsics[..., 0, 0].reshape(B, num_views, 1)
    fy = intrinsics[..., 1, 1].reshape(B, num_views, 1)
    cx = intrinsics[..., 0, 2].reshape(B, num_views, 1)
    cy = intrinsics[..., 1, 2].reshape(B, num_views, 1)

    grid = (
        grid.unsqueeze(0).unsqueeze(0).expand(B, num_views, -1, -1)
    )  # [B, num_views, H*W, 3]

    # Adjust for intrinsics
    directions = grid.clone()
    directions[..., 0] = (directions[..., 0] - cx) / fx
    directions[..., 1] = (directions[..., 1] - cy) / fy
    directions = directions / (
        torch.norm(directions, dim=-1, keepdim=True) + 1e-6
    )  # Normalize

    # Convert transforms to tensors
    transforms = torch.tensor(extrinsics, dtype=torch.float32, device=device)
    rotation = transforms[..., :3, :3]
    # rotation = torch.linalg.qr(rotation)[0]  # Ensure orthogonality #! don't understand
    translation = transforms[..., :3, 3]

    # Transform directions to world space
    directions_world = torch.einsum("bvij,bvnj->bvni", rotation, directions)
    directions_world = directions_world / (
        torch.norm(directions_world, dim=-1, keepdim=True) + 1e-6
    )

    # Compute ray origins in world space
    origins_world = translation.unsqueeze(2).expand_as(directions_world)

    # Compute Plücker coordinates
    rays_dxo = torch.cross(origins_world, directions_world, dim=-1)
    plucker = torch.cat([rays_dxo, directions_world], dim=-1)

    # Reshape to final format
    assert plucker.shape[2] == H * W, "Mismatch in plucker rays shape"
    plucker_rays = plucker.view(B, num_views, H, W, 6).permute(0, 1, 4, 2, 3)

    return plucker_rays