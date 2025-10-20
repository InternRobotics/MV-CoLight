import argparse
import os
from functools import partial
from pathlib import Path

from PIL import Image
import imageio
import lpips
import numpy as np
import skvideo.io
import torch

from einops import rearrange, repeat
    
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as fused_ssim
import lpips

from utils.camera_trajectory.interpolation import interpolate_extrinsics, interpolate_intrinsics
from utils.camera_trajectory.wobble import generate_wobble, generate_wobble_transformation

def prep_image(image):
    # Handle batched images.
    if image.ndim == 4:
        image = rearrange(image, "b c h w -> c h (b w)")

    # Handle single-channel images.
    if image.ndim == 2:
        image = rearrange(image, "h w -> () h w")

    # Ensure that there are 3 or 4 channels.
    channel, _, _ = image.shape
    if channel == 1:
        image = repeat(image, "() h w -> c h w", c=3)
    assert image.shape[0] in (3, 4)

    image = (image.detach().clip(min=0, max=1) * 255).type(torch.uint8)
    return rearrange(image, "c h w -> h w c").cpu().numpy()

def save_video(images, path, fps=None):
    """Save an image. Assumed to be in range 0-1."""

    # Create the parent directory if it doesn't already exist.
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    
    # Save the image.
    # Image.fromarray(prep_image(image)).save(path)
    frames = []
    for image in images:
        frames.append(prep_image(image))

    outputdict = {'-pix_fmt': 'yuv420p', '-crf': '23',
                  '-vf': f'setpts=1.*PTS'}
                  
    if fps is not None:
        outputdict.update({'-r': str(fps)})
    
    writer = skvideo.io.FFmpegWriter(path,
                                     outputdict=outputdict)
    for frame in frames:
        writer.writeFrame(frame)
    writer.close()

def render_video_interpolation_exaggerated(batch):
    # Two views are needed to get the wobble radius.
    _, v, _, _ = batch["input_extrinsics"].shape
    if v != 2:
        return

    def trajectory_fn(t):
        origin_a = batch["input_extrinsics"][:, 0, :3, 3]
        origin_b = batch["input_extrinsics"][:, 1, :3, 3]
        delta = (origin_a - origin_b).norm(dim=-1)
        tf = generate_wobble_transformation(
            delta * 0.5,
            t,
            5,
            scale_radius_with_t=False,
        )
        extrinsics = interpolate_extrinsics(
            batch["input_extrinsics"][:, 0],
            (
                batch["input_extrinsics"][:, 1]
                if v == 2
                else batch["input_intrinsics"][:, 0]
            ),
            t * 5 - 2,
        )
        intrinsics = interpolate_intrinsics(
            batch["input_intrinsics"][:, 0],
            (
                batch["input_intrinsics"][:, 1]
                if v == 2
                else batch["input_intrinsics"][:, 0]
            ),
            t * 5 - 2,
        )
        return extrinsics @ tf, intrinsics

    return trajectory_fn

def render_video_wobble(batch):
    # Two views are needed to get the wobble radius.
    _, v, _, _ = batch["input_extrinsics"].shape
    if v != 2:
        return
    
    def trajectory_fn(t):
        origin_a = batch["input_extrinsics"][:, 0, :3, 3]
        origin_b = batch["input_extrinsics"][:, 1, :3, 3]
        delta = (origin_a - origin_b).norm(dim=-1)

        if (delta == 0).any():
            delta = torch.ones_like(delta) * 1

        extrinsics = generate_wobble(
            batch["input_extrinsics"][:, 0],
            delta * 0.25,
            # delta * 1.0,
            t,
        )
        intrinsics = repeat(
            batch["input_intrinsics"][:, 0],
            "b i j -> b v i j",
            v=t.shape[0],
        )
        return extrinsics, intrinsics
    return trajectory_fn

def render_video_interpolation(batch):
    _, v, _, _ = batch["input_extrinsics"].shape
    
    def trajectory_fn(t):
        extrinsics = interpolate_extrinsics(
            batch["input_extrinsics"][:, 0],
            (
                batch["input_extrinsics"][:, 1]
                if v == 2
                else batch["target_extrinsics"][:, 0]
            ),
            t,
        )
        intrinsics = interpolate_intrinsics(
            batch["input_intrinsics"][:, 0],
            (
                batch["input_intrinsics"][:, 1]
                if v == 2
                else batch["target_intrinsics"][:, 0]
            ),
            t,
        )
        return extrinsics, intrinsics

    return trajectory_fn