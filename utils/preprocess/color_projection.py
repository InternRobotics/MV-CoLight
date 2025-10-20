import argparse
import os
import json
import imageio
import math
import numpy as np
import cv2
import torch
import random
from plyfile import PlyData, PlyElement
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import cv2
import shutil
import os
from concurrent.futures import ThreadPoolExecutor
import torch
import argparse

import os
import torch
import numpy as np
import argparse
from plyfile import PlyData
from typing import NamedTuple
import concurrent.futures
import time
from multiprocessing import Pool  

INPUT_DIR = "/path/to/renkerui/DTC-MultiLight-2DEval/simple_scenes"
OUTPUT_DIR = "/path/to/DTC-MultiLight-3DEval/simple_scenes"
PROJ_DIR = "/path/to/DTC-MultiLight-3D/simple_scenes"
JSON_PATH = "./utils/preprocess/test_json/simple_scenes.json"

# EXR
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

def gen_cam_uv(wid, hei):
    u_min  = -1.0
    u_max  = 1.0
    v_min  = -1.0
    v_max  = 1.0
    half_du = 0.5 * (u_max - u_min) / wid
    half_dv = 0.5 * (v_max - v_min) / hei

    u, v = np.meshgrid(np.linspace(u_min+half_du, u_max-half_du, wid),
                       np.linspace(v_min+half_dv, v_max-half_dv, hei)[::-1])
    uvs_2d = np.dstack((u,v,np.ones_like(u)))
    return uvs_2d

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def convert_extrinsics(pose):
    blender2opencv = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    )
    opencv_c2w = np.array(pose) @ blender2opencv
    return opencv_c2w.tolist()

def rgb2color(rgbs, object_masks, env_masks):
    colors = []
    for object_mask in object_masks:
        for i in range(rgbs.shape[0]):
            depth_mask = env_masks[i] == 0
            depth_mask[object_mask[i]==0] = False
            color = rgbs[i].reshape(-1,3)/255.
            colors.append(color[depth_mask.reshape(-1)])
            
    for i in range(rgbs.shape[0]):
        color = rgbs[i].reshape(-1,3)/255.
        depth_mask = env_masks[i] == 0
        for object_mask in object_masks:       
            depth_mask[object_mask[i] > 0] = False
        colors.append(color[depth_mask.reshape(-1)])
    colors = torch.cat(colors, dim=0) * 255
    return colors

def color_proj(scene, proj_path, gs_path):

    projs = np.load(proj_path, allow_pickle=True)

    scene_path = os.path.join(INPUT_DIR, scene)
    save_path = os.path.join(OUTPUT_DIR, scene)
    os.makedirs(save_path, exist_ok=True)
    composite_path = os.path.join(scene_path, 'composites')
    
    x = []
    for img in sorted(os.listdir(composite_path), key=lambda x: int(x.split('.')[0])):
        img = cv2.imread(os.path.join(composite_path, img), -1)
        x.append(img)
    
    x = torch.from_numpy(np.array(x)).unsqueeze(0).permute(0, 1, 4, 2, 3).float()

    B, N, C, H, W = x.shape
    Gh = Gw = 512
    x = x.reshape(B, N, C, H * W)
    new_color = torch.zeros((B, C, Gh, Gw), device=x.device)

    # 计算索引
    proj_indices = projs[:, 1]

    # 使用高级索引直接赋值
    batch_indices = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, Gh * Gw)
    height_indices = projs[:, 2]
    width_indices = projs[:, 3]
    channel_indices = projs[:, 0]

    # 使用 gather 和 scatter
    x_selected = x[batch_indices, channel_indices, :, proj_indices]
    new_color[batch_indices, :, height_indices, width_indices] = x_selected

    new_color = new_color.permute(0, 2, 3, 1).squeeze(0)
    cv2.imwrite(os.path.join(save_path, f'composite.png'), new_color.cpu().numpy())

    gt_path = os.path.join(scene_path, 'gts')

    x = []
    for img in sorted(os.listdir(gt_path), key=lambda x: int(x.split('.')[0])):
        img = cv2.imread(os.path.join(gt_path, img), -1)
        x.append(img)
    
    x = torch.from_numpy(np.array(x)).unsqueeze(0).permute(0, 1, 4, 2, 3).float()

    B, N, C, H, W = x.shape
    x = x.reshape(B, N, C, H * W)
    new_color = torch.zeros((B, C, Gh, Gw), device=x.device)

    # 计算索引
    proj_indices = projs[:, 1]

    # 使用高级索引直接赋值
    batch_indices = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, Gh * Gw)
    height_indices = projs[:, 2]
    width_indices = projs[:, 3]
    channel_indices = projs[:,  0]

    # 使用 gather 和 scatter
    x_selected = x[batch_indices, channel_indices, :, proj_indices]
    new_color[batch_indices, :, height_indices, width_indices] = x_selected

    new_color = new_color.permute(0, 2, 3, 1).squeeze(0)
    cv2.imwrite(os.path.join(save_path, f'gt.png'), new_color.cpu().numpy())

    shutil.copy(gs_path, os.path.join(save_path, f"gs.npy"))

def main():
    with open(JSON_PATH) as f:
        data_dict = json.load(f)

    for scene in sorted(os.listdir(INPUT_DIR)):
        base_idx = data_dict[scene][0]
        lcd_list = sorted(os.listdir(os.path.join(PROJ_DIR, scene)))
        proj_path = os.path.join(PROJ_DIR, scene, "projs.npy")
        gs_path = os.path.join(PROJ_DIR, scene, lcd_list[base_idx], "gs.npy")
        color_proj(scene, proj_path, gs_path)

if __name__ == "__main__":
    main()