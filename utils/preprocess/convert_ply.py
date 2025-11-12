import os
import shutil
import json
import torch
import numpy as np
from tqdm import tqdm
import cv2
from plyfile import PlyData
import torchvision.transforms as tf
from PIL import Image

C0 = 0.28209479177387814

def SH2RGB(sh):
    return sh * C0 + 0.5

def hilbert_curve_2d(order, x0, y0, xi, xj, yi, yj, points):
    if order == 0:
        x = x0 + (xi + yi) / 2
        y = y0 + (xj + yj) / 2
        points.append((x, y))
    else:
        hilbert_curve_2d(order - 1, x0, y0, yi / 2, yj / 2, xi / 2, xj / 2, points)
        hilbert_curve_2d(order - 1, x0 + xi / 2, y0 + xj / 2, xi / 2, xj / 2, yi / 2, yj / 2, points)
        hilbert_curve_2d(order - 1, x0 + xi / 2 + yi / 2, y0 + xj / 2 + yj / 2, xi / 2, xj / 2, yi / 2, yj / 2, points)
        hilbert_curve_2d(order - 1, x0 + xi / 2 + yi, y0 + xj / 2 + yj, -yi / 2, -yj / 2, -xi / 2, -xj / 2, points)

def convert_ply(path, hilbert, order):
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3))
    features_dc[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])
    rgbs = SH2RGB(features_dc)

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    scales = torch.from_numpy(scales).float()
    scales = torch.exp(scales)
    scales = scales.numpy()

    rots = torch.from_numpy(rots).float()
    rots = torch.nn.functional.normalize(rots)
    rots = rots.numpy()

    feat = np.zeros((2 ** order, 2 ** order, 3+1+3+4), dtype=np.float32)
    for i, point, opacity, scale, rot in zip(range(len(xyz)), xyz, opacities, scales, rots):
        if i < len(hilbert):
            x, y = hilbert[i]
            feat[int(x), int(y)] = np.concatenate([point, opacity, scale, rot])
    
    return feat

ROOT_PATH = "/path/to/DTC-MultiLight3D"
for scene in tqdm(sorted(os.listdir(ROOT_PATH))):
    point_cloud_path = os.path.join(ROOT_PATH, scene, "composite_scene/gs/point_cloud/iteration_10000/point_cloud.ply")
    order = 9
    hilbert = []
    hilbert_curve_2d(order, 0, 0, 2 ** order, 0, 0, 2 ** order, hilbert)
    feat = convert_ply(point_cloud_path, hilbert, order)
    np.save(os.path.join(ROOT_PATH, scene, "gs.npy"), feat)
