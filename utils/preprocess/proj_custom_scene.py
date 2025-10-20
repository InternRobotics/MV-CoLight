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
from lapjv import lapjv
from multiprocessing import Pool  
import open3d as o3d

DATA_DIR = "/path/to/custom_scene"
OUTPUT_DIR = "/path/to/output_directory"

# EXR
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

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

def process_scene(tree, hilbert, image_height, image_width):
    scene_path = os.path.join(DATA_DIR)
    composite_ply_path = os.path.join(OUTPUT_DIR, "composite_scene", "points3d.ply")
    gt_ply_path = os.path.join(OUTPUT_DIR, "gt_scene", "points3d.ply")
    harmonized_ply_path = os.path.join(OUTPUT_DIR, "harmonized_scene", "points3d.ply")
    with open(os.path.join(DATA_DIR, "transforms.json"), "r") as f:
        data = json.load(f)
    
    w = 1920
    h = 1080
    fovx = data["camera_angle_x"]
    fx = fov2focal(fovx, w)
    fy = fx
    cx = w / 2
    cy = h / 2

    intrinsic = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
    depths, c2ws, composite_rgbs, gt_rgbs, harmonized_rgbs = [], [], [], [], []
    frames = data['frames'] 
    for img in sorted(os.listdir(os.path.join(DATA_DIR, "depths"))):
        depth = cv2.imread(os.path.join(DATA_DIR, "depths", img), -1)[:, :, 0] * 2048
        composite_image = cv2.imread(os.path.join(OUTPUT_DIR, "composite_scene", "images", img.replace('.exr', '.png')), -1) 
        composite_image = cv2.cvtColor(composite_image, cv2.COLOR_BGR2RGB)
        gt_image = cv2.imread(os.path.join(OUTPUT_DIR, "gt_scene", "images", img.replace('.exr', '.png')), -1) 
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        harmonized_image = cv2.imread(os.path.join(OUTPUT_DIR, "harmonized_scene", "images", img.replace('.exr', '.png')), -1) 
        harmonized_image = cv2.cvtColor(harmonized_image, cv2.COLOR_BGR2RGB)
        c2w = np.array(frames[int(img.split('.')[0])]["transform_matrix"])
        c2ws.append(c2w)
        depths.append(depth)
        composite_rgbs.append(composite_image)
        gt_rgbs.append(gt_image)
        harmonized_rgbs.append(harmonized_image)

    c2ws = np.stack(c2ws)
    depths = np.stack(depths)
    composite_rgbs = np.stack(composite_rgbs)
    gt_rgbs = np.stack(gt_rgbs)
    harmonized_rgbs = np.stack(harmonized_rgbs)

    depths = torch.from_numpy(depths).float()
    intrinsic = torch.from_numpy(intrinsic).float()
    c2ws = torch.from_numpy(c2ws).float()
    composite_rgbs = torch.from_numpy(composite_rgbs).float()
    gt_rgbs = torch.from_numpy(gt_rgbs).float()
    harmonized_rgbs = torch.from_numpy(harmonized_rgbs).float()

    # project to world
    all_points, composite_colors, gt_colors, harmonized_colors = [], [], [], []
    projs = []
    # Compute the pixel coordinates of each point in the depth image
    for i in range(depths.shape[0]):
        y, x = torch.meshgrid([torch.arange(0, h, dtype=torch.float32, device=depths.device),
                            torch.arange(0, w, dtype=torch.float32, device=depths.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(h * w), x.view(h * w)
        xyz = torch.stack((x, y, torch.ones_like(x)))
        depth_mask = depths[i] < 1024
        
        # Convert pixel coordinates to camera coordinates
        inv_K = torch.inverse(intrinsic)
        cam_coords1 = inv_K.clone() @ (xyz.clone() * depths[i].reshape(-1))
        cam_coords1[1,:] = -cam_coords1[1,:]
        cam_coords1[2,:] = -cam_coords1[2,:]
        world_coords = (c2ws[i] @ torch.cat([cam_coords1, torch.ones((1, cam_coords1.shape[1]))], dim=0)).T
        world_coords = world_coords[:,:3]
        depth_mask = depth_mask.reshape(-1)
        world_coords = world_coords[depth_mask]
        composite_color = composite_rgbs[i].reshape(-1,3)/255.
        composite_color = composite_color[depth_mask]
        gt_color = gt_rgbs[i].reshape(-1,3)/255.
        gt_color = gt_color[depth_mask]
        harmonized_color = harmonized_rgbs[i].reshape(-1,3)/255.
        harmonized_color = harmonized_color[depth_mask]
        
        all_points.append(world_coords)
        composite_colors.append(composite_color)
        gt_colors.append(gt_color)
        harmonized_colors.append(harmonized_color)

        for j in range(len(depth_mask)):
            if depth_mask[j]:
                projs.append([i,j])

    all_points = torch.cat(all_points, dim=0)
    composite_colors = torch.cat(composite_colors, dim=0) * 255
    gt_colors = torch.cat(gt_colors, dim=0) * 255
    harmonized_colors = torch.cat(harmonized_colors, dim=0) * 255
    projs = torch.tensor(projs)
    len_all = len(all_points)

    mask = [False] * len_all
    indices = np.linspace(0, len_all - 1, image_height * image_width, dtype=int)
    for index in indices:
        mask[index] = True
    mask = torch.tensor(mask)
    all_points = all_points[mask]
    sorted_indices = plot_hilbert_with_points_3d(tree, all_points)
    sorted_indices = torch.tensor(sorted_indices)

    all_points = all_points[sorted_indices]
    composite_colors = composite_colors[mask][sorted_indices]
    gt_colors = gt_colors[mask][sorted_indices]
    harmonized_colors = harmonized_colors[mask][sorted_indices]
    projs = projs[mask][sorted_indices]

    img2pcd = []
    for i, proj in enumerate(projs):
        uid = int(proj[0])
        index = int(proj[1])
        x = int(hilbert[i][0])
        y = int(hilbert[i][1])
        img2pcd.append([uid, index, x, y])
    img2pcd = np.stack(img2pcd).astype(np.int32)
    np.save(os.path.join(scene_path, "projs.npy"), img2pcd)

    storePly(composite_ply_path, all_points, composite_colors)
    storePly(gt_ply_path, all_points, gt_colors)
    storePly(harmonized_ply_path, all_points, harmonized_colors)

    map_colors_to_feat(os.path.join(scene_path, "composite.png"), hilbert, composite_colors, image_height, image_width)
    map_colors_to_feat(os.path.join(scene_path, "gt.png"), hilbert, gt_colors, image_height, image_width)
    map_colors_to_feat(os.path.join(scene_path, "harmonized.png"), hilbert, harmonized_colors, image_height, image_width)

def plot_hilbert_with_points_3d(tree, points):
    # return np.arange(len(points))

    # Generate random points
    np.random.seed(42)
    points = points.cpu().numpy()
    xyz = (points - points.min(axis=0)) / (points.max(axis=0) - points.min(axis=0))
    xyz = xyz * 32
    _, indices = tree.query(xyz)

    sorted_indices = np.argsort(indices)

    return sorted_indices

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

def hilbert_curve_2d_23(order, x0, y0, xi, xj, yi, yj, points):
    hilbert_curve_2d(order - 1, x0, y0, yi / 2, yj / 2, xi / 2, xj / 2, points)
    hilbert_curve_2d(order - 1, x0 + xi / 2, y0 + xj / 2, xi / 2, xj / 2, yi / 2, yj / 2, points)
    hilbert_curve_2d(order - 1, x0 + xi / 2 + yi / 2, y0 + xj / 2 + yj / 2, xi / 2, xj / 2, yi / 2, yj / 2, points)
    hilbert_curve_2d(order - 1, x0 + xi / 2 + yi, y0 + xj / 2 + yj, -yi / 2, -yj / 2, -xi / 2, -xj / 2, points)
    hilbert_curve_2d(order - 1, x0 + yi, y0 + yj, yi / 2, yj / 2, xi / 2, xj / 2, points)
    hilbert_curve_2d(order - 1, x0 + xi /2 + yi, y0 + xj /2 + yj, xi / 2, xj / 2, yi / 2, yj / 2, points)

def map_colors_to_feat(save_path, hilbert, colors, image_height, image_width):
    rgb = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    color = colors.cpu().numpy()
    for i, c in enumerate(color):
        if i < len(hilbert):
            x, y = hilbert[i]
            rgb[int(x), int(y)] = c
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, rgb)
    
def main():
    order = 11
    image_height, image_width = 2048, 2048 
    hilbert = []
    hilbert_curve_2d(order, 0, 0, 2**order, 0, 0, 2**order, hilbert)

    # Generate Hilbert curve points
    hilbert3d = np.load("./checkpoints/hilbert3d_order9.npy", allow_pickle=True)
    tree = KDTree(hilbert3d)

    process_scene(tree, hilbert, image_height, image_width)

if __name__ == "__main__":
    main()