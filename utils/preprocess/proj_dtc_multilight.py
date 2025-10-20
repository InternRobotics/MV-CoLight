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

DATA_DIR = "/path/to/blender_renderings"
INPUT_DIR = "/path/to/dtc_multi_light_2d"
OUTPUT_DIR = "/path/to/dtc_multi_light_3d"

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

def process_train_scene(scene, tree, hilbert, image_height, image_width):
    last_lcd = sorted(os.listdir(os.path.join(INPUT_DIR, scene)))[-1]
    last_object_id = int(sorted(os.listdir(os.path.join(INPUT_DIR, scene, last_lcd, "object_bgs")))[-1].split("_object")[-1].split(".")[0])
    num_objects = last_object_id + 1
    im_path = os.path.join(OUTPUT_DIR, scene, last_lcd, f"mask{last_object_id}.png")
    if os.path.exists(im_path):
        print("Scene "+scene+" skip")
        return 
    tmp_path = os.path.join(OUTPUT_DIR, scene, "tmp")
    if os.path.exists(tmp_path):
        print("Scene "+scene+" is being processed")
        return 
    os.makedirs(tmp_path)
    scene_path = os.path.join(DATA_DIR, scene)
    save_path = os.path.join(OUTPUT_DIR, scene)
    with open(os.path.join(scene_path, os.listdir(scene_path)[0], f'transforms.json'), "r") as f:
        data = json.load(f)
    w=256
    h=256
    fovx = data["camera_angle_x"]
    fx = fov2focal(fovx, 384)
    fy = fx
    cx = w / 2
    cy = h / 2

    intrinsic = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
    depths, c2ws, env_masks, object_masks = [], [], [], []
    for i in range(num_objects):
        object_masks.append([])
    frames = data['frames'] 
    for frame in frames:
        depth = cv2.imread(os.path.join(DATA_DIR, frame["depth_path"]), -1)[::2,128:640:2,0:1]
        c2w = np.array(frame["transform_matrix"])
        env_mask = torch.from_numpy(cv2.imread(os.path.join(DATA_DIR, frame["env_mask_path"]), -1))[::2,128:640:2]
        for i in range(num_objects):
            object_masks[i].append(torch.from_numpy(cv2.imread(os.path.join(DATA_DIR, frame["object_mask_path"][i]), -1))[::2,128:640:2])
        c2ws.append(c2w)
        depths.append(depth)
        env_masks.append(env_mask)

    c2ws = np.stack(c2ws)
    depths = np.stack(depths)
    env_masks = np.stack(env_masks) 
    for i in range(num_objects):
        object_masks[i] = np.stack(object_masks[i])

    depths = torch.from_numpy(depths).float()
    intrinsic = torch.from_numpy(intrinsic).float()
    c2ws = torch.from_numpy(c2ws).float()
    env_masks = torch.from_numpy(env_masks).float()
    for i in range(num_objects):
        object_masks[i] = torch.from_numpy(object_masks[i]).float()

    # project to world
    world_coords_list = []
    # Compute the pixel coordinates of each point in the depth image
    for i in range(depths.shape[0]):
        y, x = torch.meshgrid([torch.arange(0, h, dtype=torch.float32, device=depths.device),
                            torch.arange(0, w, dtype=torch.float32, device=depths.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(h * w), x.view(h * w)
        xyz = torch.stack((x, y, torch.ones_like(x)))
        
        # Convert pixel coordinates to camera coordinates
        inv_K = torch.inverse(intrinsic)
        cam_coords1 = inv_K.clone() @ (xyz.clone() * depths[i].reshape(-1))
        cam_coords1[1,:] = -cam_coords1[1,:]
        cam_coords1[2,:] = -cam_coords1[2,:]
        world_coords = (c2ws[i] @ torch.cat([cam_coords1, torch.ones((1, cam_coords1.shape[1]))], dim=0)).T
        world_coords = world_coords[:,:3]
        world_coords_list.append(world_coords)

    all_points = []
    projs = []
    len_object = []
    for object_mask in object_masks:
        for i, world_coords in enumerate(world_coords_list):
            depth_mask = env_masks[i] == 0
            depth_mask[object_mask[i]==0] = False
            depth_mask = depth_mask.reshape(-1)
            for j in range(len(depth_mask)):
                if depth_mask[j]:
                    projs.append([i,j])
            all_points.append(world_coords[depth_mask])

        len_object.append(len(torch.cat(all_points, dim=0)))

    for i, world_coords in enumerate(world_coords_list):
        depth_mask = env_masks[i] == 0
        for object_mask in object_masks:
            depth_mask[object_mask[i] > 0] = False
        depth_mask = depth_mask.reshape(-1)
        for j in range(len(depth_mask)):
            if depth_mask[j]:
                projs.append([i,j])
        all_points.append(world_coords[depth_mask])

    all_points = torch.cat(all_points, dim=0)
    projs = torch.tensor(projs)
    len_all = len(all_points)

    OBJ_MASK = []
    cur_index = 0
    for i in range(len(len_object)):
        obj_mask = torch.zeros(len_all, dtype=torch.bool)
        obj_mask[cur_index:cur_index+len_object[i]] = True
        cur_index += len_object[i]
        OBJ_MASK.append(obj_mask)

    mask = [False] * len_all
    indices = np.linspace(0, len_all - 1, image_height*image_width, dtype=int)
    for index in indices:
        mask[index] = True
    mask = torch.tensor(mask)

    all_points = all_points[mask]
    sorted_indices = plot_hilbert_with_points_3d(tree, all_points)
    sorted_indices = torch.tensor(sorted_indices)
    all_points = all_points[sorted_indices]

    for i in range(num_objects):
        OBJ_MASK[i] = OBJ_MASK[i][mask][sorted_indices]

    projs = projs[mask][sorted_indices]
    img2pcd = []
    for i, proj in enumerate(projs):
        uid = int(proj[0])
        index = int(proj[1])
        x = int(hilbert[i][0])
        y = int(hilbert[i][1])
        img2pcd.append([uid, index, x, y])
    img2pcd = np.stack(img2pcd).astype(np.int32)
    np.save(os.path.join(save_path, "projs.npy"), img2pcd)

    all_colors = {}
    all_bg_colors = []
    for i in range(num_objects):
        all_bg_colors.append({})
    for lcd in os.listdir(scene_path):
        if not os.path.isdir(os.path.join(scene_path, lcd)):
            continue
        if lcd != "base":
            input_path = os.path.join(scene_path, lcd)
            with open(os.path.join(input_path, f'transforms.json'), "r") as f:
                data = json.load(f)

            ev0_rgbs, ev1_rgbs, ev2_rgbs, ev3_rgbs = [], [], [], []
            object_ev0_bgs, object_ev1_bgs, object_ev2_bgs, object_ev3_bgs = [], [], [], []
            for i in range(num_objects):
                object_ev0_bgs.append([])
                object_ev1_bgs.append([])
                object_ev2_bgs.append([])
                object_ev3_bgs.append([])
            frames = data['frames']
            for frame in frames:
                ev0_rgbs.append(cv2.imread(os.path.join(INPUT_DIR, scene, lcd+"_ev0", "images", frame["image_path"].split('/')[-1].split('.')[0]+".png"), -1))
                ev1_rgbs.append(cv2.imread(os.path.join(INPUT_DIR, scene, lcd+"_ev1", "images", frame["image_path"].split('/')[-1].split('.')[0]+".png"), -1))
                ev2_rgbs.append(cv2.imread(os.path.join(INPUT_DIR, scene, lcd+"_ev2", "images", frame["image_path"].split('/')[-1].split('.')[0]+".png"), -1))
                ev3_rgbs.append(cv2.imread(os.path.join(INPUT_DIR, scene, lcd+"_ev3", "images", frame["image_path"].split('/')[-1].split('.')[0]+".png"), -1))
                for i in range(num_objects):
                    object_ev0_bgs[i].append(cv2.imread(os.path.join(INPUT_DIR, scene, lcd+"_ev0", "object_bgs", frame["object_bg_path"][i].split('/')[-1].split('.')[0]+".png"), -1))
                    object_ev1_bgs[i].append(cv2.imread(os.path.join(INPUT_DIR, scene, lcd+"_ev1", "object_bgs", frame["object_bg_path"][i].split('/')[-1].split('.')[0]+".png"), -1))
                    object_ev2_bgs[i].append(cv2.imread(os.path.join(INPUT_DIR, scene, lcd+"_ev2", "object_bgs", frame["object_bg_path"][i].split('/')[-1].split('.')[0]+".png"), -1))
                    object_ev3_bgs[i].append(cv2.imread(os.path.join(INPUT_DIR, scene, lcd+"_ev3", "object_bgs", frame["object_bg_path"][i].split('/')[-1].split('.')[0]+".png"), -1))

            ev0_rgbs = torch.from_numpy(np.stack(ev0_rgbs)).float()
            ev1_rgbs = torch.from_numpy(np.stack(ev1_rgbs)).float()
            ev2_rgbs = torch.from_numpy(np.stack(ev2_rgbs)).float()
            ev3_rgbs = torch.from_numpy(np.stack(ev3_rgbs)).float()
            for i in range(num_objects):
                object_ev0_bgs[i] = torch.from_numpy(np.stack(object_ev0_bgs[i])).float()
                object_ev1_bgs[i] = torch.from_numpy(np.stack(object_ev1_bgs[i])).float()
                object_ev2_bgs[i] = torch.from_numpy(np.stack(object_ev2_bgs[i])).float()
                object_ev3_bgs[i] = torch.from_numpy(np.stack(object_ev3_bgs[i])).float()

            # project to world
            all_colors[lcd] = {
                "ev0": rgb2color(ev0_rgbs, object_masks, env_masks)[mask][sorted_indices],
                "ev1": rgb2color(ev1_rgbs, object_masks, env_masks)[mask][sorted_indices],
                "ev2": rgb2color(ev2_rgbs, object_masks, env_masks)[mask][sorted_indices],
                "ev3": rgb2color(ev3_rgbs, object_masks, env_masks)[mask][sorted_indices],
            }

            for i in range(num_objects):
                all_bg_colors[i][lcd] = {
                    "ev0": rgb2color(object_ev0_bgs[i], object_masks, env_masks)[mask][sorted_indices],
                    "ev1": rgb2color(object_ev1_bgs[i], object_masks, env_masks)[mask][sorted_indices],
                    "ev2": rgb2color(object_ev2_bgs[i], object_masks, env_masks)[mask][sorted_indices],
                    "ev3": rgb2color(object_ev3_bgs[i], object_masks, env_masks)[mask][sorted_indices],
                }

    torch.save(all_colors, f"{save_path}/colors.pt")
    torch.save(all_points, f"{save_path}/points.pt")
    map_colors_to_feat(scene_path, save_path, hilbert, all_colors, image_height, image_width)
    map_bgs_to_feat(scene_path, save_path, hilbert, all_bg_colors, OBJ_MASK, image_height, image_width)
    map_masks_to_feat(scene_path, save_path, hilbert, OBJ_MASK, image_height, image_width)
    shutil.rmtree(tmp_path)

def plot_hilbert_with_points_3d(tree, points):
    # return np.arange(len(points))

    # Generate random points
    np.random.seed(42)
    points = points.cpu().numpy()
    xyz = (points - points.min(axis=0)) / (points.max(axis=0) - points.min(axis=0))
    xyz = xyz * 16
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

def map_colors_to_feat(scene_path, save_path, hilbert, colors, image_height, image_width):
    for lcd in sorted(os.listdir(scene_path)):
        if not os.path.isdir(os.path.join(scene_path, lcd)):
            continue
        if lcd == "base":
            continue
        for ev in ["ev0", "ev1", "ev2", "ev3"]:
            rgb = np.zeros((image_height, image_width, 3), dtype=np.uint8)
            color = colors[lcd][ev].cpu().numpy()
            for i, c in enumerate(color):
                if i < len(hilbert):
                    x, y = hilbert[i]
                    rgb[int(x), int(y)] = c
            os.makedirs(os.path.join(save_path, lcd+"_"+ev), exist_ok=True)
            cv2.imwrite(os.path.join(save_path, lcd+"_"+ev, "img.png"), rgb)

def map_masks_to_feat(scene_path, save_path, hilbert, object_masks, image_height, image_width):
    for lcd in sorted(os.listdir(scene_path)):
        if not os.path.isdir(os.path.join(scene_path, lcd)):
            continue
        if lcd == "base":
            continue
        for idx, object_mask in enumerate(object_masks):
            mask = np.zeros((image_height, image_width), dtype=np.uint8)
            for i in range(len(object_mask)):
                if i < len(hilbert):
                    x, y = hilbert[i]
                    if object_mask[i]:
                        mask[int(x), int(y)] = 255
            for ev in ["ev0", "ev1", "ev2", "ev3"]:
                os.makedirs(os.path.join(save_path, lcd+"_"+ev), exist_ok=True)
                Image.fromarray(mask).save(f"{save_path}/{lcd}_{ev}/mask{idx}.png")

def map_bgs_to_feat(scene_path, save_path, hilbert, bgs, object_masks, image_height, image_width):
    for lcd in sorted(os.listdir(scene_path)):
        if not os.path.isdir(os.path.join(scene_path, lcd)):
            continue
        if lcd == "base":
            continue
        for ev in ["ev0", "ev1", "ev2", "ev3"]:
            os.makedirs(os.path.join(save_path, lcd+"_"+ev), exist_ok=True)
            for idx in range(len(bgs)):
                background = np.zeros((image_height, image_width, 3), dtype=np.uint8)
                bg = bgs[idx][lcd][ev].cpu().numpy()
                for i, b in enumerate(bg):
                    if i < len(hilbert):
                        x, y = hilbert[i]
                        background[int(x), int(y)] = b
                        if object_masks[idx][i]:
                            background[int(x), int(y)] = [127, 127, 127]
                cv2.imwrite(os.path.join(save_path, lcd+"_"+ev, f"bg{idx}.png"), background)

def main():
    order = 9
    image_height, image_width = 512, 512
    hilbert = []
    hilbert_curve_2d(order, 0, 0, 2**order, 0, 0, 2**order, hilbert)

    # Generate Hilbert curve points
    hilbert3d = np.load("./checkpoints/hilbert3d_order9.npy", allow_pickle=True)
    tree = KDTree(hilbert3d)

    for scene in sorted(os.listdir(DATA_DIR)):
        process_train_scene(scene, tree, hilbert, image_height, image_width)

if __name__ == "__main__":
    main()