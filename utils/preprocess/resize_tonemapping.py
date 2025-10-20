import numpy as np
from plyfile import PlyData, PlyElement
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import cv2
import os
from concurrent.futures import ThreadPoolExecutor
import torch
import argparse
import json
import os
# load EXR
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

def reinhard_tonemap(H, maxH):
    H = H*(1+H/(maxH*maxH))/(1+H)
    return H

def get_ev(image_dir, env_mask_dir):
    log_mean_list = []
    max_H = 0
    for image_name in os.listdir(image_dir):
        img = cv2.imread(os.path.join(image_dir, image_name), -1)[:, :, :3]
        env_mask = cv2.imread(os.path.join(env_mask_dir, image_name.replace('.exr', '.png')), -1)
        env_mask = (env_mask == 0)[:, :, np.newaxis]
        env_mask = np.concatenate([env_mask, env_mask, env_mask], axis=2)
        env_mask[img<=0] = False
        max_H = max(max_H, np.max(img[env_mask]))
        log_mean_list.append(np.log(img[env_mask]).mean())
    ev = - np.array(log_mean_list).mean() - 2.0
    return ev, max_H

def get_max_D(depth_dir, env_mask_dir):
    max_D = 0.0
    for depth_name in os.listdir(depth_dir):
        depth = cv2.imread(os.path.join(depth_dir, depth_name), -1)[:, :, 0]
        env_mask = cv2.imread(os.path.join(env_mask_dir, depth_name.replace('.exr', '.png')), -1)
        max_D = max(max_D, depth[env_mask==0].max())
    return max_D

def save_images(image_dir, scene, lcd, ev, max_H):
    for i in range(4):
        ev = ev + 0.5
        save_path = os.path.join(OUTPUT_DIR, scene, lcd+"_ev{:d}".format(i))
        for image_name in os.listdir(image_dir):
            img = cv2.imread(os.path.join(image_dir, image_name), -1)[:, :, :3]
            H, W = img.shape[:2]
            hdr_img = img * np.exp(ev)
            result = np.zeros_like(hdr_img)
            for i in range(3):  # Process each channel separately
                result[:, :, i] = reinhard_tonemap(hdr_img[:, :, i].astype(np.float64), max_H * np.exp(ev))
            result = np.clip(result * 255, 0, 255).astype(np.uint8)
            img = cv2.resize(result, (W //2, H //2), interpolation=cv2.INTER_AREA)
            img = img[:, (W-H) // 4: (W+H) // 4, :]
            os.makedirs(os.path.join(save_path, "images"), exist_ok=True)
            cv2.imwrite(os.path.join(save_path, "images", image_name.replace('.exr', '.png')), img)

def save_object_bgs(image_dir, scene, lcd, ev, max_H):
    for i in range(4):
        ev = ev + 0.5
        save_path = os.path.join(OUTPUT_DIR, scene, lcd+"_ev{:d}".format(i))
        for image_name in os.listdir(image_dir):
            img = cv2.imread(os.path.join(image_dir, image_name), -1)[:, :, :3]
            H, W = img.shape[:2]
            hdr_img = img * np.exp(ev)
            result = np.zeros_like(hdr_img)
            for i in range(3):  # Process each channel separately
                result[:, :, i] = reinhard_tonemap(hdr_img[:, :, i].astype(np.float64), max_H * np.exp(ev))
            result = np.clip(result * 255, 0, 255).astype(np.uint8)
            img = cv2.resize(result, (W //2, H //2), interpolation=cv2.INTER_AREA)
            img = img[:, (W-H) // 4: (W+H) // 4, :]
            os.makedirs(os.path.join(save_path, "object_bgs"), exist_ok=True)
            cv2.imwrite(os.path.join(save_path, "object_bgs", image_name.replace('.exr', '.png')), img)

def save_depths(image_dir, env_mask_dir, scene, lcd, max_D):
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        env_mask_path = os.path.join(env_mask_dir, image_name.replace('.exr', '.png'))
        img = cv2.imread(image_path, -1)[:, :, 0]
        mask = cv2.imread(env_mask_path, -1)
        img[mask == 0] /= max_D
        img[mask !=0 ] = 1.0
        H, W = img.shape[:2]
        img[img>1] = 1
        img[img<0] = 0
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        img = cv2.resize(img, (W //2, H //2), interpolation=cv2.INTER_AREA)
        img = img[:, (W-H) // 4: (W+H) // 4]
        for i in range(0, 4):
            save_path = os.path.join(OUTPUT_DIR, scene, lcd+"_ev{:d}".format(i))
            os.makedirs(os.path.join(save_path, 'depths'), exist_ok=True)
            cv2.imwrite(os.path.join(save_path, 'depths', image_name.replace('.exr', '.png')), img)

def save_env_masks(image_dir, scene, lcd):
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        img = cv2.imread(image_path, -1) / 255.
        H, W = img.shape[:2]
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        img = cv2.resize(img, (W //2, H //2), interpolation=cv2.INTER_AREA)
        img = img[:, (W-H) // 4: (W+H) // 4]
        for i in range(0, 4):
            save_path = os.path.join(OUTPUT_DIR, scene, lcd+"_ev{:d}".format(i))
            os.makedirs(os.path.join(save_path, 'env_masks'), exist_ok=True)
            cv2.imwrite(os.path.join(save_path, 'env_masks', image_name), img)

def save_object_masks(image_dir, scene, lcd):
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        img = cv2.imread(image_path, -1) / 255.
        H, W = img.shape[:2]
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        img = cv2.resize(img, (W //2, H //2), interpolation=cv2.INTER_AREA)
        img = img[:, (W-H) // 4: (W+H) // 4]
        for i in range(0, 4):
            save_path = os.path.join(OUTPUT_DIR, scene, lcd+"_ev{:d}".format(i))
            os.makedirs(os.path.join(save_path, 'object_masks'), exist_ok=True)
            cv2.imwrite(os.path.join(save_path, 'object_masks', image_name), img)

DATA_DIR = "/path/to/blender_renderings"
OUTPUT_DIR = "/path/to/dtc_multi_light_2d"
for scene in os.listdir(DATA_DIR):
    scene_path = os.path.join(DATA_DIR, scene)
    for lcd in sorted(os.listdir(scene_path)):
        if not os.path.isdir(os.path.join(scene_path, lcd)):
            continue
        image_dir = os.path.join(scene_path, lcd, "images")
        depth_dir = os.path.join(scene_path, lcd, "depths")
        env_mask_dir = os.path.join(scene_path, lcd, "env_masks")
        object_bg_dir = os.path.join(scene_path, lcd, "object_bgs")
        object_mask_dir = os.path.join(scene_path, lcd, "object_masks")
        ev, max_H = get_ev(image_dir, env_mask_dir)
        max_D = get_max_D(depth_dir, env_mask_dir)
        save_images(image_dir, scene, lcd, ev, max_H)
        save_object_bgs(object_bg_dir, scene, lcd, ev, max_H)
        save_depths(depth_dir, env_mask_dir, scene, lcd, max_D)
        save_env_masks(env_mask_dir, scene, lcd)
        save_object_masks(object_mask_dir, scene, lcd)