import json
import os
import sys
import math
from io import BytesIO
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import numpy as np
import torch
import torchvision.transforms as tf
import torchvision.transforms.functional as F
from einops import rearrange, repeat
from registry import DATASETS
from mmengine.config import Config
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets.utils import (
    rescale_and_crop,
    load_image, 
    load_mask, 
    load_estimate_depth,
    average_camera_poses,
)

def get_image_size(data_path):
    image_path = os.path.join(data_path, os.listdir(data_path)[0])
    if image_path.endswith(".npy"):
        image = np.load(image_path, allow_pickle=True)
        return image.shape[1:], image_path.split(".")[-1]
    else:
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        return image.shape[:2], image_path.split(".")[-1]


@DATASETS.register_module("RelightingObjectsEval")
class RelightingObjectsEvalDataset(Dataset):
    def __init__(
        self,
        data_json, 
        num_context_views=1,
        num_target_views=1,
        image_size=[256, 256],
        tonemap_evs=[0, 1, 2, 3],
        add_background=True,
        add_depth=True,
    ):
        super().__init__()
        self.data_root = data_root
        self.data_json = data_json
        self.stage = split
        self.num_context_views = num_context_views
        self.num_target_views = num_target_views
        self.image_size = image_size
        self.tonemap_evs = tonemap_evs
        self.add_background = add_background
        self.add_depth = add_depth
        
        data_dict = json.load(open(self.data_json, 'r'))["relighting_objects"]
        data_list = [os.path.join(data_dict["data_root"], item) for item in data_dict["scenes"]]
        self.org_image_size, self.image_type = get_image_size(self.data_list[0])

        self.scene_ids = {}
        self.scenes = {}
        index = 0
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(self.load_jsons, scene_path) for scene_path in self.data_list]
            for future in as_completed(futures):
                scene_frames, scene_id = future.result()
                self.scenes[scene_id] = scene_frames
                self.scene_ids[index] = scene_id
                index += 1
                
    def load_jsons(self, scene_path):
        scene_id = os.path.basename(scene_path)
        json_path = os.path.join(scene_path, "transforms.json")
        with open(json_path, "r") as f:
            data = json.load(f)

        return data, scene_id

    def getitem(self, index):
        scene = self.scene_ids[index]
        example = self.scenes[scene]
        
        # load images
        context_image = load_image(os.path.join(self.data_root, example["image_path"])).unsqueeze(0)
        context_bg_image = load_image(os.path.join(self.data_root, example["gt_path"])).unsqueeze(0)
        context_depth = load_estimate_depth(load_image(os.path.join(self.data_root, example["image_path"]))).unsqueeze(0)
        target_image = load_image(os.path.join(self.data_root, example["gt_path"])).unsqueeze(0)
        context_indice = torch.tensor([0])
        target_indice = torch.tensor([0])
    
        org_image_size = [target_image.shape[2], target_image.shape[3]] 
        print(org_image_size)
        if self.image_size != org_image_size:
            
            context_image = rescale_and_crop(context_image, org_image_size, self.image_size)
            context_bg_image = rescale_and_crop(context_bg_image, org_image_size, self.image_size)
            context_depth = rescale_and_crop(context_depth, org_image_size, self.image_size)
            target_image = rescale_and_crop(target_image, org_image_size, self.image_size)

        if self.add_background:
            context_image = torch.cat([context_image, context_bg_image], dim=1)
        if self.add_depth:
            context_image = torch.cat([context_image, context_depth], dim=1)

        example = {
            "input_images": context_image, 
            "input_index": context_indice,
            "target_images": target_image,
            "target_index": target_indice,
            "scene": scene,
        }

        return example

    def __len__(self):
        return len(self.scene_ids)

    def __getitem__(self, index):
        return self.getitem(index)