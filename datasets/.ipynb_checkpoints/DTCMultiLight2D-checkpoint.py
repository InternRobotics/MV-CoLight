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
    get_image_size,
    load_image, 
    load_mask, 
    load_depth,
    average_camera_poses,
)

@DATASETS.register_module("DTCMultiLight2D")
class DTCMultiLightDataset(Dataset):
    def __init__(
        self,
        data_root,
        data_json, 
        split="train",
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
        
        data_dict = json.load(open(self.data_json, 'r'))
        data_list = []
        for data_id in data_dict.keys():
            dataset = data_dict[data_id]
            for scene in dataset["scenes"]:
                data_list.append((data_id, os.path.join(dataset["data_root"], scene)))
        self.org_image_size, self.image_type = get_image_size(data_list[0][1])

        self.scene_ids = {}
        self.scenes = {}
        index = 0
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(self.load_jsons, data_id, scene_path) for data_id, scene_path in data_list]
            for future in as_completed(futures):
                scene_frames, scene_id, num_cd = future.result()
                self.scenes[scene_id] = scene_frames
                for i in range(num_cd):
                    self.scene_ids[index] = (scene_id, i)
                    index += 1
    def load_jsons(self, data_id, scene_path):
        scene_frames = []
        scene_id = data_id+"/"+os.path.basename(scene_path)
        data_root = os.path.dirname(scene_path)
        for light_cd in sorted(os.listdir(scene_path)):
            light_path = os.path.join(scene_path, light_cd)
            if not os.path.isdir(light_path):
                continue
        
            json_path = os.path.join(light_path, "transforms.json")
            with open(json_path, "r") as f:
                try:
                    data = json.load(f)
                except:
                    print(json_path)

            image_paths, depth_paths, env_mask_paths, object_mask_paths, object_bg_paths = [], [], [], [], []
            for i, frame in enumerate(data["frames"]):
                image_paths.append(os.path.join(data_root, frame["image_path"]))
                depth_paths.append(os.path.join(data_root, frame["depth_path"]))
                env_mask_paths.append(os.path.join(data_root, frame["env_mask_path"]))
                object_mask_paths.append([os.path.join(data_root,f) for f in frame["object_mask_path"]])
                object_bg_paths.append([os.path.join(data_root,f) for f in frame["object_bg_path"]])

            tmp_frame = {}
            tmp_frame["image_path"] = image_paths
            tmp_frame["depth_path"] = depth_paths
            tmp_frame["env_mask_path"] = env_mask_paths
            tmp_frame["object_mask_path"] = object_mask_paths
            tmp_frame["object_bg_path"] = object_bg_paths

            if self.image_type == "exr":
                for ev in self.tonemap_evs:
                    tmp_frame["ev"] = ev
                    scene_frames.append(tmp_frame)
            else:
                scene_frames.append(tmp_frame)

        return scene_frames, scene_id, len(scene_frames)

    def load_input_frames(self, input_object_bg_paths, input_object_mask_paths, insert_image_paths, base_ev, insert_ev):
        def load_frame(index, input_object_bg_path, input_object_mask_path, insert_image_path, base_ev, insert_ev):
            bg_image = load_image(input_object_bg_path, base_ev)
            insert_image = load_image(insert_image_path, insert_ev)
            mask = load_mask(input_object_mask_path)
            image = insert_image * mask + bg_image * (1 - mask)
            return index, image, bg_image

        image_list = [None] * len(input_object_bg_paths)
        bg_image_list = [None] * len(input_object_bg_paths)
        for idx, (bg_path, mask_path, insert_path) in enumerate(zip(input_object_bg_paths, input_object_mask_paths, insert_image_paths)):
            index, image, bg_image = load_frame(idx, bg_path, mask_path, insert_path, base_ev, insert_ev)
            image_list[index] = image
            bg_image_list[index] = bg_image
        images = torch.stack(image_list)
        bg_images = torch.stack(bg_image_list)

        return images, bg_images

    def load_input_depths(self, input_depth_paths):
        def load_frame(index, input_depth_path):
            depth = load_depth(input_depth_path)
            return index, depth

        depth_list = [None] * len(input_depth_paths)
        for idx, depth_path in enumerate(input_depth_paths):
            index, depth = load_frame(idx, depth_path)
            depth_list[index] = depth
        depths = torch.stack(depth_list)

        return depths

    def load_target_frames(self, target_image_paths, base_ev):
        def load_frame(index, image_path, base_ev):
            image = load_image(image_path, base_ev)
            return index, image

        image_list = [None] * len(target_image_paths)
        for idx, image_path in enumerate(target_image_paths):
            index, image = load_frame(idx, image_path, base_ev)
            image_list[index] = image
        images = torch.stack(image_list)
        
        return images

    def getitem(self, index):
        scene, base_index = self.scene_ids[index]
        base_example = self.scenes[scene][base_index]
        insert_index = np.random.randint(0, len(self.scenes[scene]))
        insert_example = self.scenes[scene][insert_index]
        base_ev = base_example["ev"] if "ev" in base_example else None
        insert_ev = insert_example["ev"] if "ev" in insert_example else None

        random_indices = torch.randperm(len(base_example['image_path']))
        context_indices = random_indices[:self.num_context_views]
        target_indices = random_indices[:self.num_target_views]
        num_object = len(base_example["object_bg_path"][0])
        insert_object_id = np.random.randint(0, num_object)
        
        # load images
        input_object_bg_paths = [base_example["object_bg_path"][i][insert_object_id] for i in context_indices]
        input_object_mask_paths = [base_example["object_mask_path"][i][insert_object_id] for i in context_indices]
        input_depth_paths = [base_example["depth_path"][i] for i in context_indices]
        insert_image_paths = [insert_example["image_path"][i] for i in context_indices]
        target_image_paths = [base_example["image_path"][i] for i in target_indices]
        context_images, context_bg_images = self.load_input_frames(input_object_bg_paths, input_object_mask_paths, insert_image_paths, base_ev, insert_ev)
        context_depths = self.load_input_depths(input_depth_paths)
        target_images = self.load_target_frames(target_image_paths, base_ev)
        
        org_image_size = [target_images.shape[2], target_images.shape[3]] 
        if self.image_size != org_image_size:
            context_images = rescale_and_crop(context_images, org_image_size, self.image_size)
            context_bg_images = rescale_and_crop(context_bg_images, org_image_size, self.image_size)
            context_depths = rescale_and_crop(context_depths, org_image_size, self.image_size)
            target_images = rescale_and_crop(target_images, org_image_size, self.image_size)

        if self.add_background:
            context_images = torch.cat([context_images, context_bg_images], dim=1)
        if self.add_depth:
            context_images = torch.cat([context_images, context_depths], dim=1)

        example = {
            "input_images": context_images, 
            "input_index": context_indices,
            "target_images": target_images,
            "target_index": target_indices,
            "scene": scene,
        }

        return example

    def __len__(self):
        return len(self.scene_ids)

    def __getitem__(self, index):
        return self.getitem(index)