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
    load_gaus,
    adjust_intrinsics,
    average_camera_poses,
)

@DATASETS.register_module("DTCMultiLight3D")
class DTCMultiLightDataset(Dataset):
    def __init__(
        self,
        stage, 
        data_json, 
        num_render_views=4,
        image_size=[256, 256],
        tonemap_evs=[0, 1, 2, 3],
        add_background=True,
        add_depth=True,
        render_2d=True
    ):
        super().__init__()
        self.stage = stage
        self.num_render_views = num_render_views
        self.image_size = image_size
        self.tonemap_evs = tonemap_evs
        self.add_background = add_background
        self.add_depth = add_depth
        self.render_2d = render_2d

        data_dict = json.load(open(data_json, 'r'))
        data_list = []
        for data_id in data_dict.keys():
            dataset = data_dict[data_id]
            for scene in dataset["scenes"]:
                data_list.append((data_id, os.path.join(dataset["data_root_2d"], scene), os.path.join(dataset["data_root_3d"], scene)))
        self.org_image_size, self.image_type = get_image_size(data_list[0][1])

        self.scene_ids = {}
        self.scenes = {}
        index = 0
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(self.load_jsons, data_id, scene_2d_path, scene_3d_path) for data_id, scene_2d_path, scene_3d_path in data_list]
            for future in as_completed(futures):
                scene_frames, scene_id, num_cd, flag = future.result()
                self.scenes[scene_id] = scene_frames
                for i in range(num_cd):
                    self.scene_ids[index] = (scene_id, i, flag)
                    index += 1

    def load_jsons(self, data_id, scene_2d_path, scene_3d_path):
        scene_id = data_id+"/"+os.path.basename(scene_2d_path)
        if os.path.exists(os.path.join(scene_2d_path, "transforms.json")):
            frame, num_cd, flag = self.load_test_json(scene_2d_path, scene_3d_path)
        else:
            frame, num_cd, flag = self.load_train_json(scene_2d_path, scene_3d_path)
        return frame, scene_id, num_cd, flag

    def load_test_json(self, scene_2d_path, scene_3d_path):
        json_2d_path = os.path.join(scene_2d_path, "transforms.json")
        data_root_2d = os.path.dirname(scene_2d_path)
        with open(json_2d_path, "r") as f:
            try:
                data_2d = json.load(f)
            except:
                print(json_2d_path)

        composite_paths, depth_paths, gt_paths, bg_paths, mask_paths, intrinsics, extrinsics = [], [], [], [], [], [], []
        for i, frame in enumerate(data_2d["frames"]):
            composite_paths.append(os.path.join(data_root_2d, frame["composite_path"]))
            bg_paths.append(os.path.join(data_root_2d, frame["bg_path"]))
            depth_paths.append(os.path.join(data_root_2d, frame["depth_path"]))
            gt_paths.append(os.path.join(data_root_2d, frame["gt_path"]))
            mask_paths.append(os.path.join(data_root_2d, frame["mask_path"]))
            try:
                if "camera_angle_x" in data_2d:
                    intrinsics.append(self.convert_intrinsics(data_2d["camera_angle_x"], self.org_image_size))
                else:
                    intrinsics.append(self.convert_intrinsics_from_focal(frame["fl_x"], self.org_image_size))
            except:
                print(json_2d_path)
            extrinsics.append(self.convert_extrinsics(frame["transform_matrix"]))

        tmp_frame = {}
        tmp_frame["composite_path"] = composite_paths
        tmp_frame["depth_path"] = depth_paths
        tmp_frame["gt_path"] = gt_paths
        tmp_frame["bg_path"] = bg_paths
        tmp_frame["mask_path"] = mask_paths
        tmp_frame["intrinsics"] = intrinsics
        tmp_frame["extrinsics"] = extrinsics
        tmp_frame["rgb_composite_path"] = os.path.join(scene_3d_path, "composite.png")
        tmp_frame["rgb_gt_path"] = os.path.join(scene_3d_path, "gt.png")
        tmp_frame["gs_path"] = os.path.join(scene_3d_path, "gs.npy")
        tmp_frame["proj_path"] = os.path.join(scene_3d_path, "projs.npy")
        
        return tmp_frame, 1, True

    def load_train_json(self, scene_2d_path, scene_3d_path):
        scene_frames = []
        data_root_2d = os.path.dirname(scene_2d_path)
        data_root_3d = os.path.dirname(scene_3d_path)
        for light_cd in sorted(os.listdir(scene_2d_path)):
            light_2d_path = os.path.join(scene_2d_path, light_cd)
            if not os.path.isdir(light_2d_path):
                continue
            
            json_2d_path = os.path.join(light_2d_path, "transforms.json")
            with open(json_2d_path, "r") as f:
                data_2d = json.load(f)
            
            image_paths, depth_paths, env_mask_paths, object_mask_paths, object_bg_paths, intrinsics, extrinsics = [], [], [], [], [], [], []
            for i, frame in enumerate(data_2d["frames"]):
                image_paths.append(os.path.join(data_root_2d, frame["image_path"]))
                depth_paths.append(os.path.join(data_root_2d, frame["depth_path"]))
                env_mask_paths.append(os.path.join(data_root_2d, frame["env_mask_path"]))
                object_mask_paths.append([os.path.join(data_root_2d,f) for f in frame["object_mask_path"]])
                object_bg_paths.append([os.path.join(data_root_2d,f) for f in frame["object_bg_path"]])
                if "camera_angle_x" in data_2d:
                    intrinsics.append(self.convert_intrinsics(data_2d["camera_angle_x"], self.org_image_size))
                else:
                    intrinsics.append(self.convert_intrinsics_from_focal(frame["fl_x"], self.org_image_size))
                extrinsics.append(self.convert_extrinsics(frame["transform_matrix"]))

            tmp_frame = {}
            tmp_frame["image_path"] = image_paths
            tmp_frame["depth_path"] = depth_paths
            tmp_frame["env_mask_path"] = env_mask_paths
            tmp_frame["object_mask_path"] = object_mask_paths
            tmp_frame["object_bg_path"] = object_bg_paths
            tmp_frame["image_path"] = image_paths
            tmp_frame["intrinsics"] = intrinsics
            tmp_frame["extrinsics"] = extrinsics

            if self.image_type == "exr":
                for ev in self.tonemap_evs:
                    tmp_frame["ev"] = ev
                    light_3d_path = os.path.join(scene_3d_path, light_cd+"_ev{:d}".format(ev))
                    json_3d_path = os.path.join(light_3d_path, "transforms.json")
                    with open(json_3d_path, "r") as f:
                        data_3d = json.load(f)
                    tmp_frame["rgb_path"] = os.path.join(data_root_3d, data_3d["rgb_path"])
                    tmp_frame["rgb_bg_path"] = [os.path.join(data_root_3d, f) for f in data_3d["rgb_bg_path"]]
                    tmp_frame["rgb_mask_path"] = [os.path.join(data_root_3d, f) for f in data_3d["rgb_mask_path"]]
                    tmp_frame["gs_path"] = os.path.join(data_root_3d, data_3d["gs_path"])
                    scene_frames.append(tmp_frame)
            else:
                light_3d_path = os.path.join(scene_3d_path, light_cd)
                json_3d_path = os.path.join(light_3d_path, "transforms.json")
                with open(json_3d_path, "r") as f:
                    data_3d = json.load(f)
                tmp_frame["rgb_path"] = os.path.join(data_root_3d, data_3d["rgb_path"])
                tmp_frame["rgb_bg_path"] = [os.path.join(data_root_3d, f) for f in data_3d["rgb_bg_path"]]
                tmp_frame["rgb_mask_path"] = [os.path.join(data_root_3d, f) for f in data_3d["rgb_mask_path"]]
                tmp_frame["gs_path"] = os.path.join(data_root_3d, data_3d["gs_path"])
                tmp_frame["proj_path"] = os.path.join(scene_3d_path, "projs.npy")
                scene_frames.append(tmp_frame)

        return scene_frames, len(scene_frames), False

    def convert_intrinsics(self, fov, image_size):
        intrinsics = np.eye(3, dtype=np.float32)
        intrinsics[0, 0] = 1.0 / (2 * math.tan(fov / 2))
        intrinsics[1, 1] = 1.0 * image_size[1] / (2 * image_size[0] * math.tan(fov / 2))   
        intrinsics[0, 2] = 1.0 / 2
        intrinsics[1, 2] = 1.0 / 2 
        return intrinsics.tolist()
    
    def convert_intrinsics_from_focal(self, fl_x, image_size):
        intrinsics = np.eye(3, dtype=np.float32)
        fl_y = fl_x * image_size[0] / image_size[1]
        intrinsics[0, 0] = fl_x / image_size[1]
        intrinsics[1, 1] = fl_y / image_size[0]
        intrinsics[0, 2] = 1.0 / 2
        intrinsics[1, 2] = 1.0 / 2 
        return intrinsics.tolist()
        
    def convert_extrinsics(self, pose):
        blender2opencv = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )
        opencv_c2w = np.array(pose) @ blender2opencv
        return opencv_c2w.tolist()
    
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

    def load_frames(self, image_paths):
        def load_frame(index, image_path):
            image = load_image(image_path)
            return index, image

        image_list = [None] * len(image_paths)
        for idx, image_path in enumerate(image_paths):
            index, image = load_frame(idx, image_path)
            image_list[index] = image
        images = torch.stack(image_list)
        
        return images

    def load_target_masks(self, target_mask_paths):
        def load_frame(index, mask_path):
            mask = 1 - load_mask(mask_path)
            return index, mask

        mask_list = [None] * len(target_mask_paths)
        for idx, mask_path in enumerate(target_mask_paths):
            index, mask = load_frame(idx, mask_path)
            mask_list[index] = mask
        masks = torch.stack(mask_list)
        
        return masks

    def load_input_rgb(self, input_object_bg_path, input_object_mask_path, insert_image_path):
        bg_img = load_image(input_object_bg_path)
        mask = load_mask(input_object_mask_path)
        insert_image = load_image(insert_image_path)
        img = insert_image * mask + bg_img * (1 - mask)
        return img.unsqueeze(0)

    def load_target_rgb(self, target_image_path):
        img = load_image(target_image_path)
        return img.unsqueeze(0)

    def load_rgb(self, image_path):
        img = load_image(image_path)
        return img.unsqueeze(0)

    def load_gs(self, gs_path):
        gs = load_gaus(gs_path)
        return gs

    def load_projs(self, proj_path):
        proj = np.load(proj_path, allow_pickle=True)
        return torch.tensor(proj)

    def getitem(self, index):
        scene, base_index, flag = self.scene_ids[index]
        if flag:
            base_example = self.scenes[scene]
            extrinsics = torch.tensor(base_example["extrinsics"], dtype=torch.float32)
            intrinsics = torch.tensor(base_example["intrinsics"], dtype=torch.float32)

            context_indices = target_indices = torch.arange(len(base_example['gt_path']))

            composite_paths = [base_example["composite_path"][i] for i in context_indices]
            bg_paths = [base_example["bg_path"][i] for i in context_indices]
            depth_paths = [base_example["depth_path"][i] for i in context_indices]
            gt_paths = [base_example["gt_path"][i] for i in target_indices]
            context_images = self.load_frames(composite_paths)
            context_bg_images = self.load_frames(bg_paths)
            context_depths = self.load_input_depths(depth_paths)
            target_images = self.load_frames(gt_paths)

            # load rgb
            rgb_composite_path = base_example["rgb_composite_path"]
            rgb_gt_path = base_example["rgb_gt_path"]
            context_rgb = self.load_rgb(rgb_composite_path)
            target_rgb = self.load_rgb(rgb_gt_path)
            projs = self.load_projs(base_example["proj_path"])
            if self.render_2d:
                render_indices = context_indices[:self.num_render_views]
                render_image_paths = [base_example["gt_path"][i] for i in render_indices]
                render_mask_paths = [base_example["mask_path"][i] for i in render_indices]
                render_images = self.load_frames(render_image_paths)
                render_masks = self.load_target_masks(render_mask_paths)
                render_gs_path = base_example["gs_path"]
                render_gs = self.load_gs(render_gs_path)
                render_intrinsics = intrinsics[render_indices]
                render_extrinsics = extrinsics[render_indices]
        else:
            base_example = self.scenes[scene][base_index]
            extrinsics = torch.tensor(base_example["extrinsics"], dtype=torch.float32)
            intrinsics = torch.tensor(base_example["intrinsics"], dtype=torch.float32)
            insert_index = np.random.randint(0, len(self.scenes[scene]))
            insert_example = self.scenes[scene][insert_index]
            base_ev = base_example["ev"] if "ev" in base_example else None
            insert_ev = insert_example["ev"] if "ev" in insert_example else None

            num_object = len(base_example["rgb_bg_path"])
            insert_object_id = np.random.randint(0, num_object)
            context_indices = target_indices = torch.arange(len(base_example['image_path']))
            
            # load images
            input_object_bg_paths = [base_example["object_bg_path"][i][insert_object_id] for i in context_indices]
            input_object_mask_paths = [base_example["object_mask_path"][i][insert_object_id] for i in context_indices]
            input_depth_paths = [base_example["depth_path"][i] for i in context_indices]
            insert_image_paths = [insert_example["image_path"][i] for i in context_indices]
            target_image_paths = [base_example["image_path"][i] for i in target_indices]
            context_images, context_bg_images = self.load_input_frames(input_object_bg_paths, input_object_mask_paths, insert_image_paths, base_ev, insert_ev)
            context_depths = self.load_input_depths(input_depth_paths)
            target_images = self.load_target_frames(target_image_paths, base_ev)

            # load rgb
            input_rgb_bg_path = base_example["rgb_bg_path"][insert_object_id]
            input_rgb_mask_path = base_example["rgb_mask_path"][insert_object_id]
            insert_rgb_path = insert_example["rgb_path"]
            context_rgb = self.load_input_rgb(input_rgb_bg_path, input_rgb_mask_path, insert_rgb_path)
            target_rgb_path = base_example["rgb_path"]
            target_rgb = self.load_target_rgb(target_rgb_path)
            projs = self.load_projs(base_example["proj_path"])

            if self.render_2d:
                render_indices = context_indices[:self.num_render_views]
                render_image_paths = [base_example["image_path"][i] for i in render_indices]
                render_mask_paths = [base_example["env_mask_path"][i] for i in render_indices]
                render_images = self.load_target_frames(render_image_paths, base_ev)
                render_masks = self.load_target_masks(render_mask_paths)
                render_gs_path = base_example["gs_path"]
                render_gs = self.load_gs(render_gs_path)
                render_intrinsics = intrinsics[render_indices]
                render_extrinsics = extrinsics[render_indices]

        if self.image_size != self.org_image_size:
            context_images = rescale_and_crop(context_images, self.org_image_size, self.image_size)
            context_bg_images = rescale_and_crop(context_bg_images, self.org_image_size, self.image_size)
            context_depths = rescale_and_crop(context_depths, self.org_image_size, self.image_size)
            target_images = rescale_and_crop(target_images, self.org_image_size, self.image_size)
            if self.render_2d:
                render_images = rescale_and_crop(render_images, self.org_image_size, self.image_size)
                render_intrinsics = adjust_intrinsics(render_intrinsics, self.org_image_size, self.image_size)

        if self.add_background:
            context_images = torch.cat([context_images, context_bg_images], dim=1)
        if self.add_depth:
            context_images = torch.cat([context_images, context_depths], dim=1)

        example = {
            "input_images": context_images, 
            "input_index": context_indices,
            "input_rgb" : context_rgb,
            "target_images": target_images,
            "target_index": target_indices,
            "target_rgb": target_rgb,
            "projs": projs,
            "scene": scene,
        }

        if self.render_2d:
            example.update({
                "render_intrinsics": render_intrinsics,
                "render_extrinsics": render_extrinsics,
                "render_images": render_images,
                "render_index": render_indices,
                "render_masks": render_masks,
                "render_gs": render_gs,
                "render_gs_path": render_gs_path,
            })

        return example

    def __len__(self):
        return len(self.scene_ids)

    def __getitem__(self, index):
        return self.getitem(index)