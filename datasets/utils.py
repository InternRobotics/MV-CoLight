import os
from einops import einsum, rearrange
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import numpy as np
from packaging import version as pver
from jaxtyping import Float
from torch import Tensor
from types import SimpleNamespace
from PIL import Image
import cv2
from plyfile import PlyData, PlyElement
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
from submodules.Depth-Anything-V2.depth_anything_v2.dpt import DepthAnythingV2

def numpy_normalize(v):
    return v / np.linalg.norm(v)

def average_camera_poses(poses):
    """
    Assuming the directions of x,y,z axis are right, down, forward
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.
    
    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 4, 4)
    Outputs:
        pose_avg: (4, 4) the average pose
    """
    # 1. Compute the center
    center = poses[:, :3, 3].mean(0)  # (3)
    
    # 2. Compute the z axis
    z = numpy_normalize(poses[:, :3, 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[:, :3, 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = numpy_normalize(np.cross(y_, z))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x)  # (3)
    
    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)
    pose_avg = np.concatenate((pose_avg, np.asarray([[0, 0, 0, 1]])), axis=0)       # (4, 4)

    return pose_avg

def average_camera_poses_torch(input_c2ws):
    # noramlize input camera poses
    position_avg = input_c2ws[:, :3, 3].mean(0) # (3,)
    forward_avg = input_c2ws[:, :3, 2].mean(0) # (3,)
    down_avg = input_c2ws[:, :3, 1].mean(0) # (3,)
    # gram-schmidt process
    forward_avg = nn.functional.normalize(forward_avg, dim=0)
    down_avg = nn.functional.normalize(down_avg - down_avg.dot(forward_avg) * forward_avg, dim=0)
    right_avg = torch.cross(down_avg, forward_avg)
    pos_avg = torch.stack([right_avg, down_avg, forward_avg, position_avg], dim=1) # (3, 4)
    pos_avg = torch.cat([pos_avg, torch.tensor([[0, 0, 0, 1]], device=pos_avg.device).float()], dim=0) # (4, 4)
    # pos_avg_inv = torch.inverse(pos_avg)
    # return pos_avg_inv
    return pos_avg

class RandomHorizontalFlipWithPose(nn.Module):
    def __init__(self, p=0.5):
        super(RandomHorizontalFlipWithPose, self).__init__()
        self.p = p

    def get_flip_flag(self, n_image):
        return torch.rand(n_image) < self.p

    def forward(self, image, flip_flag=None):
        n_image = image.shape[0]
        if flip_flag is not None:
            assert n_image == flip_flag.shape[0]
        else:
            flip_flag = self.get_flip_flag(n_image)
        
        ret_images = []
        for fflag, img in zip(flip_flag, image):
            if fflag:
                ret_images.append(F.hflip(img))
            else:
                ret_images.append(img)
        return torch.stack(ret_images, dim=0)

class Camera(object):
    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

def get_fov(intrinsics: Float[Tensor, "batch 3 3"]) -> Float[Tensor, "batch 2"]:
    intrinsics_inv = intrinsics.inverse()

    def process_vector(vector):
        vector = torch.tensor(vector, dtype=torch.float32, device=intrinsics.device)
        vector = einsum(intrinsics_inv, vector, "b i j, j -> b i")
        return vector / vector.norm(dim=-1, keepdim=True)

    left = process_vector([0, 0.5, 1])
    right = process_vector([1, 0.5, 1])
    top = process_vector([0.5, 0, 1])
    bottom = process_vector([0.5, 1, 1])
    fov_x = (left * right).sum(dim=-1).acos()
    fov_y = (top * bottom).sum(dim=-1).acos()
    return torch.stack((fov_x, fov_y), dim=-1)

def rescale(
    image: Float[Tensor, "c h_in w_in"],
    shape: tuple[int, int],
) -> Float[Tensor, "c h_out w_out"]:
    h, w = shape
    image_new = (image * 255).clip(min=0, max=255).type(torch.uint8)
    image_new = rearrange(image_new, "c h w -> h w c").detach().cpu().numpy()
    if image_new.shape[2] == 1:
        image_new = Image.fromarray(image_new[:, :, 0])
        image_new = image_new.resize((w, h), Image.LANCZOS)
        image_new = np.array(image_new) / 255
        image_new = torch.tensor(image_new, dtype=image.dtype, device=image.device)
        image_new = image_new.unsqueeze(-1)
    elif image_new.shape[2] == 3:
        image_new = Image.fromarray(image_new)
        image_new = image_new.resize((w, h), Image.LANCZOS)
        image_new = np.array(image_new) / 255
        image_new = torch.tensor(image_new, dtype=image.dtype, device=image.device)
    elif image_new.shape[2] == 4:
        image_new = Image.fromarray(image_new[:, :, :3])
        image_new = image_new.resize((w, h), Image.LANCZOS)
        image_new = np.array(image_new) / 255
        image_new = torch.tensor(image_new, dtype=image.dtype, device=image.device)
    else:
        raise ValueError(f"Unsupported image shape: {image_new.shape}")
    return rearrange(image_new, "h w c -> c h w")

def center_crop(
    images: Float[Tensor, "*#batch c h w"],
    shape: tuple[int, int],
) -> Float[Tensor, "*#batch c h_out w_out"]:
    *_, h_in, w_in = images.shape
    h_out, w_out = shape

    # Note that odd input dimensions induce half-pixel misalignments.
    row = (h_in - h_out) // 2
    col = (w_in - w_out) // 2

    # Center-crop the image.
    images = images[..., :, row : row + h_out, col : col + w_out]
    
    return images

def rescale_and_crop(
    images: Float[Tensor, "*#batch c h w"],
    origin_shape: tuple[int, int],
    shape: tuple[int, int],
) -> Float[Tensor, "*#batch c h_out w_out"]:

    h_in, w_in = origin_shape
    h_out, w_out = shape
    assert h_out <= h_in and w_out <= w_in
    
    scale_factor = max(h_out / h_in, w_out / w_in)
    h_scaled = round(h_in * scale_factor)
    w_scaled = round(w_in * scale_factor)
    assert h_scaled == h_out or w_scaled == w_out
    
    *batch, c, h, w = images.shape
    images = images.reshape(-1, c, h, w)
    images = torch.stack([rescale(image, (h_scaled, w_scaled)) for image in images])
    images = images.reshape(*batch, c, h_scaled, w_scaled)
    
    return center_crop(images, shape)

def adjust_intrinsics(
    intrinsics: Float[Tensor, "*#batch 3 3"],
    origin_shape: tuple[int, int],
    shape: tuple[int, int],
) -> Float[Tensor, "*#batch 3 3"]:

    h_in, w_in = origin_shape
    h_out, w_out = shape
    assert h_out <= h_in and w_out <= w_in
    
    scale_factor = max(h_out / h_in, w_out / w_in)
    h_in = round(h_in * scale_factor)
    w_in = round(w_in * scale_factor)
    assert h_in == h_out or w_in == w_out

    intrinsics = intrinsics.clone()
    intrinsics[..., 0, 0] *= w_in / w_out  # fx
    intrinsics[..., 1, 1] *= h_in / h_out  # fy

    return intrinsics

def reinhard_tonemap(H, maxH):
    H = H*(1+H/(maxH*maxH))/(1+H)
    return H

def load_image(image_path, ev=None):
    if image_path.endswith('.npy'):
        image = torch.from_numpy(np.load(image_path, allow_pickle=True)) / 255.0
    elif image_path.endswith('.exr') or image_path.endswith('.hdr'):
        try:
            hdr_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)[:, :, :3]
        except:
            print(f"Failed to load image: {image_path}")
        hdr_image = cv2.cvtColor(hdr_image, cv2.COLOR_BGR2RGB)
        max_H = np.max(hdr_image)
        image = np.zeros_like(hdr_image)
        for i in range(3):  
            image[:, :, i] = reinhard_tonemap(hdr_image[:, :, i].astype(np.float64) * np.power(2.0, ev), max_H * np.power(2.0, ev))
        image = torch.from_numpy(image).permute(2, 0, 1)
    else:
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1) / 255.0
    return image

def load_mask(mask_path):
    if mask_path.endswith('.npy'):
        mask = torch.from_numpy(np.load(mask_path, allow_pickle=True)) / 255.0
    else:
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        mask = torch.from_numpy(mask).unsqueeze(0) / 255.0
    return mask

def load_depth(depth_path):
    if depth_path.endswith('.npy'):
        depth = torch.from_numpy(np.load(depth_path, allow_pickle=True)) / 255.0
    elif depth_path.endswith('.png'):
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if len(depth.shape) == 3:
            depth = depth[:, :, 0]
        depth = torch.from_numpy(depth).unsqueeze(0) / 255.0
    else:
        raise ValueError(f"Unsupported depth format: {depth_path}")
    return depth

def load_gaus(gs_path):
    if gs_path.endswith('.npy'):
        gs = np.load(gs_path, allow_pickle=True)
    else:
        gs = convert_ply(gs_path)
    gs = torch.tensor(gs, dtype=torch.float32).permute(2, 0, 1)
    return gs

def get_image_size(data_path):
    image_dir = os.path.join(data_path, "composites")
    if not os.path.exists(image_dir):
        image_dir = os.path.join(data_path, os.listdir(data_path)[0], "images")
    image_path = os.path.join(image_dir, os.listdir(image_dir)[0])
    if image_path.endswith(".npy"):
        image = np.load(image_path, allow_pickle=True)
        return image.shape[1:], image_path.split(".")[-1]
    else:
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        return image.shape[:2], image_path.split(".")[-1]

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

def convert_ply(path):
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

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
    
    opacities = np.ones_like(opacities)

    scales = torch.from_numpy(scales).float()
    scales = torch.exp(scales)
    scales = scales.numpy()

    rots = torch.from_numpy(rots).float()
    rots = torch.nn.functional.normalize(rots)
    rots = rots.numpy()

    hilbert = []
    if len(xyz)==256*384:
        order = 8
        hilbert_curve_2d_23(order, 0, 0, 2 ** order, 0, 0, 2 ** order, hilbert)
        feat = np.zeros((256, 384, 3+1+3+4), dtype=np.float32)
    else:
        order = 8
        hilbert_curve_2d(order, 0, 0, 2 ** order, 0, 0, 2 ** order, hilbert)
        feat = np.zeros((256, 256, 3+1+3+4), dtype=np.float32)
    for i, point, opacity, scale, rot in zip(range(len(xyz)), xyz, opacities, scales, rots):
        if i < len(hilbert):
            x, y = hilbert[i]
            feat[int(x), int(y)] = np.concatenate([point, opacity, scale, rot])
    
    return feat

def load_estimate_depth(image):
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = 'vitl'
    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'./checkpoints/depth_anything_v2_vitl.pth', map_location='cpu'))
    model = model.cuda().eval()
    depth = 100 / model.infer_image(image.permute(1,2,0).cpu().numpy()*255) # HxW raw depth map in numpy
    depth = np.nan_to_num(depth, nan=1)
    depth = np.clip(depth, 0, 1)
    return torch.from_numpy(depth).unsqueeze(0)
