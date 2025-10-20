import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import subprocess
import numpy as np
import cv2
import os
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree
from PIL import Image
import json
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
import torch.nn.functional as F
import math
import shutil
from colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat

INPUT_DIR = "/path/to/real_scene_2d" 
OUPUT_DIR = "/path/to/real_scene_3d" 

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

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

def convert_bin_to_json(scene_path):

    cameras_extrinsic_file = os.path.join(scene_path, "sparse/0", "images.bin")
    cameras_intrinsic_file = os.path.join(scene_path, "sparse/0", "cameras.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)

    json_path = os.path.join(scene_path, "transforms.json")
    
    json_data = {
        "frames": [],
    }

    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):

        frame = {}

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        w2c = np.eye(4,4)
        w2c[:3, :3] = np.transpose(R)
        w2c[:3, 3] = T
        c2w = np.linalg.inv(w2c)
        c2w[:3, 1:3] *= -1
        frame["image_path"] = os.path.join(scene_path, "images", extr.name)
        frame["depth_path"] = os.path.join(scene_path, "depths", extr.name)
        frame["object_mask_path"] = [os.path.join(scene_path, "object_masks", extr.name.split('.')[0]+"_object0.png")]
        frame["env_mask_path"] = os.path.join(scene_path, "env_masks", extr.name)
        frame["object_bg_path"] = [os.path.join(scene_path, "object_bgs", extr.name.split('.')[0]+"_object0.png")]
        frame["transform_matrix"] = c2w.tolist()
        frame["fl_x"] = focal_length_x
        frame["fl_y"] = focal_length_y
        json_data["frames"].append(frame)

    json_data["frames"] = sorted(
        json_data["frames"],
        key=lambda x: int(x["image_path"].split('/')[-1].split(".")[0])
    )
    with open(json_path, "w") as f:
        json.dump(json_data, f)

def rgb2color(rgbs, object_masks, points_conf, conf_threshold):
    colors = []
    for i in range(rgbs.shape[0]):
        depth_mask = (points_conf >= conf_threshold)[i]
        depth_mask[object_masks[i]==0] = False
        color = rgbs[i].reshape(-1,3)/255.
        colors.append(color[depth_mask.reshape(-1)])
    for i in range(rgbs.shape[0]):
        depth_mask = (points_conf >= conf_threshold)[i]
        depth_mask[object_masks[i]>0] = False
        color = rgbs[i].reshape(-1,3)/255.
        colors.append(color[depth_mask.reshape(-1)])
    colors = torch.cat(colors, dim=0) * 255
    return colors

def process_images(model, device, dtype, scene_path, save_path, tree, hilbert, image_height, image_width, conf_percentile=1, max_points=9_000_000):
    """
    Process input images through the VGGT model to extract parameters.

    Returns:
        intrinsic: Camera intrinsic parameters
        extrinsic: Camera extrinsic parameters (world2camera)
        points: 3D point cloud coordinates (S * W * H * 3 --> Flatten)
        point_colors: RGB colors for each point
        images: Processed input images
        image_names: List of image file names
    """
    image_names = [os.path.join(scene_path, "light4", "images", img) for img in sorted(os.listdir(os.path.join(scene_path, "light4", "images")))]
    original_image_shape = np.array(Image.open(image_names[0])).shape
    images = load_and_preprocess_images(image_names).to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        depth_map, _ = model.depth_head(aggregated_tokens_list, images, ps_idx)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        _, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)
        depth_map = F.interpolate(depth_map[0].permute(0, 3, 1, 2), size=(256, 256), mode='nearest').permute(0,2,3,1).unsqueeze(0)
        depths = depth_map[0] / depth_map.max()
        for i in range(len(depths)):
            depth = depths[i].squeeze(0).detach().cpu().numpy()
            depth = (depth * 255).astype(np.uint8)
            for lcd in os.listdir(scene_path):
                os.makedirs(os.path.join(scene_path, lcd, "depths"), exist_ok=True)
                os.makedirs(os.path.join(scene_path, lcd, "env_masks"), exist_ok=True)
                cv2.imwrite(os.path.join(scene_path, lcd, "depths", f"{i}.png"), depth)
                cv2.imwrite(os.path.join(scene_path, lcd, "env_masks", f"{i}.png"), np.zeros_like(depth))

        point_conf = F.interpolate(point_conf[0].unsqueeze(1), size=(256, 256), mode='nearest').permute(1,0,2,3)
        intrinsic = intrinsic * 256 / 518
        point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map.squeeze(0), 
                                                                     extrinsic.squeeze(0), 
                                                                     intrinsic.squeeze(0))

    points_conf = point_conf.squeeze(0).detach().cpu().numpy()
    point_colors = images.squeeze(0).permute(0, 2, 3, 1).squeeze(0).detach().cpu()
    point_colors = F.interpolate(point_colors.permute(0, 3, 1, 2), size=(256, 256), mode='nearest').permute(0,2,3,1)

    conf_threshold = np.percentile(points_conf, conf_percentile)

    all_points = []
    projs = []
    object_masks = []
    for i in range(point_map_by_unprojection.shape[0]):
        object_mask = cv2.imread(os.path.join(scene_path, "light4", "object_masks", f"{i}_object0.png"), -1)[:, :, 0]
        object_masks.append(object_mask)

    for i in range(point_map_by_unprojection.shape[0]):
        depth_mask = (points_conf >= conf_threshold)[i]
        depth_mask[object_masks[i]==0] = False
        depth_mask = depth_mask.reshape(-1)
        for j in range(len(depth_mask)):
            if depth_mask[j]:
                projs.append([i,j])
        xyz = torch.tensor(point_map_by_unprojection[i]).reshape(-1, 3)
        all_points.append(xyz[depth_mask])
    
    len_object = len(torch.cat(all_points, dim=0))

    for i in range(point_map_by_unprojection.shape[0]):
        depth_mask = (points_conf >= conf_threshold)[i]
        depth_mask[object_masks[i]>0] = False
        depth_mask = depth_mask.reshape(-1)
        for j in range(len(depth_mask)):
            if depth_mask[j]:
                projs.append([i,j])
        xyz = torch.tensor(point_map_by_unprojection[i]).reshape(-1, 3)
        all_points.append(xyz[depth_mask])

    all_points = torch.cat(all_points, dim=0)
    len_all = len(all_points)
    object_mask = torch.zeros(len_all, dtype=torch.bool)
    object_mask[:len_object] = True
    
    mask = [False] * len_all
    indices = np.linspace(0, len_all - 1, image_height * image_width, dtype=int)
    for index in indices:
        mask[index] = True
    mask = torch.tensor(mask)

    all_points = torch.tensor(all_points)[mask]
    object_mask = torch.tensor(object_mask)[mask]

    sorted_indices = plot_hilbert_with_points_3d(tree, all_points)
    sorted_indices = torch.tensor(sorted_indices)

    all_points = all_points[sorted_indices]
    object_mask = object_mask[sorted_indices]
    projs = torch.tensor(projs)[mask][sorted_indices]

    img2pcd = []
    for i, proj in enumerate(projs):
        uid = int(proj[0])
        index = int(proj[1])
        x = int(hilbert[i][0])
        y = int(hilbert[i][1])
        img2pcd.append([uid, index, x, y])
    img2pcd = np.stack(img2pcd).astype(np.int32)
    np.save(os.path.join(save_path, "projs.npy"), img2pcd)

    colors = {}
    bg_colors = {}

    for lcd in os.listdir(scene_path):
        if not os.path.isdir(os.path.join(scene_path, lcd)):
            continue
        if lcd != "base":
            input_path = os.path.join(scene_path, lcd)
            with open(os.path.join(input_path, f'transforms.json'), "r") as f:
                data = json.load(f)

            rgbs, bg_rgbs = [], []
            frames = data['frames']
            cnt = 0
            for frame in frames:
                rgbs.append(cv2.imread(os.path.join(scene_path, lcd, "images", f"{cnt}.png"), -1))
                bg_rgbs.append(cv2.imread(os.path.join(scene_path, lcd, "object_bgs", f"{cnt}_object0.png"), -1))
                cnt+=1
            rgbs = torch.from_numpy(np.stack(rgbs)).float()
            bg_rgbs = torch.from_numpy(np.stack(bg_rgbs)).float()
            # project to world
            colors[lcd] = rgb2color(rgbs, object_masks, points_conf, conf_threshold)[mask][sorted_indices]
            bg_colors[lcd] = rgb2color(bg_rgbs, object_masks, points_conf, conf_threshold)[mask][sorted_indices]

    map_colors_to_feat(scene_path, save_path, hilbert, colors, image_height, image_width)
    map_bgs_to_feat(scene_path, save_path, hilbert, bg_colors, object_mask, image_height, image_width)
    map_masks_to_feat(scene_path, save_path, hilbert, object_mask, image_height, image_width)

    torch.save(colors, f"{save_path}/colors.pt")
    torch.save(all_points, f"{save_path}/points.pt")

    all_points = all_points.cpu().numpy()

    return intrinsic, extrinsic, all_points, colors, images, image_names, original_image_shape
    
def map_colors_to_feat(scene_path, save_path, hilbert, colors, image_height, image_width):
    for lcd in sorted(os.listdir(scene_path)):
        if not os.path.isdir(os.path.join(scene_path, lcd)):
            continue
        rgb = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        color = colors[lcd].cpu().numpy()
        for i, c in enumerate(color):
            if i < len(hilbert):
                x, y = hilbert[i]
                rgb[int(x), int(y)] = c
        os.makedirs(os.path.join(save_path, lcd), exist_ok=True)
        cv2.imwrite(os.path.join(save_path, lcd, "img.png"), rgb)

def map_masks_to_feat(scene_path, save_path, hilbert, object_mask, image_height, image_width):
    for lcd in sorted(os.listdir(scene_path)):
        if not os.path.isdir(os.path.join(scene_path, lcd)):
            continue
        mask = np.zeros((image_height, image_width), dtype=np.uint8)
        for i in range(len(object_mask)):
            if i < len(hilbert):
                x, y = hilbert[i]
                if object_mask[i]:
                    mask[int(x), int(y)] = 255
        os.makedirs(os.path.join(save_path, lcd), exist_ok=True)
        Image.fromarray(mask).save(f"{save_path}/{lcd}/mask.png")

def map_bgs_to_feat(scene_path, save_path, hilbert, bgs, object_mask, image_height, image_width):
    for lcd in sorted(os.listdir(scene_path)):
        if not os.path.isdir(os.path.join(scene_path, lcd)):
            continue
        background = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        bg = bgs[lcd].cpu().numpy()
        for i, b in enumerate(bg):
            if i < len(hilbert):
                x, y = hilbert[i]
                background[int(x), int(y)] = b
                if object_mask[i]:
                    background[int(x), int(y)] = [127, 127, 127]
        cv2.imwrite(os.path.join(save_path, lcd, "bg.png"), background)
def save_cameras(file_path, intrinsic, width, height, original_image_shape):
    """
    Save camera parameters to a COLMAP-compatible format.  
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # scale_factor_x = original_image_shape[1] / width
    # scale_factor_y = original_image_shape[0] / height

    with open(file_path, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        
        for i, intr in enumerate(intrinsic):
            fx = intr[0, 0].item()  # fx
            fy = intr[1, 1].item()  # fy

            f.write("{} PINHOLE {} {} {} {} {} {}\n".format(
                i,
                original_image_shape[1],  # Image width
                original_image_shape[0],  # Image height
                fx,  # fx
                fy,  # fy
                original_image_shape[1]/2,  # cx
                original_image_shape[0]/2,  # cy
                # 0.0,  # k1 (distortion) if MODEL==OPENCV required
                # 0.0,  # k2
                # 0.0,  # p1
                # 0.0   # p2
            ))

def save_images(file_path, extrinsic, image_names):
    """
    Save image parameters including camera poses to a COLMAP-compatible format.
    """
    with open(file_path, 'w') as f:
        f.write("# Image list\n")
        f.write("# ImageId, Qw, Qx, Qy, Qz, Tx, Ty, Tz, CameraId, Name\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for image_id, (extr, image_name) in enumerate(zip(extrinsic, image_names)):

            # extr_homo = np.vstack((extr, np.array([0, 0, 0, 1])))  # 添加最后一行 [0, 0, 0, 1]
            # extr_w2c = np.linalg.inv(extr_homo)
            rotation_matrix = extr[:3, :3]
            qvec = R.from_matrix(rotation_matrix).as_quat()

            # Ensure the quaternion is in the order (qw, qx, qy, qz)
            qvec = np.roll(qvec, 1)

            qw, qx, qy, qz = qvec
            tx, ty, tz = extr[0, 3], extr[1, 3], extr[2, 3]  # Translation

            f.write("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(
                image_id,
                qw, qx, qy, qz,
                tx, ty, tz, 
                image_id,  # The same as camera_id variable
                image_name 
            ))
            # 假设 POINTS2D 是空的
            f.write("\n")

def save_points3D_with_colors(file_path, points, point_colors):
    """
    Save 3D points and their colors in both TXT and PLY formats.
    
    Args:
        file_path: Output file path for 3D points
        points: 3D point coordinates
        point_colors: RGB colors for each point
    """

    point_colors = np.clip(point_colors * 255, 0, 255).astype(np.uint8)

    # Save as TXT
    with open(file_path, 'w') as f:
        f.write("# 3D point list\n")
        f.write("# PointId, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        
        for point_id, (point, color) in enumerate(zip(points, point_colors)):
            f.write("{} {} {} {} {} {} {} {} {}\n".format(
                point_id, point[0].item(), point[1].item(), point[2].item(),  # X, Y, Z
                color[0].item(), color[1].item(), color[2].item(),             # R, G, B
                0.0,                                                          # ERROR
                "1 1"                                                         # TRACK[]
            ))

    # Save as PLY
    ply_file_path = file_path.replace('.txt', '.ply')
    with open(ply_file_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for point, color in zip(points, point_colors):
            f.write("{} {} {} {} {} {}\n".format(
                point[0].item(), point[1].item(), point[2].item(),  # X, Y, Z
                int(color[0].item()), int(color[1].item()), int(color[2].item())  # R, G, B
            ))

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

def convert_txt_to_bin(input_dir, output_dir):
    """
    Convert COLMAP text format files to binary format using COLMAP's model_converter.
    
    Args:
        input_dir: Input directory containing text format files
        output_dir: Output directory for binary format files
    """
    subprocess.run([
        'colmap', 'model_converter',
        '--input_path', input_dir,
        '--output_path', output_dir,
        '--output_type', 'BIN'
    ], check=True)
    print("Conversion from TXT to BIN complete!")


def main(scene_path, save_path, tree, hilbert, image_height, image_width):
    """
    Main function to process images using VGGT model and save results in COLMAP format.
    Handles the complete pipeline from image processing to final binary conversion.
    """
    ################## Change There #########################################
    conf_percentile = 30  # Percentage: 1 == 1%
    max_points = 9_000_000  
    ################## Change There #########################################

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    
    intrinsic, extrinsic, points, colors, images, image_names, original_image_shape = process_images(
        model, device, dtype, scene_path, save_path, tree, hilbert, image_height, image_width, conf_percentile, max_points
    )
    image_names = [os.path.basename(name) for name in image_names]

    for lcd in os.listdir(scene_path):
        point_colors = colors[lcd].cpu().numpy() / 255.
        output_dir = os.path.join(scene_path, lcd, "sparse/0")
        save_cameras(os.path.join(output_dir, "cameras.txt"), intrinsic.squeeze(0).cpu().numpy(), images.shape[-1], images.shape[-2], original_image_shape)
        save_images(os.path.join(output_dir, "images.txt"), extrinsic.squeeze(0).cpu().numpy(), image_names)
        save_points3D_with_colors(os.path.join(output_dir, "points3D.txt"), points, point_colors)
        convert_txt_to_bin(output_dir, output_dir)
        convert_bin_to_json(os.path.join(scene_path, lcd))

        shutil.rmtree(os.path.dirname(output_dir))

        print("Done")
        print("Output path:", output_dir)

if __name__ == "__main__":
    order = 9
    image_height, image_width = 512, 512
    hilbert = []
    hilbert_curve_2d(order, 0, 0, 2**order, 0, 0, 2**order, hilbert)

    # Generate Hilbert curve points
    hilbert3d = np.load("./checkpoints/hilbert3d_order9.npy", allow_pickle=True)
    tree = KDTree(hilbert3d)

    for scene in os.listdir(INPUT_DIR):
        scene_path = os.path.join(INPUT_DIR, scene)
        save_path = os.path.join(OUPUT_DIR, scene)
        os.makedirs(save_path, exist_ok=True)
        main(scene_path, save_path, tree, hilbert, image_height, image_width)