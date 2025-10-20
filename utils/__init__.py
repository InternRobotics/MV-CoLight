import numpy as np
import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from einops import einsum, rearrange
from plyfile import PlyData, PlyElement

C0 = 0.28209479177387814

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def gs_render(gs, intr, extr, H, W):
    B, C, Gh, Gw = gs.shape
    B, N = intr.shape[:2]
    pred_images = []
    for i in range(B):
        gs_i = gs[i]
        gs_i = gs_i.permute(1, 2, 0).view(Gh * Gw, -1).cuda()
        shs = RGB2SH(gs_i[:, 0:3]).unsqueeze(1)
        means3D = gs_i[:, 3:6]
        means2D = torch.zeros_like(means3D, dtype=means3D.dtype, device="cuda") + 0
        opacity = torch.sigmoid(gs_i[:, 6:7])
        scales = gs_i[:, 7:10]
        rotations = gs_i[:, 10:14]
        for j in range(N):
            cur_intr = intr[i][j]
            cur_extr = extr[i][j]
            with torch.no_grad():
                viewmat = torch.linalg.inv(cur_extr).cuda()

                tanfovx = 1 / (2 * cur_intr[0, 0])
                tanfovy = 1 / (2 * cur_intr[1, 1])
                bg_color = torch.ones(3, dtype=torch.float32, device="cuda")

                world_view_transform = viewmat.transpose(0, 1).cuda()
                projection_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=2*math.atan(tanfovx), fovY=2*math.atan(tanfovy)).transpose(0,1).cuda()
                full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0).float().cuda()
                camera_center = world_view_transform.inverse()[3, :3].cuda()
                raster_settings = GaussianRasterizationSettings(
                    image_height=H,
                    image_width=W,
                    tanfovx=tanfovx,
                    tanfovy=tanfovy,
                    bg=bg_color,
                    scale_modifier=1.0,
                    viewmatrix=world_view_transform,
                    projmatrix=full_proj_transform,
                    sh_degree=0,
                    campos=camera_center,
                    prefiltered=False,
                    debug=False,
                    antialiasing=True,
                )

            rasterizer = GaussianRasterizer(raster_settings=raster_settings)


            # Rasterize visible Gaussians to image, obtain their radii (on screen). 
            rendered_image, radii, depth_image = rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = shs,
                opacities = opacity,
                scales = scales,
                rotations = rotations)
            
            pred_images.append(rendered_image)
    pred_images = torch.stack(pred_images, dim=0)

    return pred_images

C0 = 0.28209479177387814

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

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

def map_colors_to_feat(save_path, hilbert, colors, image_height, image_width):
    rgb = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    color = colors.cpu().numpy()
    for i, c in enumerate(color):
        if i < len(hilbert):
            x, y = hilbert[i]
            rgb[int(x), int(y)] = c
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, rgb)
    
def save_ply(ply_path, rgb, new_ply_path):
    def construct_list_of_attributes():
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(shs.shape[1]):
            l.append('f_dc_{}'.format(i))
        l.append('opacity')
        for i in range(scales.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rots.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    assert ply_path.endswith(".npy")
    feat = np.load(ply_path, allow_pickle=True)
    
    feat = rearrange(feat, "h w c-> (h w) c")
    shs = RGB2SH(rearrange(rgb, "f c h w -> (f h w) c").detach().cpu().numpy())
    xyz = feat[:, 0:3]
    opacities = feat[:, 3:4]
    scales = np.log(feat[:, 4:7])
    rots = feat[:, 7:11]
    normals = np.zeros_like(xyz)   
    
    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, shs, opacities, scales, rots), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(new_ply_path)