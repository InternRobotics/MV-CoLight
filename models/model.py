#-*- coding:utf-8 -*-
import math
import time
import torch
import torch.nn as nn
from einops import repeat
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from thop import profile
from registry import MODELS
from models.swint2d import SwinT2D
from models.swint3d import SwinT3D

@MODELS.register_module("ObjectCompositing3D")
class ObjectCompositing3D(nn.Module):
    def __init__(
        self, 
        # 2d object compositing
        num_in_ch_2d=7,
        num_out_ch_2d=3,
        img_size_2d=256,
        patch_size_2d=16,
        embed_dim_2d=96, 
        depths_2d=[6, 6, 6], 
        num_heads_2d=[6, 6, 6],
        window_size_2d=8, 
        # 3D object compositing
        num_in_ch=6,    
        num_out_ch=3,     
        img_size=256, 
        patch_size=16, 
        embed_dim=96, 
        depths=[6, 6, 6], 
        num_heads=[6, 6, 6],
        window_size=8, 
        mlp_ratio=4., 
        qkv_bias=True, 
        qk_scale=None,
        drop_rate=0., 
        attn_drop_rate=0., 
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm, 
        ape=False, 
        patch_norm=True,
        use_checkpoint=False, 
        resi_connection='1conv',
        **kwargs):
        super(ObjectCompositing3D, self).__init__()
        self.swin2d = SwinT2D(
            num_in_ch=num_in_ch_2d,
            num_out_ch=num_out_ch,
            img_size=img_size_2d, 
            patch_size=patch_size_2d, 
            embed_dim=embed_dim_2d, 
            depths=depths_2d, 
            num_heads=num_heads_2d,
            window_size=window_size_2d, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale,
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate, 
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer, 
            ape=ape, 
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint, 
            resi_connection=resi_connection,
            **kwargs)
        self.swin3d = SwinT3D(
            num_in_ch=num_in_ch,
            num_out_ch=num_out_ch,
            img_size=img_size, 
            patch_size=patch_size, 
            embed_dim=embed_dim, 
            depths=depths, 
            num_heads=num_heads,
            window_size=window_size, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale,
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate, 
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer, 
            ape=ape, 
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint, 
            resi_connection=resi_connection,
            **kwargs)
        
    def projection(self, x, projs, Gh, Gw):
        B, N, C, H, W = x.shape
        x = x.reshape(B, N, C, H * W)
        new_color = torch.zeros((B, 1, C, Gh, Gw), device=x.device)

        # 计算索引
        proj_indices = projs[:, :, 1]

        # 使用高级索引直接赋值
        batch_indices = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, Gh * Gw)
        height_indices = projs[:, :, 2]
        width_indices = projs[:, :, 3]
        channel_indices = projs[:, :, 0]

        # 使用 gather 和 scatter
        x_selected = x[batch_indices, channel_indices, :, proj_indices]
        new_color[batch_indices, 0, :, height_indices, width_indices] = x_selected
    
        return new_color

    def forward(self, x2d: torch.Tensor, x3d: torch.Tensor, projs: torch.Tensor):
        B, N_2d, _, H, W = x2d.shape
        _, N_3d, _, Gh, Gw  = x3d.shape
        
        o2d = self.swin2d(x2d)
        proj_x = self.projection(o2d, projs, Gh, Gw)

        x3d = torch.cat([x3d, proj_x], dim=2)
        o3d = self.swin3d(x3d)

        return o2d, o3d

    def get(self, proj_x: torch.Tensor, x3d: torch.Tensor):
        x3d = torch.cat([x3d, proj_x], dim=2)
        o3d = self.swin3d(x3d)

        return o3d