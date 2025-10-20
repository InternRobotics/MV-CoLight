import argparse
import os
import time

import imageio
import lpips
import numpy as np
import torch
import torchvision
from easydict import EasyDict as edict
from einops import rearrange

from einops import rearrange
from mmengine.config import Config
from omegaconf import OmegaConf
from registry import build_module, DATASETS, MODELS
from torch.utils.data import DataLoader
from torchmetrics.functional import (
    peak_signal_noise_ratio as psnr,
    structural_similarity_index_measure as fused_ssim,
)
from tqdm import tqdm
from utils import gs_render, save_ply


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate model")

    # Model configuration
    parser.add_argument("-m", "--model_path", type=str, required=True)
    parser.add_argument("-s", "--save_dir", type=str, required=False, default="test")
    parser.add_argument("--iteration", type=int, required=False, default=-1)
    return parser.parse_args()

class Evaluator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.config = self.load_config(args.model_path)
        self.device = torch.device(int(torch.cuda.current_device()))
        self.model = self.load_model(args.model_path, args.iteration)
        self.eval_dataloader = self.load_eval_dataloader(args)
        self.lpips_fn = lpips.LPIPS(net="vgg").to(self.device)

    def load_config(self, model_path: str) -> OmegaConf:
        config_path = os.path.join(model_path, "config.py")
        config = OmegaConf.load(config_path)
        config.merge_with_dotlist(
            [f"{k}={v}" for k, v in self.args.__dict__.items() if v is not None]
        )
        config = edict(OmegaConf.to_container(config, resolve=True))
        if config.model.type.endswith("3D"):
            config.model.num_ch_2d = 3
            config.model.num_ch_3d = 6
            if config.dataset.train.add_background:
                config.model.num_ch_2d += 3
            if config.dataset.train.add_depth:
                config.model.num_ch_2d += 1
            self.model_type = '3d'
        elif config.model.type.endswith("2D"):
            config.model.num_in_ch = 3
            if config.dataset.train.add_background:
                config.model.num_in_ch += 3
            if config.dataset.train.add_depth:
                config.model.num_in_ch += 1
            self.model_type = '2d'
        else:
            raise ValueError(f"Unknown model type: {config.model.type}")
        return config

    def load_model(self, model_path: str, max_saved_iters: int = -1):
        model = build_module(self.config.model, MODELS).to(self.device)
        model_ckpt = [step for step in os.listdir(model_path) if "pt" in step]
        if max_saved_iters==-1:
            max_saved_iters = max(
                [int(fname.split("_")[-1].split(".")[0]) for fname in model_ckpt]
            )
        print(max_saved_iters)
        ckpt_path = os.path.join(model_path, f"model_step_{max_saved_iters}.pt")
        ckpt = torch.load(ckpt_path)
        model_state_dict = ckpt["model_state_dict"]
        model.load_state_dict(model_state_dict)
        del model_state_dict
        del ckpt
        print(f"load {ckpt_path}!")
        return model

    def load_eval_dataloader(self, args: argparse.Namespace):
        eval_dataset = build_module(self.config.dataset.test, DATASETS)
        eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
        return eval_dataloader

    def render(self, rgb, intr, extr, tgt_gs, tgt_masks, tgt_imgs):
        B, _, C, Gh, Gw = rgb.shape
        _, N, _, H, W = tgt_imgs.shape
        rgb = rearrange(rgb, "b f c h w -> (b f) c h w")
        gs = torch.concat([rgb, tgt_gs], dim=1)
        pred_images = gs_render(gs, intr, extr, H, W)
        tgt_masks = rearrange(tgt_masks, "b f c h w -> (b f) c h w")
        tgt_imgs = rearrange(tgt_imgs, "b f c h w -> (b f) c h w")
        pred_images = pred_images * tgt_masks + tgt_imgs * (1 - tgt_masks) 
        images = rearrange(pred_images, "(b f) c h w -> b f c h w", b=B)
        return images

    @torch.no_grad()
    def _compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict:
        """Compute all evaluation metrics for a batch."""
        flat_preds = predictions.flatten(start_dim=0, end_dim=1)
        flat_targets = targets.flatten(start_dim=0, end_dim=1)
        
        _psnr = psnr(flat_preds, flat_targets, data_range=1.0)
        _ssim = fused_ssim(flat_preds, flat_targets, data_range=1.0)
        _lpips = self.lpips_fn(flat_preds, flat_targets).mean()

        return {
            "psnr": _psnr.item(),
            "ssim": _ssim.item(),
            "lpips": _lpips.item(),
        }

    @torch.no_grad()
    def evaluate(self, args: argparse.Namespace):
        self.model.eval()

        total_lpips = {}
        total_psnr = {}
        total_ssim = {}
        total_count = {}
        t_list = []

        test_path = os.path.join(args.save_dir)

        with torch.no_grad():
            pbar = tqdm(self.eval_dataloader, desc="Evaluating")
            for batch in pbar:
                scene_id = batch["scene"][0]
                dataset_id = scene_id.split("/")[0]
                if dataset_id not in total_lpips:
                    total_lpips[dataset_id] = 0.0
                if dataset_id not in total_psnr:
                    total_psnr[dataset_id] = 0.0
                if dataset_id not in total_ssim:
                    total_ssim[dataset_id] = 0.0
                if dataset_id not in total_count:
                    total_count[dataset_id] = 0
                scene_save_path = os.path.join(test_path, f"{scene_id}")
                os.makedirs(scene_save_path, exist_ok=True)

                if self.model_type == '2d':
                    in_imgs = batch["input_images"].to(self.device).float()
                    tgt_imgs = batch["target_images"].to(self.device).float()
                    t0 = time.time()
                    pred_imgs = self.model(in_imgs)
                    t1 = time.time()
                else:
                    in_imgs = batch["input_images"].to(self.device).float()
                    tgt_imgs = batch["target_images"].to(self.device).float()
                    in_rgb = batch["input_rgb"].to(self.device).float()
                    tgt_rgb = batch["target_rgb"].to(self.device).float()
                    projs = batch["projs"].to(self.device).int()  
                    render_intr = batch["render_intrinsics"].to(self.device).float()
                    render_extr = batch["render_extrinsics"].to(self.device).float()
                    render_imgs = batch["render_images"].to(self.device).float()
                    render_masks = batch["render_masks"].to(self.device).float()
                    render_gs = batch["render_gs"].to(self.device).float()
                    render_gs_path = batch["render_gs_path"][0]
                    t0 = time.time()
                    out_imgs, pred_rgb = self.model(in_imgs, in_rgb, projs)
                    t1 = time.time()
                    pred_imgs = self.render(pred_rgb, render_intr, render_extr, render_gs, render_masks, render_imgs)
                    # save_ply(render_gs_path, pred_rgb, os.path.join(scene_save_path, f"point_cloud_pred.ply"))

                context_index = batch["input_index"][0]
                target_index = batch["target_index"][0]

                V = tgt_imgs.shape[1]
                t_list.append((t1 - t0) / V)

                metrics = self._compute_metrics(pred_imgs, tgt_imgs)
                total_lpips[dataset_id] += metrics["lpips"]
                total_psnr[dataset_id] += metrics["psnr"]
                total_ssim[dataset_id] += metrics["ssim"]  

                # B=1
                pred_imgs = pred_imgs.squeeze(0)
                tgt_imgs = tgt_imgs.squeeze(0)
                in_imgs = in_imgs.squeeze(0)
                for i in range(pred_imgs.shape[0]):
                    torchvision.utils.save_image(
                        pred_imgs[i],
                        os.path.join(
                            scene_save_path, f"ours_{target_index[i]}.png"
                        ),
                    )
                    torchvision.utils.save_image(
                        tgt_imgs[i],
                        os.path.join(scene_save_path, f"gt_{target_index[i]}.png"),
                    )
                    torchvision.utils.save_image(
                        in_imgs[i][:3, :, :],
                        os.path.join(
                            scene_save_path, f"input_{context_index[i]}.png"
                        ),
                    )

                total_count[dataset_id] += 1

            
            for key in total_lpips.keys():
                total_lpips[key] /= total_count[key]
                total_psnr[key] /= total_count[key] 
                total_ssim[key] /= total_count[key]
                print(
                    f"LPIPS: {total_lpips[key]:.3f}, PSNR: {total_psnr[key]:.3f}, SSIM: {total_ssim[key]:.3f}"
                )

            t = np.array(t_list[5:])
            fps = 1.0 / t.mean()

            # Save metrics to a text file
            metrics_path = os.path.join(test_path, "metrics.txt")
            with open(metrics_path, "w") as f:
                for key in total_lpips.keys():
                    f.write(f"DATASET: {key}:\n")
                    f.write(f"LPIPS: {total_lpips[key]:.3f}\n")
                    f.write(f"PSNR: {total_psnr[key]:.3f}\n")
                    f.write(f"SSIM: {total_ssim[key]:.3f}\n")
                f.write(f"FPS: {fps:.3f}\n")
def main():
    args = parse_args()
    evaluator = Evaluator(args)
    evaluator.evaluate(args)


if __name__ == "__main__":
    main()
