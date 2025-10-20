import argparse
import gc
import logging
import os
import shutil
import math
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
import warnings
from easydict import EasyDict as edict
from omegaconf import OmegaConf
from registry import build_module, DATASETS, LOSSES, MODELS
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchmetrics.functional import (
    peak_signal_noise_ratio as psnr,
    structural_similarity_index_measure as fused_ssim,
)
from tqdm import tqdm
from train_helpers import (
    amp_dtype_mapping,
    calculate_grad_norm,
    create_optimizer,
    create_scheduler,
    EMA,
    get_gpu_memory,
    save_and_log_images,
    save_and_log_rgb,
    saveRuntimeCode,
)
from utils.config_utils import (
    format_numel_str,
    get_model_numel,
    init_dist,
    set_seed,
    setup_logger,
)

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train model in distributed mode")

    # Model configuration
    parser.add_argument(
        "--config", type=str, required=True, help="Path to model config file"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    return parser.parse_args()


class Trainer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        
        self.config = self.load_config(args.config)
        self.setup_distributed()
        self.logger, self.exp_dir = self.setup_logger()
        self.initialize_dataloaders()
        self.model = self.initialize_model()
        self.optimizer, self.scheduler, self.scaler = self.initialize_optimizer()
        self.resume_checkpoint()
        if self.model_type=='3d':
            self.load_2d_model()
        self.wandb_run = self.initialize_wandb()
        self.loss_fn = self.initialize_loss_functions()
        self.ema = EMA(beta=self.config.training.ema_beta)
        self.ema_loss = None

    def load_config(self, config_path: str) -> OmegaConf:
        config = OmegaConf.load(config_path)
        if "base_config" in config:
            base_config = OmegaConf.load(config.base_config)
            config = OmegaConf.merge(base_config, config)
        config.merge_with_dotlist(
            [f"{k}={v}" for k, v in self.args.__dict__.items() if v is not None]
        )
        config = edict(OmegaConf.to_container(config, resolve=True))
        if config.model.type.endswith("3D"):
            config.model.img_size_2d = config.model2d.img_size
            config.model.patch_size_2d = config.model2d.patch_size
            config.model.embed_dim_2d = config.model2d.embed_dim
            config.model.depths_2d = config.model2d.depths
            config.model.num_heads_2d = config.model2d.num_heads
            config.model.window_size_2d = config.model2d.window_size
            config.model.num_in_ch_2d = 3
            config.model.num_in_ch_3d = 6
            if config.dataset.train.add_background:
                config.model.num_in_ch_2d += 3
            if config.dataset.train.add_depth:
                config.model.num_in_ch_2d += 1
            self.model_type = '3d'
            self.find_unused_parameters = True
            self.render_2d = config.dataset.train.render_2d
        elif config.model.type.endswith("2D"):
            config.model.num_in_ch = 3
            if config.dataset.train.add_background:
                config.model.num_in_ch += 3
            if config.dataset.train.add_depth:
                config.model.num_in_ch += 1
            self.model_type = '2d'
            self.find_unused_parameters = False
        else:
            raise ValueError(f"Unknown model type: {config.model.type}")
        return config

    def setup_distributed(self) -> None:
        # set up for distributed environment
        init_dist(self.config)
        self.device = torch.device(torch.cuda.current_device())
        self.local_rank = self.config.local_rank
        self.is_master = dist.get_rank() == 0
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        # set distributed seed
        seed = torch.zeros(1, device=self.device)
        if self.is_master:
            if self.config.get("seed", None) != None:
                seed = torch.tensor([self.config.seed], device=self.device)
            else:
                seed = torch.randint(
                    low=20000,
                    high=30000,
                    size=[
                        1,
                    ],
                    device="cuda",
                )
                self.config.seed = seed.item()

        # broadcast seed to all device
        dist.broadcast(seed, src=0)
        
        # notice that per gpu seed should vary.
        seed = int(seed.item()) + dist.get_rank()
        set_seed(seed)

    def setup_logger(self) -> tuple[logging.Logger, str]:
        logger, exp_dir = setup_logger(
            self.config.log_dir, self.config.experiment_name, self.is_master
        )
        if self.is_master:
            shutil.copy(self.args.config, os.path.join(exp_dir, "config.py"))
            saveRuntimeCode(os.path.join(exp_dir, "backup"))
            logger.info(f"Starting experiment: {self.config.experiment_name}")
            logger.info(f"Experiment directory: {exp_dir}")
            logger.info(f"Configuration: {self.config}")
            logger.info(f"Device: {self.device}")
            logger.info(f"Initial GPU Memory: {get_gpu_memory()}")
        return logger, exp_dir
    
    def initialize_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        trainset = build_module(self.config.dataset.train, DATASETS)

        self.train_sampler = DistributedSampler(
            trainset, self.world_size, self.rank, shuffle=True
        )
        loader_kwargs = {
            "batch_size": self.config.training.batch_size,
            "num_workers": self.config.training.num_workers,
            "pin_memory": True,
            "shuffle": False,
        }
        self.train_dataloader = DataLoader(
            trainset,
            sampler=self.train_sampler,
            prefetch_factor=self.config.training.num_workers,
            **loader_kwargs,
        )

        if self.is_master:
            train_size = len(trainset)
            total_batch_size = self.config.training.batch_size
            total_batch_size *= dist.get_world_size()
            self.logger.info(f"Train dataset size: {train_size}")
            self.logger.info(f"Total batch size: {total_batch_size}")

    def initialize_model(self):
        model = build_module(self.config.model, MODELS).to(self.device)
        model = DDP(
            model, 
            device_ids=[self.local_rank],
            find_unused_parameters=self.find_unused_parameters,
        )
        if self.is_master:
            model_numel, model_numel_trainable = get_model_numel(model)
            self.logger.info(
                f"Trainable model params: {format_numel_str(model_numel_trainable)}, "
                f"Total model params: {format_numel_str(model_numel)}"
            )
        self.param_optim_dict = {
            n: p for n, p in model.named_parameters() if p.requires_grad
        }
        self.param_optim_list = [p for p in self.param_optim_dict.values()]
        return model

    def initialize_optimizer(
        self,
    ) -> tuple[torch.optim.AdamW, torch.optim.lr_scheduler.LambdaLR, GradScaler]:
        param_update_steps = int(
            self.config.training.max_iterations / self.config.training.grad_accum_steps
        )
        optimizer = create_optimizer(
            self.model,
            self.config.training.weight_decay,
            self.config.training.lr,
            (self.config.training.beta1, self.config.training.beta2),
        )
        scheduler = create_scheduler(
            optimizer,
            param_update_steps,
            self.config.training.warmup_steps,
            self.config.training.get("scheduler_type", "cosine"),
        )
        scaler = GradScaler()
        return optimizer, scheduler, scaler

    def save_checkpoint(self) -> None:
        checkpoint = {
            "model_state_dict": self.model.module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "step": self.global_step,
        }
        model_path = os.path.join(self.exp_dir, f"model_step_{self.global_step}.pt")
        torch.save(checkpoint, model_path)
        self.logger.info(f"Model saved at step {self.global_step} to {model_path}")
    
    def load_2d_model(self) -> None:
        checkpoint = torch.load(self.config.training.swint2d_model_path, map_location=self.device)
        if isinstance(self.model, DDP):
            self.model.module.swin2d.load_state_dict(checkpoint["model_state_dict"])

            for param in self.model.module.swin2d.parameters():
                param.requires_grad = False
        else:
            self.model.swin2d.load_state_dict(checkpoint["model_state_dict"])

            for param in self.model.swin2d.parameters():
                param.requires_grad = False
                
    def resume_checkpoint(self) -> None:
        resume_file = self.config.training.get("resume_ckpt", None)
        if resume_file is None:
            self.logger.info("No checkpoint founded, start from scratch")
            self.global_step = 0
        else:
            self.logger.info(f"Resume from checkpoint: {resume_file}")
            checkpoint = torch.load(resume_file, map_location=self.device)
            if isinstance(self.model, DDP):
                status = self.model.module.load_state_dict(
                    checkpoint["model_state_dict"], strict=False
                )
            else:
                status = self.model.load_state_dict(
                    checkpoint["model_state_dict"], strict=False
                )
            self.logger.info(f"Loaded model with status: {status}")

            if self.config.training.resume_optimizer:
                train_steps_done = checkpoint["step"]
                self.logger.info(f"Resume from train_steps_done: {train_steps_done}")
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                self.logger.info(f"Loaded optimizer and scheduler")
                self.global_step = train_steps_done
            else:
                self.global_step = 0  
        dist.barrier()

    def initialize_wandb(self) -> wandb.sdk.wandb_run.Run:
        if self.is_master and self.config.use_wandb:
            
            wandb.login(key="6e56e085446849b249cc2a6fc43615451a308651")  # 替换为你的实际 API 密钥

            wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=self.config.experiment_name,
                config=self.config,
            )
            wandb_run.watch(self.model.module, log="all", log_freq=100)
            self.logger.info(f"Initialized wandb")
            return wandb_run
        return None

    def initialize_loss_functions(self) -> tuple[nn.Module, nn.Module]:
        loss_fn = dict()
        loss_fn = build_module(self.config.loss, LOSSES)
        return loss_fn

    def train_one_step_3d(self, batch: dict):
        in_imgs = batch["input_images"].to(self.device).float()
        tgt_imgs = batch["target_images"].to(self.device).float()
        in_rgb = batch["input_rgb"].to(self.device).float()
        tgt_rgb = batch["target_rgb"].to(self.device).float()
        projs = batch["projs"].to(self.device).int()  
        render_intr = batch["render_intrinsics"].to(self.device).float() if self.render_2d else None
        render_extr = batch["render_extrinsics"].to(self.device).float() if self.render_2d else None
        render_imgs = batch["render_images"].to(self.device).float() if self.render_2d else None
        render_masks = batch["render_masks"].to(self.device).float() if self.render_2d else None
        render_gs = batch["render_gs"].to(self.device).float() if self.render_2d else None
        pred_imgs, pred_rgb = self.model(in_imgs, in_rgb, projs)
        loss_dict = self.loss_fn(pred_rgb, tgt_rgb, render_intr, render_extr, render_gs, render_masks, render_imgs)
        train_psnr = psnr(
            pred_rgb.flatten(start_dim=0, end_dim=1), 
            tgt_rgb.flatten(start_dim=0, end_dim=1),
            data_range=1.0,
        )
        return in_imgs, pred_imgs,tgt_imgs, in_rgb, pred_rgb, tgt_rgb, loss_dict, train_psnr

    def train_one_step_2d(self, batch: dict):
        in_img = batch["input_images"].to(self.device).float()
        tgt_img = batch["target_images"].to(self.device).float()
        pred_img = self.model(in_img)
        loss_dict = self.loss_fn(pred_img, tgt_img)
        train_psnr = psnr(
            pred_img.flatten(start_dim=0, end_dim=1), 
            tgt_img.flatten(start_dim=0, end_dim=1),
            data_range=1.0,
        )
        return in_img, pred_img, tgt_img, loss_dict, train_psnr

    def train(self) -> None:
        # clear cache before launching training
        gc.collect()
        torch.cuda.empty_cache()

        max_iterations = self.config.training.max_iterations - self.global_step
        grad_accum_steps = self.config.training.grad_accum_steps
        train_iter = iter(self.train_dataloader)
        pbar = tqdm(range(max_iterations), desc="Training", disable=(self.rank != 0))

        for step in pbar:

            if self.train_sampler is not None:
                self.train_sampler.set_epoch(step)
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_dataloader)
                batch = next(train_iter)

            self.optimizer.zero_grad()

            torch.cuda.synchronize()
            update_param = (self.global_step + 1) % grad_accum_steps == 0
            context = torch.autocast(
                enabled=self.config.use_amp,
                device_type="cuda",
                dtype=amp_dtype_mapping[self.config.amp_dtype],
            )
            if not update_param:
                context = self.model.no_sync(), context
            with context:
                if self.model_type == "2d":
                    in_img, pred_img, tgt_img, loss_dict, train_psnr = self.train_one_step_2d(batch)
                else:
                    in_imgs, pred_imgs, tgt_imgs, in_rgb, pred_rgb, tgt_rgb, loss_dict, train_psnr = self.train_one_step_3d(batch)

            torch.cuda.synchronize()

            # Backward & update
            loss = loss_dict["loss"]
            self.scaler.scale(loss / grad_accum_steps).backward()

            skip_optimizer_step = False
            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.warning(f"NaN or Inf loss detected, skip this iteration")
                skip_optimizer_step = True
                loss_dict["loss"] = torch.tensor(0.0).to(self.device)

            torch.cuda.synchronize()
            if update_param and (not skip_optimizer_step):
                # Unscales the gradients of optimizer's assigned parameters in-place
                self.scaler.unscale_(self.optimizer)
                with torch.no_grad():
                    for n, p in self.param_optim_dict.items():
                        if p.grad is not None:
                            p.grad.nan_to_num_(nan=0.0, posinf=1e-3, neginf=-1e-3)
                total_grad_norm = 0.0

                if self.config.training.grad_clip_norm > 0:
                    grad_clip_norm = self.config.training.grad_clip_norm
                    total_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.param_optim_list, max_norm=grad_clip_norm
                    ).item()
                    allowed_gradnorm = grad_clip_norm * self.config.training.get(
                        "allowed_gradnorm_factor", 5.0
                    )
                    if total_grad_norm > allowed_gradnorm:
                        skip_optimizer_step = True
                        self.logger.warning(
                            f"step {self.global_step} grad norm too large {total_grad_norm} > {allowed_gradnorm}, skipping optimizer step"
                        )
                if not skip_optimizer_step:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.scaler.update()
                
                self.scheduler.step()
            torch.cuda.synchronize()

            loss_tensor = torch.tensor(loss.item(), device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            loss_reduced = loss_tensor.item() / self.world_size

            self.ema_loss = self.ema.update(self.ema_loss, loss_reduced)
            current_lr = self.scheduler.get_last_lr()[0]

            if self.is_master:
                pbar.set_postfix(
                    {
                        "psnr": f"{train_psnr.item():.4f}",
                        "total_loss": f"{loss_reduced:.4f}",
                        "ema": (
                            f"{self.ema_loss:.4f}"
                            if self.ema_loss is not None
                            else "N/A"
                        ),
                    }
                )
                if self.wandb_run is not None:
                    log_dict = {
                        "step_loss": loss_reduced,
                        "ema_loss": self.ema_loss,
                        "psnr": train_psnr.item(),
                        "learning_rate": current_lr,
                        "grad_norm": total_grad_norm,
                    }
                    self.wandb_run.log(
                        {f"train/{k}": v for k, v in log_dict.items()},
                        step=self.global_step,
                    )
                    
                if step % self.config.training.vis_every == 0:
                    if self.model_type == "2d":
                        save_and_log_images(in_img, tgt_img, pred_img, os.path.join(self.exp_dir, f"global_step_{self.global_step}"), self.global_step, split="train")
                    else:
                        save_and_log_images(in_imgs, tgt_imgs, pred_imgs, os.path.join(self.exp_dir, f"global_step_{self.global_step}"), self.global_step, split="train")
                        save_and_log_rgb(in_rgb, tgt_rgb, pred_rgb, os.path.join(self.exp_dir, f"global_step_{self.global_step}"), self.global_step, split="train")
    
                if (
                    step % self.config.training.save_interval == 0
                    or step == max_iterations - 1
                ):
                    self.logger.info(f"Saving checkpoint at step {step}")
                    self.save_checkpoint()

            self.global_step += 1

            dist.barrier()

        if self.is_master and self.wandb_run is not None:
            self.wandb_run.finish()

        dist.destroy_process_group()

def main():
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
