import os
import random
import math
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime
import uuid
import shutil
import numbers
from contextlib import contextmanager
import json
from tqdm import tqdm

import numpy as np
import torch
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from diffusers.training_utils import compute_snr
from diffusers import DPMSolverMultistepScheduler
from diffusers import DDPMScheduler


def _torch_save(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)

def _torch_load(path, device="cpu"):
    return torch.load(path, map_location=device)

class DiffusionTrainer:
    def __init__(
        self,
        sd,
        dataloader: DataLoader,
        device: str = "cuda",
        lr: float = 1e-4,
        unet_lr: float = 1e-4,
        te_lr: float = 5e-5,
        snr_gamma: float = 0.5,
        noise_offset: float = 0.0,
        weight_decay: float = 0.00,
        max_steps: int = 800,
        grad_accum_steps: int = 8,
        mixed_precision: str = "fp16",      # "fp16" | "bf16" | "no"
        clip_grad_norm: float = 1.0,
        sample_prompts: Optional[List[str]] = None,
        sample_every: int = 200,
        embed_param_group: Optional[Dict] = None,
        experiment_name: Optional[str] = None,          
        runs_root: str = "runs",
        lr_scheduler_type: str = "cosine",   # "cosine" | "constant"
        warmup_steps: int = 50,
        datasets: Optional[List[str]] = None,                   
    ):
        
        self.sd = sd
        self.device = device
        self.pipe = sd.pipe
        self.pipe.to(device)
        self.unet = self.pipe.unet
        self.vae = self.pipe.vae
        self.text_encoder = self.pipe.text_encoder
        self.tokenizer = self.pipe.tokenizer

        self.train_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
            prediction_type=getattr(self.pipe.scheduler.config, "prediction_type", "epsilon"),
            variance_type="fixed_small",
        )
        self.scheduler = self.train_scheduler
        #self.scheduler = self.pipe.scheduler

        self.dataloader = dataloader
        self.lr = lr
        self.unet_lr = unet_lr
        self.te_lr = te_lr
        self.snr_gamma = snr_gamma
        self.weight_decay = weight_decay
        self.max_steps = max_steps
        self.noise_offset = noise_offset
        self.lr_scheduler_type = lr_scheduler_type
        self.warmup_steps = int(warmup_steps)
        self.grad_accum_steps = grad_accum_steps
        self.mixed_precision = mixed_precision
        self.clip_grad_norm = clip_grad_norm

        self.sample_prompts = sample_prompts or ["a busy market, in <sks> style"]
        self.sample_every = sample_every

        # robust VAE scaling factor
        self.latent_scaling_factor = getattr(getattr(self.vae, "config", None), "scaling_factor", 0.18215)


        # Collect trainable parameters
        self._collect_trainable_parameters(embed_param_group)
        # Initialize optimizer
        self._init_optimizer()
        #self._sanitize_optimizer_param_groups(self.optimizer)
        bs = self._infer_batch_size()
        # Set run directory layout
        cfg = dict(
            lr=lr, unet_lr= unet_lr, te_lr=te_lr , batch_szie=bs, weight_decay=weight_decay, snr_gamma=snr_gamma,
            noise_offset=noise_offset, max_steps=max_steps, grad_accum_steps=grad_accum_steps,
            mixed_precision=mixed_precision, clip_grad_norm=clip_grad_norm,
            sample_every=sample_every, experiment_name=experiment_name, runs_root=runs_root,
            lr_scheduler_type=lr_scheduler_type, warmup_steps=warmup_steps, datasets=datasets
        )
        self.set_run_directory_layout(runs_root,experiment_name, cfg=cfg)

        # GradScaler (portable across torch versions)
        try:
            # torch.amp.GradScaler on >=2.2 supports device kwarg
            self.scaler = GradScaler(device="cuda", enabled=(self.mixed_precision == "fp16"))
        except TypeError:
            # fallback for older torch
            self.scaler = GradScaler(enabled=(self.mixed_precision == "fp16"))

        # Modes
        self.unet.train()
        self.text_encoder.train()
        self.vae.eval()

        from torch.optim.lr_scheduler import LambdaLR
        import math

        def warmup_cosine(step, warmup=50, max_steps=self.max_steps):
            if step < warmup:
                return float(step) / float(max(1, warmup))
            progress = (step - warmup) / float(max(1, max_steps - warmup))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=lambda s: warmup_cosine(s, warmup=self.max_steps))

    def _build_lr_scheduler(self):
        # Linear warmup to 1.0, then cosine to 0.0 over remaining steps
        def lr_lambda_global(step: int):
            if self.lr_scheduler_type == "constant":
                # Optional warmup even for "constant" (comment out if you want *truly* constant from step 0)
                if step < self.warmup_steps and self.warmup_steps > 0:
                    return float(step) / float(self.warmup_steps)
                return 1.0

            # cosine with warmup (default)
            if self.warmup_steps > 0 and step < self.warmup_steps:
                return float(step) / float(self.warmup_steps)

            # progress in [0,1]
            denom = max(1, self.max_steps - self.warmup_steps)
            progress = float(step - self.warmup_steps) / float(denom)
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        # One LambdaLR covering all param groups
        return LambdaLR(self.optimizer, lr_lambda=lr_lambda_global)


    def _infer_batch_size(self):
        bs = getattr(self.dataloader, "batch_size", None)
        if bs is None:
            bs = getattr(getattr(self.dataloader, "batch_sampler", None), "batch_size", None)
        return bs
    
    def set_run_directory_layout(self, runs_root: str, experiment_name: Optional[str] = None, cfg: Optional[Dict] = None):
        """Set the run directory layout with a new root and optional experiment name."""
        # ---------- run directory layout ----------
        os.makedirs(runs_root, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        suffix = experiment_name or uuid.uuid4().hex[:6]
        self.run_dir = Path(runs_root) / f"{ts}_{suffix}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # samples + weights inside the run directory
        self.samples_dir = self.run_dir / "samples"
        self.samples_dir.mkdir(parents=True, exist_ok=True)

        self.save_path = str(self.run_dir / "weights.safetensors")

        # optional: save a minimal config for reproducibility
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)
        self._snapshot_lora_configs()
   
    def json_sanitize(self, obj):
        if obj is None or isinstance(obj, (str, bool, numbers.Number)):
            return obj
        if isinstance(obj, dict):
            return {str(self.json_sanitize(k)): self.json_sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self.json_sanitize(x) for x in obj]
        if isinstance(obj, set):
            # make deterministic
            return [self.json_sanitize(x) for x in sorted(list(obj), key=lambda x: str(x))]
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        if isinstance(obj, (torch.dtype, torch.device)):
            return str(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # enums / dataclasses / anything else
        v = getattr(obj, "value", None)
        return self.json_sanitize(v) if v is not None else str(obj)

    def _snapshot_lora_configs(self):
        # Copy original YAML (if we know it)
        src = getattr(self.sd, "_last_lora_cfg_path", None)
        if src and isinstance(src, str) and os.path.exists(src):
            try:
                shutil.copy(src, self.run_dir / "lora_config.yaml")
            except Exception:
                pass

        # Save the *effective* configs that are actually attached to the models
        try:
            effective = self.sd.active_lora_configs()
        except Exception as e:
            effective = {"error": f"could not read active lora configs: {e}"}

        # Include a few handy stats
        meta = {
            "model_id": getattr(self.sd, "model_id", None),
            "effective_lora": effective,  # UNet/TE -> adapter name -> LoraConfig dict
            "trainable_params_total": int(sum(p.numel() for p in self.trainable_params)),
            "optimizer_lr_groups": [float(g.get("lr", self.lr)) for g in self.param_groups],
            "weight_decay_groups": [float(g.get("weight_decay", self.weight_decay)) for g in self.param_groups],
        }
        with open(self.run_dir / "lora_config.resolved.json", "w") as f:
            json.dump(self.json_sanitize(meta), f, indent=2)

    def _collect_trainable_parameters(self, embed_param_group=None):
        """
        Build clean, disjoint param groups:
        - UNet LoRA (lr=1e-4, wd=0.0)
        - TE   LoRA (lr=5e-5, wd=0.0)
        - Optional embed group (custom lr, wd=0.0)
        """
        def as_param_list(xs):
            # keep only trainable nn.Parameter, preserve order & dedupe
            seen = set()
            out = []
            for p in xs:
                if isinstance(p, torch.nn.Parameter) and p.requires_grad:
                    pid = id(p)
                    if pid not in seen:
                        seen.add(pid)
                        out.append(p)
            return out

        unet_params = as_param_list(self.unet.parameters())
        te_params   = as_param_list(self.text_encoder.parameters())

        groups = []
        if unet_params:
            groups.append({"name": "unet_lora", "params": unet_params, "lr": self.unet_lr, "weight_decay": self.weight_decay})
        if te_params:
            groups.append({"name": "te_lora",   "params": te_params,   "lr": self.te_lr, "weight_decay": self.weight_decay})

        if embed_param_group and "params" in embed_param_group:
            emb_params = as_param_list(embed_param_group["params"])
            if emb_params:
                emb_lr = float(embed_param_group.get("lr", self.lr * 0.05))
                groups.append({"name": "embeds", "params": emb_params, "lr": emb_lr, "weight_decay": 0.0})

        # Validate: no duplicates across groups
        seen = set()
        for g in groups:
            for p in g["params"]:
                pid = id(p)
                if pid in seen:
                    raise ValueError(f"Parameter appears in multiple groups (group={g['name']})")
                seen.add(pid)

        if not groups:
            raise ValueError("No trainable parameters found (are LoRA adapters attached?)")

        # Store for optimizer (strip names) + convenience
        self.param_groups = [{k: v for k, v in g.items() if k != "name"} for g in groups]
        self._param_group_names = [g["name"] for g in groups]
        self.trainable_params = [p for g in groups for p in g["params"]]

        # Logging
        print("=== Param groups ===")
        for name, g in zip(self._param_group_names, self.param_groups):
            nparams = sum(p.numel() for p in g["params"])
            print(f"{name}: lr={g['lr']} wd={g['weight_decay']} trainable={nparams:,}")

    def _init_optimizer(self):
        # Ensure numeric types
        for g in self.param_groups:
            g["lr"] = float(g.get("lr", self.lr))
            g["weight_decay"] = float(g.get("weight_decay", self.weight_decay))

        # Try fused AdamW if available; fall back cleanly
        use_fused = False
        if torch.cuda.is_available():
            try:
                major, _ = torch.cuda.get_device_capability()
                use_fused = major >= 8  # Ampere/Hopper
            except Exception:
                use_fused = False

        kwargs = {"foreach": False}
        try:
            self.optimizer = torch.optim.AdamW(self.param_groups, fused=use_fused, **kwargs)
        except Exception:
            # older torch builds without 'fused' kw or unsupported device
            self.optimizer = torch.optim.AdamW(self.param_groups, **kwargs)

        print("=== Optimizer groups ===")
        for i, g in enumerate(self.optimizer.param_groups):
            nparams = sum(p.numel() for p in g["params"])
            print(f"Group {i}: lr={g['lr']} wd={g['weight_decay']} trainable={nparams:,}")


    def _encode_text(self, texts):

        tok = self.tokenizer(texts,padding="max_length",
                            truncation=True,
                            max_length=self.tokenizer.model_max_length,  # 77 for SD1.5 CLIP
                            return_tensors="pt",)
        input_ids = tok["input_ids"].to(self.unet.device)
        attention_mask = tok["attention_mask"].to(self.unet.device)
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state.to(dtype=self.unet.dtype)

    @torch.no_grad()
    def _encode_images_to_latents(self, pixel_values: torch.Tensor):
        pixel_values = pixel_values.to(self.unet.device, dtype=self.unet.dtype)
        latents = self.vae.encode(pixel_values).latent_dist.sample() * self.latent_scaling_factor
        return latents

    def _loss_target(self, latents, noise, timesteps):
        pred_type = getattr(getattr(self.scheduler, "config", None), "prediction_type", "epsilon")
        if pred_type == "epsilon":
            return noise
        if pred_type in ("v_prediction", "v-prediction", "v"):
            alphas_cumprod = self.scheduler.alphas_cumprod.to(latents.device, latents.dtype)
            a_t = alphas_cumprod[timesteps].sqrt().view(-1, 1, 1, 1)
            one_minus = (1.0 - alphas_cumprod[timesteps]).sqrt().view(-1, 1, 1, 1)
            return a_t * noise - one_minus * latents
        return noise

    # ---------- training ----------

    def train(self, resume_state: str | None = None, resume_lora: str | None = None):
        # Optionally resume training state
        start_step = self.load_checkpoint(state_path=resume_state, lora_path=resume_lora)

        step = start_step
        pbar = tqdm(total=self.max_steps, desc="Training LoRA")
        data_iter = iter(self.dataloader)

        while step < self.max_steps:
            self.optimizer.zero_grad(set_to_none=True)

            for _ in range(self.grad_accum_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.dataloader)
                    batch = next(data_iter)

                pixel_values = batch["pixel_values"]
                texts = batch["text"]
                B = pixel_values.size(0)
                timesteps = torch.randint(
                    0, self.scheduler.config.num_train_timesteps,
                    (B,), device=self.unet.device, dtype=torch.long
                )

                latents = self._encode_images_to_latents(pixel_values)
                noise = torch.randn_like(latents)
                if self.noise_offset is None or self.noise_offset <= 0.0:
                    # plain noise offset
                    # noise = noise + self.noise_offset * torch.randn_like(noise)  
                    # channel-wise noise offset
                    offset = self.noise_offset * torch.randn(noise.size(0), noise.size(1), 1, 1, device=noise.device, dtype=noise.dtype)
                    noise = noise + offset 
                noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
                text_states = self._encode_text(texts)

                use_autocast = (self.mixed_precision in ("fp16", "bf16"))
                dtype = torch.float16 if self.mixed_precision == "fp16" else (
                    torch.bfloat16 if self.mixed_precision == "bf16" else None
                )
                with autocast(device_type=self.device, enabled=use_autocast, dtype=dtype):
                    model_pred = self.unet(noisy_latents, timesteps, text_states)[0]
                    target = self._loss_target(latents, noise, timesteps)
                    
                    if self.snr_gamma is None or self.snr_gamma <= 0.0:
                        loss = torch.nn.functional.mse_loss(model_pred, target)
                    else:
                        snr = compute_snr(self.scheduler, timesteps)
                        w = torch.minimum(snr, self.snr_gamma * torch.ones_like(snr)) / snr
                        mse = torch.nn.functional.mse_loss(model_pred, target, reduction="none")
                        mse = mse.mean(dim=list(range(1, mse.ndim)))
                        loss = (w * mse).mean()

                if self.scaler.is_enabled():
                    self.scaler.scale(loss / self.grad_accum_steps).backward()
                else:
                    (loss / self.grad_accum_steps).backward()

            if self.clip_grad_norm and self.clip_grad_norm > 0:
                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.trainable_params, self.clip_grad_norm)

            if self.scaler.is_enabled():
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            # Step the LR scheduler once per optimizer step
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            step += 1
            pbar.update(1)
            pbar.set_postfix(loss=float(loss.detach().item()))

            if step % self.sample_every == 0:
                #self.sd.save_lora_weights(self.save_path)
                u_train, te_train = self.unet.training, self.text_encoder.training
                self.unet.eval(); self.text_encoder.eval()
                try:
                    self._sample_and_save(step)
                    self.save_checkpoint(step)
                finally:
                    self.unet.train(u_train); self.text_encoder.train()

        pbar.close()
        self.sd.save_lora_weights(self.save_path)

    @contextmanager
    def use_inference_scheduler(self, pipe, base_config, scheduler_cls=DPMSolverMultistepScheduler):
        orig = pipe.scheduler
        pipe.scheduler = scheduler_cls.from_config(base_config)
        try:
            assert pipe.scheduler.config.prediction_type == base_config.prediction_type
            yield
        finally:
            pipe.scheduler = orig
    
    def save_checkpoint(self, step: int):
        """Save LoRA weights + training state to resume later."""
        # 1) LoRA weights (portable)
        lora_path = self.run_dir / f"lora_step_{step:06d}.safetensors"
        self.sd.save_lora_weights(str(lora_path))

        # 2) Training state (optimizer/scheduler/scaler/etc.)
        state = {
            "step": int(step),
            "optimizer": self.optimizer.state_dict(),
            "scaler": (self.scaler.state_dict() if self.scaler is not None else None),
            "lr_scheduler": (self.lr_scheduler.state_dict() if getattr(self, "lr_scheduler", None) else None),
            "param_group_lrs": [pg.get("lr") for pg in self.optimizer.param_groups],
            "rng": {
                "python": random.getstate(),
                "numpy": None,  # fill if you care: np.random.get_state()
                "torch": torch.get_rng_state(),
                "cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
            },
        }
        state_path = self.run_dir / f"train_state_step_{step:06d}.pt"
        _torch_save(state, state_path)
        print(f"[ckpt] saved {lora_path.name} & {state_path.name}")

    def load_checkpoint(self, state_path: str | Path | None = None,
                        lora_path: str | Path | None = None) -> int:
        """
        Load training state. If only lora_path is given, we load LoRA weights and
        return step=0 (soft resume). If state_path exists, restore optimizer/scaler/scheduler and step.
        """
        start_step = 0

        # Always (re)load LoRA weights if provided
        if lora_path is not None and os.path.exists(lora_path):
            # Make sure LoRA adapters are attached with the SAME config (rank, targets like to_out.0)
            self.sd.load_lora_weights(lora_path)
            print(f"[ckpt] loaded LoRA weights from {lora_path}")

        # Restore training state if available
        if state_path is not None and os.path.exists(state_path):
            state = _torch_load(state_path, device=self.unet.device)
            self.optimizer.load_state_dict(state["optimizer"])
            if self.scaler is not None and state.get("scaler") is not None:
                self.scaler.load_state_dict(state["scaler"])
            if getattr(self, "lr_scheduler", None) and state.get("lr_scheduler") is not None:
                self.lr_scheduler.load_state_dict(state["lr_scheduler"])
            start_step = int(state.get("step", 0))

            # (Optional) restore RNG for determinism
            try:
                random.setstate(state["rng"]["python"])
                torch.set_rng_state(state["rng"]["torch"])
                if torch.cuda.is_available() and state["rng"]["cuda"] is not None:
                    torch.cuda.set_rng_state_all(state["rng"]["cuda"])
            except Exception:
                pass

            print(f"[ckpt] restored training state from {state_path} @ step {start_step}")

        return start_step

    @torch.no_grad()
    def _sample_and_save(self, step: int):
        # IMPORTANT: base config from the *training* scheduler
        base_cfg = self.scheduler.config

        with self.use_inference_scheduler(self.sd.pipe, base_cfg, DPMSolverMultistepScheduler):
            neg_str = ("blurry, faceless, deformed hands, extra fingers, film grain, "
            "speckles, water droplets, banding, jpeg artifacts, overexposed highlights")

            negatives = [neg_str] * len(self.sample_prompts)   # <-- match batch size & type
            imgs = self.sd.generate_image(
                self.sample_prompts,
                num_inference_steps=60,
                guidance_scale=7.5,   # try 9–11 for illustration styles
                guidance_rescale=0.7,             # if your diffusers version supports it
                negative_prompt=negatives,
                height=512,
                width=512,
            )

        for i, im in enumerate(imgs):
            im.save(self.samples_dir / f"step_{step:04d}_{i:02d}.png")
