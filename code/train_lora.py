import os
from pathlib import Path
import yaml

import torch
from torch.utils.data import DataLoader

from model.diffusers_model import StableDiffusion
from dataset.image_caption import ImageCaptionDataset, collate_fn
from trainer.diffusion_trainer import DiffusionTrainer

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    lora_cfg_path = os.path.join(ROOT, "configs", "lora_config.yaml")
    train_cfg_path = os.path.join(ROOT, "configs", "train_config.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load YAML
    with open(train_cfg_path, "r", encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)
    
    # Load model and prepare it
    sd = StableDiffusion(train_cfg["model"], device=device)
    
    sd.add_style_token(train_cfg['token']['style_token'], ref_words=train_cfg['token']['ref_words'])
    sd.freeze_pipeline()
    sd.attach_lora(lora_cfg_path)  # UNet + TE via PEFT
    print("#" * 50)
    sd.print_trainable_summary(list_names=False)
    print("#" * 50)

    # Build dataset from config
    ds = ImageCaptionDataset(
        image_dirs=[os.path.join(ROOT, train_cfg["data"]["datasets_dirs" ][i]) for i in range(len(train_cfg["data"]["datasets_dirs"]))],
        captions_path=train_cfg["data"]["captions_path"],
        resolution=train_cfg["data"]["resolution"],
        center_crop=train_cfg["data"].get("center_crop", True),
        random_flip=train_cfg["data"].get("random_flip", True),
        default_caption=train_cfg["data"].get("default_caption"),
        style_token=train_cfg["data"].get("style_token"),
        caption_dropout=train_cfg["data"].get("caption_dropout", 0.0),
        class_word=train_cfg["data"].get("class_word", "illustration")
    )

    # DataLoader
    loader = DataLoader(
        ds,
        batch_size=train_cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # Create trainer
    trainer = DiffusionTrainer(
        sd,
        dataloader=loader,
        device=device,
        lr=float(train_cfg["train"]["lr"]),
        unet_lr=float(train_cfg["train"].get("unet_lr", float(train_cfg["train"]["lr"]))),
        te_lr=float(train_cfg["train"].get("te_lr", float(train_cfg["train"]["lr"]) * 0.1)),
        snr_gamma=float(train_cfg["train"].get("snr_gamma", 0.5)),  # Default SNR gamma
        weight_decay=float(train_cfg["train"]["weight_decay"]),
        max_steps=train_cfg["train"]["max_steps"],
        grad_accum_steps=train_cfg["train"]["grad_accum_steps"],
        mixed_precision=train_cfg["train"].get("mixed_precision", "fp16"),
        clip_grad_norm=train_cfg["train"].get("clip_grad_norm", 1.0),
        sample_prompts=train_cfg["train"].get("sample_prompts", ["a busy market, in <sks> style"]),
        sample_every=train_cfg["train"].get("sample_every", 200),
        embed_param_group=train_cfg["train"].get("embed_param_group", None),
        runs_root=os.path.join(ROOT,train_cfg["train"].get("runs_root", "runs")),
        lr_scheduler_type=train_cfg["train"].get("lr_scheduler", "cosine"),
        warmup_steps=train_cfg["train"].get("warmup_steps", 50),
        noise_offset=train_cfg["train"].get("noise_offset", 0.0),
        datasets=[os.path.join(ROOT, train_cfg["data"]["datasets_dirs" ][i]) for i in range(len(train_cfg["data"]["datasets_dirs"]))],
    )
    # Start training
    trainer.train(resume_lora=train_cfg["train"]["resume"].get("lora_path"),
                  resume_state=train_cfg["train"]["resume"].get("state_path"))