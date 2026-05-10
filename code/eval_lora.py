import os
from pathlib import Path
from contextlib import contextmanager

import torch
from model.diffusers_model import StableDiffusion
from diffusers import DPMSolverMultistepScheduler

def set_global_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@contextmanager
def use_inference_scheduler(pipe, base_config, scheduler_cls=DPMSolverMultistepScheduler):
    orig = pipe.scheduler
    pipe.scheduler = scheduler_cls.from_config(base_config)
    try:
        assert pipe.scheduler.config.prediction_type == base_config.prediction_type
        yield
    finally:
        pipe.scheduler = orig

@torch.no_grad()
def sample_and_save(output_dir, sd, prompts):
    # IMPORTANT: base config from the *training* scheduler
    base_cfg = sd.pipe.scheduler.config

    with use_inference_scheduler(sd.pipe, base_cfg, DPMSolverMultistepScheduler):
        neg_str = ("blurry, faceless, deformed hands, extra fingers, film grain, "
           "speckles, water droplets, banding, jpeg artifacts, overexposed highlights")

        negatives = [neg_str] * len(prompts)   # <-- match batch size & type
        imgs = sd.generate_image(
            prompts,
            num_inference_steps=60,
            guidance_scale=7.5,   # try 9–11 for illustration styles
            guidance_rescale=0.7,             # if your diffusers version supports it
            negative_prompt=negatives,
            height=512,
            width=512,
        )
    for i, img in enumerate(imgs):
        img.save(os.path.join(output_dir, f"{'_'.join(prompts[i].split(' '))}_{i}.png"))

# ---------- demo ----------
if __name__ == "__main__":
    model_name = "runwayml/stable-diffusion-v1-5"
    style_token = "<sks>"
    ROOT = Path(__file__).resolve().parent.parent
    set_global_seed(42)
    # seed 42, 90
    # weight_name = "lora_step_002800.safetensors"
    sd = StableDiffusion(model_name, "cuda" if torch.cuda.is_available() else "cpu")
    # Print initial summary
    print("#" * 50)
    print("Before LoRA:")
    sd.print_trainable_summary(list_names=False)
    print("#" * 50)
    # add style token
    print("Adding style token:", style_token)
    sd.add_style_token(style_token)
    print("#" * 50)
    # Freeze pipeline
    print("Freezing pipeline:")
    sd.freeze_pipeline()  # freeze all params
    sd.print_trainable_summary(list_names=False)      
    print("#" * 50)
    # Add LoRA adapters
    #experiment_path = ROOT / "lora_out/best/"
    experiment_path = ROOT / "lora_out"
    lora_cfg_path = experiment_path / "lora_config.yaml"
    weight_name = "pytorch_lora_weights.safetensors"
    lora_weight_path = os.path.join(experiment_path, weight_name)
    sd.attach_lora(lora_cfg_path)
    print("#" * 50)
    print("After LoRA injection:")
    sd.print_trainable_summary(list_names=False)
    sd.load_lora_weights(lora_weight_path)
    # Generate a sample image
    # prompts = [ "A close up of a face of a boy , <sks> style",
    #             "A close up of a boy in a house, in <sks> style",
    #             "A close up of a woman in a forest, in <sks> style",
    #             "A close up of a man, in <sks> style",
    #             "A busy market where people are walking, in <sks> style"]
    prompts = [ "A busy market, in <sks> style",
                "A busy market, in <sks> style",
                "A busy market, in <sks> style",
                "A busy market, in <sks> style",
                "A busy market, in <sks> style"]
    extra_prompts = [ 
    "A girl in a red jacket, <sks> style",
    "A forest, in <sks> style",
    "Two children, <sks> style",
    "a group of people sitting at a table, <sks> style",
    "A boy sitting at a table, <sks> style",
    "a man and a woman standing in a forst, <sks> style",
    ]
    prompts.extend(extra_prompts)
    output_dir = ROOT / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_and_save(output_dir, sd, prompts)
