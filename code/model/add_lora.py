import yaml
import json
from pathlib import Path
from typing import List, Dict, Any

import torch
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model, TaskType


class AddLoRA:
    def __init__(self, pipeline: StableDiffusionPipeline, config_path: str):
        self.pipe = pipeline
        self.cfg = self.load_config(config_path)

    def load_config(self, path: str | Path) -> dict:
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            if path.suffix.lower() in (".yaml", ".yml"):
                return yaml.safe_load(f)
            elif path.suffix.lower() == ".json":
                return json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {path.suffix}")
            
    def apply(self):
        # read self.cfg["targets"]["unet"] etc.
        # insert LoRA processors accordingly
        # freeze non-LoRA weights, unfreeze adapters
        
        t = self.cfg["targets"]
        if t.get("unet", {}).get("enabled", False):
            self._add_lora_to_unet(t["unet"])

        if t.get("text_encoder", {}).get("enabled", False):
            self._add_lora_to_text_encoder(t["text_encoder"])   

    def _add_lora_to_unet(self, unet_cfg: dict):
        """
        Attach LoRA to UNet via PEFT while handling parameter compatibility.
        """
        if not unet_cfg.get("enabled", False):
            return

        rank   = int(unet_cfg["rank"])
        alpha  = int(unet_cfg.get("alpha", rank))
        drop   = float(unet_cfg.get("dropout", 0.0))
        projs  = unet_cfg.get("projections", ["to_q","to_k","to_v","to_out"])

        # UNet attention: to_out is Sequential → first Linear is 'to_out.0'
        target_modules = []
        for p in projs:
            if p == "to_out":
                target_modules.append("to_out.0")
            else:
                target_modules.append(p)

        peft_cfg = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=drop,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
            target_modules=target_modules,
        )
        
        # Wrap the UNet with LoRA
        unet_peft = get_peft_model(self.pipe.unet, peft_cfg)
        
        # Store original methods
        original_forward = unet_peft.forward
        original_base_forward = unet_peft.base_model.forward
        
        # Get the signature of the original UNet to know what parameters it accepts
        import inspect
        original_sig = inspect.signature(self.pipe.unet.forward)
        valid_params = set(original_sig.parameters.keys())
        print(f"UNet valid parameters: {valid_params}")
        
        def patched_forward(sample, timestep, encoder_hidden_states, **kwargs):
            """Patch for the PEFT model's forward method"""
            # Filter kwargs to only include valid parameters
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
            return original_forward(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                **filtered_kwargs
            )
        
        def patched_base_forward(sample, timestep, encoder_hidden_states, **kwargs):
            """Patch for the base model's forward method"""
            # peft_model sends bad arguments through kwargs to the base forward
            # kwargs =
            # {'added_cond_kwargs': None, 'attention_mask': None,
            # 'cross_attention_kwargs': None, 'input_ids': None,
            # 'inputs_embeds': None, 'output_attentions': None,
            # 'output_hidden_states': None, 'return_dict': False, 'timestep_cond': None}

            # valid_params =
            # {'added_cond_kwargs', 'attention_mask', 'class_labels', 'cross_attention_kwargs',
            # 'down_block_additional_residuals', 'down_intrablock_additional_residuals', 'encoder_attention_mask',
            # 'encoder_hidden_states', 'mid_block_additional_residual', 'return_dict', 'sample', 'timestep', 'timestep_cond'}

            # Filter kwargs to only include valid parameters
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
            return original_base_forward(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                **filtered_kwargs
            )
        
        # Apply both patches
        unet_peft.forward = patched_forward
        unet_peft.base_model.forward = patched_base_forward
        
        self.pipe.unet = unet_peft
        for p in self.pipe.unet.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        print("UNet LoRA applied successfully!")
        print(f"Trainable parameters: {unet_peft.print_trainable_parameters()}")
        
    def _add_lora_to_text_encoder(self, text_encoder_cfg: dict):
        """
        Attach LoRA to CLIPTextModel via PEFT using dynamic parameter discovery.
        """
        if not text_encoder_cfg.get("enabled", False):
            return
        
        modules = text_encoder_cfg["modules"]
        rank = text_encoder_cfg["rank"]
        alpha = text_encoder_cfg["alpha"]
        dropout = text_encoder_cfg.get("dropout", 0.0)
        
        # Configure LoRA for Text Encoder
        peft_cfg = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
            target_modules=modules,
        )
        
        # Wrap the text encoder with LoRA
        text_encoder_peft = get_peft_model(self.pipe.text_encoder, peft_cfg)

        # Store original methods
        original_peft_forward = text_encoder_peft.forward
        original_base_forward = text_encoder_peft.base_model.forward
        
        # Get the signature of the original text encoder to know what parameters it accepts
        import inspect
        original_sig = inspect.signature(self.pipe.text_encoder.forward)
        valid_params = set(original_sig.parameters.keys())
        print(f"Text encoder valid parameters: {valid_params}")
        
        def patched_peft_forward(input_ids=None, attention_mask=None, **kwargs):
            """Patch for the PEFT model's forward method"""
            # Filter kwargs to only include valid parameters
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
            return original_peft_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **filtered_kwargs
            )
        
        def patched_base_forward(input_ids=None, attention_mask=None, **kwargs):
            """Patch for the base model's forward method"""
            # peft_model sends bad arguments through kwargs to the base forward
            # kwargs = {'inputs_embeds': None, 'output_attentions': None, 'output_hidden_states': None, 'return_dict': None}
            # valid_params = {'attention_mask', 'input_ids', 'output_attentions', 'output_hidden_states', 'position_ids'}
            # Filter kwargs to only include valid parameters
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
            return original_base_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **filtered_kwargs
            )
        
        # Apply both patches
        text_encoder_peft.forward = patched_peft_forward
        text_encoder_peft.base_model.forward = patched_base_forward
        
        # Replace the text encoder in the pipeline
        self.pipe.text_encoder = text_encoder_peft
        for p in self.pipe.text_encoder.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        # Verify LoRA is actually applied
        print("Text encoder LoRA applied successfully!")
        print(f"Trainable parameters: {text_encoder_peft.print_trainable_parameters()}")


if __name__ == "__main__":
    cfg_path = "/home/hpc/v123be/v123be29/repos/stable_diffusion_lora_fine_tuning/lora_config.yaml"
    model_name = "runwayml/stable-diffusion-v1-5"
    from diffusers_model import StableDiffusion
    sd = StableDiffusion(model_name, "cuda" if torch.cuda.is_available() else "cpu")
    print("#" * 50)
    print("Before LoRA:")
    sd.print_trainable_summary(list_names=False)
    sd.freeze_pipeline()  # freeze all params
    print("#" * 50)
    print("After freezing:")
    sd.print_trainable_summary(list_names=False)
    AddLoRA(sd.pipe, cfg_path).apply()
    print("#" * 50)
    print("After LoRA injection:")
    sd.print_trainable_summary(list_names=False)

