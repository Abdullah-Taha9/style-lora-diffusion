from __future__ import annotations
import os
from typing import List, Dict, Any, Optional, Tuple
import json
import yaml

import torch
from safetensors.torch import save_file, load_file
from diffusers import StableDiffusionPipeline
from peft import PeftModel

from .add_special_token import AddSpecialToken
from .add_lora import AddLoRA


class StableDiffusion:
    """
    Wrapper around StableDiffusionPipeline with:
      - component discovery
      - per-component param counts (total/trainable) for nn.Module components
      - full report export (JSON/YAML)
      - image generation
    """
    def __init__(self, model_id: str, device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        dtype = torch.float16 if (device.startswith("cuda") and torch.cuda.is_available()) else torch.float32
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)
        self._last_lora_cfg_path = None

    # ---------- discovery ----------
    def all_component_names(self) -> List[str]:
        """All keys exposed by pipe.components (modules and non-modules)."""
        return list(self.pipe.components.keys())

    def module_component_names(self) -> List[str]:
        """Only those components that are torch.nn.Module."""
        return [n for n, o in self.pipe.components.items() if isinstance(o, torch.nn.Module)]

    def component(self, name: str) -> Any:
        """Raw object (nn.Module or other) for a given component name."""
        return self.pipe.components.get(name, None)

    # ---------- counting ----------
    @staticmethod
    def _count_module_params(mod: torch.nn.Module) -> Dict[str, Any]:
        tot = trn = 0
        for p in mod.parameters():
            n = p.numel()
            tot += n
            if p.requires_grad:
                trn += n
        pct = 0.0 if tot == 0 else (100.0 * trn / tot)
        return {"total": tot, "trainable": trn, "pct": pct}

    def param_counts_by_module(self) -> Dict[str, Dict[str, Any]]:
        """Counts for each nn.Module component."""
        out: Dict[str, Dict[str, Any]] = {}
        for name in self.module_component_names():
            mod = self.component(name)
            out[name] = self._count_module_params(mod)
        return out

    # ---------- human-readable summary ----------
    def print_trainable_summary(self, list_names: bool = False, max_names: int = 20) -> None:
        mod_counts = self.param_counts_by_module()
        overall_total = sum(c["total"] for c in mod_counts.values())
        overall_train = sum(c["trainable"] for c in mod_counts.values())
        overall_pct = 0.0 if overall_total == 0 else 100.0 * overall_train / overall_total

        print("== nn.Module components ==")
        for name in self.module_component_names():
            c = mod_counts[name]
            print(f"[{name}] total: {c['total']:,} | trainable: {c['trainable']:,} ({c['pct']:.4f}%)")
            if list_names and c["trainable"] > 0:
                mod = self.component(name)
                shown = 0
                for pn, p in mod.named_parameters():
                    if p.requires_grad:
                        print(f"   • {pn} {tuple(p.shape)}")
                        shown += 1
                        if shown >= max_names:
                            remaining = sum(1 for _pn, _p in mod.named_parameters() if _p.requires_grad) - shown
                            if remaining > 0:
                                print(f"   … +{remaining} more")
                            break

        print("\n== non-module components ==")
        nonmods = [n for n in self.all_component_names() if n not in self.module_component_names()]
        if nonmods:
            for n in nonmods:
                obj = self.component(n)
                print(f"[{n}] type: {type(obj).__name__}")
        else:
            print("(none)")

        print(f"\n[overall] total: {overall_total:,} | trainable: {overall_train:,} ({overall_pct:.4f}%)")

    # ---------- structured report ----------
    def generate_report(self, list_names: bool = False, max_names: int = 50) -> Dict[str, Any]:
        """
        Returns a dict with:
          - model_id, device
          - all_components (names)
          - module_components: {name: {type, counts, (optional) trainable_params}}
          - non_module_components: [{name, type}]
          - overall (totals across nn.Module components)
        """
        all_names = self.all_component_names()
        mod_names = self.module_component_names()
        nonmod_names = [n for n in all_names if n not in mod_names]

        module_components: Dict[str, Any] = {}
        overall_total = 0
        overall_train = 0

        for name in mod_names:
            mod = self.component(name)
            counts = self._count_module_params(mod)
            overall_total += counts["total"]
            overall_train += counts["trainable"]

            comp_entry: Dict[str, Any] = {
                "type": type(mod).__name__,
                "counts": counts
            }

            if list_names and counts["trainable"] > 0:
                listed = 0
                names_list = []
                for pn, p in mod.named_parameters():
                    if p.requires_grad:
                        names_list.append({"name": pn, "shape": list(p.shape)})
                        listed += 1
                        if listed >= max_names:
                            names_list.append({"more": "truncated"})
                            break
                comp_entry["trainable_params"] = names_list

            module_components[name] = comp_entry

        overall_pct = 0.0 if overall_total == 0 else 100.0 * overall_train / overall_total

        non_module_components = [
            {"name": n, "type": type(self.component(n)).__name__}
            for n in nonmod_names
        ]

        report: Dict[str, Any] = {
            "model_id": self.model_id,
            "device": self.device,
            "all_components": all_names,                # every component name in the pipeline
            "module_components": module_components,     # only nn.Module components with counts
            "non_module_components": non_module_components,
            "overall": {
                "total": overall_total,
                "trainable": overall_train,
                "pct": overall_pct
            }
        }
        return report

    def save_report(self, path: str, fmt: str = "json", list_names: bool = False, max_names: int = 50) -> None:
        data = self.generate_report(list_names=list_names, max_names=max_names)
        fmt = fmt.lower()
        if fmt == "json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        elif fmt in ("yaml", "yml"):
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, sort_keys=False)
        else:
            raise ValueError("fmt must be 'json' or 'yaml'")

    def add_style_token(self, token: str, **kw) -> int:
        return AddSpecialToken(self.pipe, token).apply(**kw)

    def attach_lora(self, lora_cfg_path: str):
         return AddLoRA(self.pipe, lora_cfg_path).apply()  # you’ll write AddLoRA next

    def freeze_pipeline(self):
        """Set requires_grad=False for all params in the pipeline."""
        modules = [getattr(self.pipe, n, None) for n in self.module_component_names()]
        for m in modules:
            if m is None: 
                continue
            for p in m.parameters():
                p.requires_grad = False

    def has_unet_lora(self) -> bool:
        try:
            return isinstance(self.pipe.unet, PeftModel)
        except Exception:
            return False

    def has_text_encoder_lora(self) -> bool:
        try:
            return isinstance(self.pipe.text_encoder, PeftModel)
        except Exception:
            return False

    def assert_lora_ready(self, need_unet=True, need_te=True):
        ok_unet = (not need_unet) or self.has_unet_lora()
        ok_te   = (not need_te)   or self.has_text_encoder_lora()
        if not (ok_unet and ok_te):
            missing = []
            if need_unet and not ok_unet: missing.append("UNet LoRA")
            if need_te   and not ok_te:   missing.append("Text Encoder LoRA")
            raise RuntimeError(f"LoRA not attached: missing {', '.join(missing)}. "
                            "Attach via `attach_lora(...)` (same cfg as training) before saving/loading.")

    # ---- Style token presence ----
    def has_style_token(self, token: str) -> bool:
        return token in self.pipe.tokenizer.get_vocab()

    def style_token_id(self, token: str) -> int | None:
        if not self.has_style_token(self.pipe, token): return None
        return self.pipe.tokenizer.convert_tokens_to_ids(token)


    # ---------- LoRA weights saving and loading----------
    def save_lora_weights(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)  # <-- creates folder(s) if missing

        self.assert_lora_ready(need_unet=True, need_te=True)  # guard
        state = {}

        # UNet (PEFT)
        try:
            if isinstance(self.pipe.unet, PeftModel):

                for k, v in self.pipe.unet.state_dict().items():
                    if "lora" in k.lower():
                        state[f"unet.{k}"] = v.cpu()
        except Exception:
            pass
        # Text Encoder (PEFT)
        try:
            if isinstance(self.pipe.text_encoder, PeftModel):
                for k, v in self.pipe.text_encoder.state_dict().items():
                    if "lora" in k.lower():
                        state[f"text_encoder.{k}"] = v.cpu()
        except Exception:
            pass

        if not state:
            raise RuntimeError("No LoRA weights found to save.")

        # optional: store minimal metadata inside the file
        metadata = {
            "format": "sd15-lora",
            "unet": str(self.has_unet_lora()).lower(),
            "text_encoder": str(self.has_text_encoder_lora()).lower(),
            "model_id": getattr(self, "model_id", ""),
        }
        save_file(state, path, metadata=metadata)
        print(f"Saved LoRA weights to {path}")

    
    def load_lora_weights(self, path: str, auto_inject_cfg: str | None = None):
        sd = load_file(path)

        # Ensure sockets exist; if not, optionally auto-inject structure
        need_unet = any(k.startswith("unet.") for k in sd.keys())
        need_te   = any(k.startswith("text_encoder.") for k in sd.keys())

        if (need_unet and not self.has_unet_lora()) or (need_te and not self.has_text_encoder_lora()):
            if auto_inject_cfg:
                # The same config used at training time
                self.attach_lora(auto_inject_cfg)
            else:
                self.assert_lora_ready(need_unet=need_unet, need_te=need_te)

        # Split and load
        unet_sd = {k.replace("unet.", "", 1): v for k, v in sd.items() if k.startswith("unet.")}
        te_sd   = {k.replace("text_encoder.", "", 1): v for k, v in sd.items() if k.startswith("text_encoder.")}

        if unet_sd:
            self.pipe.unet.load_state_dict(unet_sd, strict=False)
        if te_sd:
            self.pipe.text_encoder.load_state_dict(te_sd, strict=False)
        print(f"Loaded LoRA weights from {path}")
    
    # ---------- inference ----------
    def generate_image(self, prompts: str | List[str], **kwargs):
        if isinstance(prompts, str):
            prompts = [prompts]
        defaults = dict(num_inference_steps=30, guidance_scale=7.0, height=512, width=512)
        defaults.update(kwargs)
        return self.pipe(prompts, **defaults).images

    def attach_lora(self, lora_cfg_path: str):
        self._last_lora_cfg_path = str(lora_cfg_path)   # remember it for run snapshots
        return AddLoRA(self.pipe, lora_cfg_path).apply()

    def active_lora_configs(self) -> Dict[str, Dict[str, dict]]:
        """
        Returns { 'unet': {'default': {...}}, 'text_encoder': {'default': {...}} }
        where the inner dicts are PEFT LoraConfig.to_dict() outputs.
        Only includes components that actually have LoRA attached.
        """
        out: Dict[str, Dict[str, dict]] = {}
        for name in ("unet", "text_encoder"):
            m = getattr(self.pipe, name, None)
            cfgs = getattr(m, "peft_config", None)
            if isinstance(cfgs, dict) and cfgs:
                out[name] = {}
                for adapter_name, cfg in cfgs.items():
                    # LoraConfig has .to_dict()
                    to_d = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
                    out[name][adapter_name] = to_d
        return out

# ---------- demo ----------
if __name__ == "__main__":
    model_name = "runwayml/stable-diffusion-v1-5"
    style_token = "<sks>"
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent.parent.parent
    lora_cfg_path = ROOT / "configs" / "lora_config.yaml"

    sd = StableDiffusion(model_name, "cuda" if torch.cuda.is_available() else "cpu")
    # Print initial summary
    print("#" * 50)
    print("Before LoRA:")
    sd.print_trainable_summary(list_names=False)
    sd.save_report(os.path.join(ROOT,"reports","sd_report.json"),
                    fmt="json", list_names=True, max_names=100)
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
    sd.attach_lora(lora_cfg_path)
    #sd.save_lora_weights("lora_out/pytorch_lora_weights.safetensors")
    print("#" * 50)
    print("After LoRA injection:")
    sd.print_trainable_summary(list_names=False)
    sd.save_report(os.path.join(ROOT,"reports","lora_sd_report.json"),
                   fmt="json", list_names=True, max_names=100)
    sd.load_lora_weights(os.path.join(ROOT, "lora_out", "pytorch_lora_weights.safetensors"))
    # Generate a sample image
    imgs = sd.generate_image("a <sks> style image")
    imgs[0].save(os.path.join(ROOT, "output/output_image.png"))
    # print("Wrote: sd_report.json, output_image.png")
