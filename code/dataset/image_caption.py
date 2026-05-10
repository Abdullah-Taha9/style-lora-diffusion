from pathlib import Path
from typing import Optional, Dict, List
from PIL import Image
import random

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ImageCaptionDataset(Dataset):
    """
    Minimal dataset for SD1.5 LoRA training.

    It loads images from a folder and provides one caption per image via:
      A) captions file with lines: "filename|your caption"
      B) sidecar files named "<image>.<ext>.txt" (same folder as image)
      C) fallback: one-caption-per-line file matched to images by sorted order
      D) default_caption string if nothing else is found

    Returns dict:
      {
        "pixel_values": FloatTensor (3, H, W) in [-1, 1],
        "text":         str caption
      }
    """

    def __init__(
        self,
        image_dirs: List[str] | List[Path],
        captions_path: Optional[str | Path] = None,
        resolution: int = 512,
        center_crop: bool = True,
        random_flip: bool = True,
        default_caption: Optional[str] = None,
        style_token: Optional[str] = None,
        caption_dropout: float = 0.0,    
        class_word: str = "illustration"
    ):
        self.image_dirs = image_dirs
        self.style_token = style_token
        self.caption_dropout = float(caption_dropout)
        self.class_word = class_word
        exts = {".png", ".jpg", ".jpeg"}
        self.images: List[Path] = []
        for image_dir in image_dirs:
            image_dir = Path(image_dir)
            if not image_dir.exists():
                raise FileNotFoundError(f"Image folder not found: {image_dir}")
            # collect images
            images: List[Path] = sorted(
                p for p in image_dir.iterdir()
                if p.is_file() and p.suffix.lower() in exts
            )
            self.images.extend(images)
            if not self.images:
                raise ValueError(f"No images found in {image_dir}")

        # captions map (filename -> caption)
        self.caption_map: Dict[str, str] = {}
        self.default_caption = default_caption

        if captions_path is not None:
            captions_path = Path(captions_path)
            if not captions_path.exists():
                raise FileNotFoundError(f"Captions file not found: {captions_path}")
            lines = [ln.strip() for ln in captions_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if any("|" in ln for ln in lines):
                # Mode A: "filename|caption"
                for ln in lines:
                    name, cap = ln.split("|", 1)
                    self.caption_map[name.strip()] = cap.strip()
            else:
                # Mode C: one caption per line -> match by sorted order
                if len(lines) != len(self.images):
                    print("[WARN] captions count != images; using sidecars/default for leftovers")
                for img, cap in zip(self.images, lines):
                    self.caption_map[img.name] = cap

        # transforms -> tensor in [-1, 1]
        t = [transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)]
        if center_crop:
            t.append(transforms.RandomResizedCrop(resolution, scale=(0.9, 1.0),
                                     interpolation=transforms.InterpolationMode.BICUBIC, antialias=True))
        if random_flip:
            t.append(transforms.RandomHorizontalFlip())
        t += [
            transforms.ToTensor(),                                         # [0,1]
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),        # [-1,1]
        ]
        self.transform = transforms.Compose(t)


    def _caption_for(self, img_path: Path) -> str:
        # existing priority
        cap = self.caption_map.get(img_path.name)
        if cap is None:
            sidecar = img_path.with_suffix(img_path.suffix + ".txt")
            cap = sidecar.read_text(encoding="utf-8").strip() if sidecar.exists() else None
        if cap is None:
            cap = self.default_caption or f"an {self.class_word}"

        # ensure style token appears
        if self.style_token and self.style_token not in cap:
            cap = f"{cap} {self.style_token}".strip()

        # caption dropout: sometimes shorten to emphasize the token
        if self.caption_dropout > 0 and random.random() < self.caption_dropout:
            cap = f"{self.style_token}".strip() if self.style_token else f"an {self.class_word}"

        return cap

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        pixel = self.transform(img).to(torch.float32)
        text = self._caption_for(img_path)
        return {"pixel_values": pixel, "text": text}


def collate_fn(batch: List[dict]) -> dict:
    """
    Stacks pixel tensors and keeps a list of raw texts.
    Tokenization happens in the training loop with the pipeline tokenizer.
    """
    px = torch.stack([b["pixel_values"] for b in batch], dim=0)
    texts = [b["text"] for b in batch]
    return {"pixel_values": px, "text": texts}


if __name__ == "__main__":
    import yaml
    import os
    from torch.utils.data import DataLoader
    # Load YAML
    ROOT = Path(__file__).resolve().parent.parent.parent

    with open(os.path.join(ROOT, "configs/train_config.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Build dataset from config
    ds = ImageCaptionDataset(
        image_dirs=[os.path.join(ROOT, cfg["data"]["datasets_dirs" ][i]) for i in range(len(cfg["data"]["datasets_dirs"]))],
        captions_path=cfg["data"]["captions_path"],
        resolution=cfg["data"]["resolution"],
        center_crop=cfg["data"].get("center_crop", True),
        random_flip=cfg["data"].get("random_flip", True),
        default_caption=cfg["data"].get("default_caption"),
        style_token=cfg["data"].get("style_token"),
        caption_dropout=cfg["data"].get("caption_dropout", 0.0),
        class_word=cfg["data"].get("class_word", "illustration")
    )

    # DataLoader
    loader = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    # test iteration
    for batch in loader:
        print("Batch pixel shape:", batch["pixel_values"].shape)
        print("Batch texts:", batch["text"])
        break