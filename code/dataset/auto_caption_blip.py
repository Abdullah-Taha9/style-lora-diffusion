import argparse
from pathlib import Path
import re
import sys
import json
from typing import List, Optional

import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration


def list_images(root_dirs: List[Path], exts=(".png", ".jpg", ".jpeg", ".webp", ".bmp")) -> List[Path]:
    out = []
    for d in root_dirs:
        if not d.exists():
            print(f"[WARN] missing dir: {d}", file=sys.stderr)
            continue
        for p in d.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                out.append(p)
    return sorted(out)


def simple_hints_from_name(path: Path) -> List[str]:
    """Very lightweight hints from filename / parent folder."""
    name = (path.stem + " " + path.parent.name).lower()
    hints = []
    # scenery
    if re.search(r"\b(forest|woods|tree|grove)\b", name): hints.append("forest")
    if re.search(r"\b(market|bazaar|shop|store|vendor)\b", name): hints.append("market")
    if re.search(r"\b(city|street|town|alley)\b", name): hints.append("cityscape")
    if re.search(r"\b(mountain|peak|alps|hill|valley)\b", name): hints.append("mountain")
    if re.search(r"\b(sea|ocean|beach|coast|shore)\b", name): hints.append("seaside")
    if re.search(r"\b(night|dusk|midnight)\b", name): hints.append("night")
    if re.search(r"\b(sunset|sunrise|dawn|dusk)\b", name): hints.append("sunset")
    if re.search(r"\b(rain|rainy|storm|stormy)\b", name): hints.append("rainy")
    if re.search(r"\b(snow|snowy|winter)\b", name): hints.append("snowy")
    # keep it short
    # dedupe while preserving order
    seen = set()
    out = []
    for h in hints:
        if h not in seen:
            out.append(h); seen.add(h)
    return out[:2]


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description="Auto-caption images with BLIP and write .txt sidecars.")
    ap.add_argument("dirs", nargs="+", type=str, help="One or more image directories (searched recursively).")
    ap.add_argument("--style-token", type=str, default="<sks>", help="Style token to include in every caption.")
    ap.add_argument("--prefix", type=str, default="", help="Optional text prefix (e.g., 'illustration,').")
    ap.add_argument("--suffix", type=str, default="style", help="Suffix word appended near the token (e.g., 'style'). Use '' to disable.")
    ap.add_argument("--batch", type=int, default=8, help="Batch size for BLIP.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing .txt sidecars.")
    ap.add_argument("--max-new-tokens", type=int, default=20, help="Max new tokens for BLIP generation.")
    ap.add_argument("--min-length", type=int, default=0, help="Min generated length (0 = no constraint).")
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"], help="Force device.")
    ap.add_argument("--model", type=str, default="Salesforce/blip-image-captioning-large", help="BLIP captioning model id.")
    ap.add_argument("--save-json", action="store_true", help="Also write a captions.json manifest per input dir.")
    args = ap.parse_args()

    roots = [Path(d).resolve() for d in args.dirs]
    images = list_images(roots)
    if not images:
        print("No images found.", file=sys.stderr)
        sys.exit(1)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Loading {args.model} on {device} ...")
    processor = BlipProcessor.from_pretrained(args.model)
    model = BlipForConditionalGeneration.from_pretrained(args.model).to(device).eval()

    def build_caption(base: str, hints: List[str]) -> str:
        parts = []
        if args.prefix.strip():
            parts.append(args.prefix.strip())
        base = base.strip().rstrip(".")
        if base:
            parts.append(base)
        # add up to two hints
        if hints:
            parts.append(", ".join(hints))
        # style token phrasing
        token_chunk = args.style_token.strip()
        if args.suffix.strip():
            token_chunk = f"{token_chunk} {args.suffix.strip()}"
        parts.append(token_chunk)
        # collapse spaces/commas
        text = ", ".join([p for p in parts if p])
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s+,", ",", text)
        return text.strip()

    # batching
    manifest = []
    pbar = tqdm(range(0, len(images), args.batch), desc="Captioning")
    for i in pbar:
        batch_paths = images[i:i+args.batch]
        # Skip those that already have sidecars unless overwrite
        todo = []
        for p in batch_paths:
            sidecar = p.with_suffix(p.suffix + ".txt")
            if sidecar.exists() and not args.overwrite:
                continue
            todo.append(p)
        if not todo:
            continue

        # load images
        pil_batch = []
        hints_batch = []
        for p in todo:
            try:
                im = Image.open(p).convert("RGB")
            except Exception as e:
                print(f"[WARN] failed to open {p}: {e}", file=sys.stderr)
                continue
            pil_batch.append(im)
            hints_batch.append(simple_hints_from_name(p))

        if not pil_batch:
            continue

        inputs = processor(images=pil_batch, return_tensors="pt").to(device)
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            min_length=args.min_length,
            num_beams=3,
            do_sample=False,
            repetition_penalty=1.1,
        )
        raw_caps = processor.batch_decode(generated_ids, skip_special_tokens=True)

        # write sidecars
        for p, base_cap, hints in zip(todo, raw_caps, hints_batch):
            cap = build_caption(base_cap, hints)
            sidecar = p.with_suffix(p.suffix + ".txt")
            try:
                sidecar.write_text(cap + "\n", encoding="utf-8")
            except Exception as e:
                print(f"[WARN] failed to write {sidecar}: {e}", file=sys.stderr)
                continue
            manifest.append({"image": str(p), "caption": cap})

        pbar.set_postfix(done=len(manifest))

    # save per-root manifest if requested
    if args.save_json:
        grouped = {}
        for row in manifest:
            # find which root contains it
            which = None
            for r in roots:
                try:
                    row_path = Path(row["image"]).resolve()
                    if r in row_path.parents or r == row_path:
                        which = r
                        break
                except Exception:
                    pass
            key = str(which) if which else "misc"
            grouped.setdefault(key, []).append(row)
        for root_str, rows in grouped.items():
            outp = Path(root_str) / "captions.json"
            try:
                outp.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception as e:
                print(f"[WARN] failed to write {outp}: {e}", file=sys.stderr)

    print("Done. Sidecars written for", len(manifest), "images.")


if __name__ == "__main__":
    main()

# Example usage:
# python auto_caption_blip.py data/512 data/trainB_ghibli --style-token "<sks>"
