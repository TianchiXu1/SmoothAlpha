"""Restore an RGB image directory with a trained SmoothAlpha checkpoint."""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from data.dataset import read_depth, read_rgb
from models import SmoothAlpha

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def input_images(directory):
    """Find supported images and reject output-name collisions before writing."""
    if not directory.is_dir():
        raise NotADirectoryError(f"Input directory not found: {directory}")
    paths = sorted(path for path in directory.iterdir()
                   if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise ValueError(f"No supported RGB images in {directory}")
    stems = [path.stem.casefold() for path in paths]
    if len(stems) != len(set(stems)):
        raise ValueError("Input images must have distinct filename stems")
    return paths


@torch.inference_mode()
def restore_directory(checkpoint, input_dir, output_dir, device, depth_dir=None, save_fields=False):
    """Restore at the paper's 256 x 256 resolution, without clean references."""
    paths = input_images(input_dir)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Choose an empty output directory: {output_dir}")
    if depth_dir is not None:
        for path in paths:
            depth_path = depth_dir / (path.stem + ".npy")
            if not depth_path.is_file():
                raise FileNotFoundError(f"Missing cached depth: {depth_path}")

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = SmoothAlpha()
    model.load_state_dict(state.get("model", state), strict=True)
    model.to(device).eval()
    processor, estimator = None, None
    if depth_dir is None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL)
        estimator = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL)
        estimator.to(device).eval().requires_grad_(False)

    output_dir.mkdir(parents=True, exist_ok=True)
    field_dir = output_dir / "fields"
    if save_fields:
        field_dir.mkdir()
    for path in paths:
        if depth_dir is not None:
            depth = read_depth(depth_dir / (path.stem + ".npy"))[None].to(device)
        else:
            with Image.open(path) as image:
                inputs = processor(images=image.convert("RGB"), return_tensors="pt").to(device)
            depth = estimator(**inputs).predicted_depth[:, None].float()
            if not torch.isfinite(depth).all():
                raise FloatingPointError(f"Non-finite estimated depth: {path.name}")
            depth = (depth - depth.amin()) / (depth.amax() - depth.amin()).clamp(min=1e-6)
            depth = F.interpolate(depth, size=(256, 256), mode="bilinear", align_corners=False)
        output = model(read_rgb(path)[None].to(device), depth)
        if not all(torch.isfinite(output[key]).all() for key in ("x_pred", "alpha", "T", "veil")):
            raise FloatingPointError(f"Non-finite model output: {path.name}")
        prediction = output["x_pred"][0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
        Image.fromarray((prediction * 255).astype(np.uint8)).save(output_dir / (path.stem + ".png"))
        if save_fields:
            np.savez_compressed(field_dir / (path.stem + ".npz"), **{
                key: output[key][0].cpu().numpy().transpose(1, 2, 0)
                for key in ("alpha", "T", "veil")
            })
        print(f"Restored {path.name}", flush=True)
    return len(paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--depth_dir", type=Path, help="Optional matching degraded-input .npy depth cache")
    parser.add_argument("--save_fields", action="store_true", help="Save raw alpha, T and veil arrays")
    args = parser.parse_args()
    count = restore_directory(args.checkpoint, args.input_dir, args.output_dir,
                              torch.device(args.device), args.depth_dir, args.save_fields)
    print(f"Saved {count} restored 256 x 256 RGB images to {args.output_dir}")


if __name__ == "__main__":
    main()
