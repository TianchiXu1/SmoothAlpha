"""Cache frozen Depth-Anything-v2-Small geometry from degraded benchmark RGB."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import DOMAINS, image_path, load_split

MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["train", "val", "test"])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    device = torch.device(args.device)
    processor = AutoImageProcessor.from_pretrained(MODEL)
    estimator = AutoModelForDepthEstimation.from_pretrained(MODEL).to(device).eval().requires_grad_(False)
    for domain in args.domains:
        root = args.data_root / domain
        destination = root / "depth"
        destination.mkdir(parents=True, exist_ok=True)
        metadata_path = root / "depth_metadata.json"
        record = {"model": MODEL, "image_source": "raw", "normalization": "per_image_minmax", "files": []}
        if metadata_path.is_file():
            record = json.loads(metadata_path.read_text())
            if any(record.get(key) != value for key, value in (
                ("model", MODEL), ("image_source", "raw"), ("normalization", "per_image_minmax"))):
                raise ValueError(f"Existing cache uses a different geometry protocol: {metadata_path}")
        names = load_split(domain)
        for name in dict.fromkeys(name for split in args.splits for name in names[split]):
            path = destination / (Path(name).stem + ".npy")
            if path.is_file() and name in record["files"]:
                continue
            if path.exists():
                raise FileExistsError(f"Unrecorded depth cache would be overwritten: {path}. Use a separate data directory.")
            with Image.open(image_path(root / "raw", name)) as image:
                inputs = processor(images=image.convert("RGB"), return_tensors="pt").to(device)
            with torch.inference_mode():
                depth = estimator(**inputs).predicted_depth[0].cpu().numpy().astype(np.float32)
            if depth.ndim != 2 or not np.isfinite(depth).all():
                raise ValueError(f"Invalid estimator output for {domain}/{name}")
            depth = (depth - depth.min()) / max(float(depth.max() - depth.min()), 1e-6)
            np.save(path, depth, allow_pickle=False)
            record["files"].append(name)
            temporary = metadata_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(record, indent=2) + "\n")
            temporary.replace(metadata_path)
        print(f"{domain}: {len(record['files'])} degraded-input depth maps", flush=True)


if __name__ == "__main__":
    main()
