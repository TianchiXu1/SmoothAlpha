"""Load the fixed paper partitions and degraded-input monocular geometry."""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

DOMAINS = ("water", "fog", "smoke")
SPLITS = {
    "water": "uieb_split.json",
    "fog": "dtu_fog_split.json",
    "smoke": "mitsuba_smoke_split.json",
}


def load_split(domain):
    """Read the released train/validation/test filename lists without reshuffling."""
    if domain not in DOMAINS:
        raise ValueError(f"Unknown medium: {domain}")
    path = Path(__file__).parent / "splits" / SPLITS[domain]
    partitions = json.loads(path.read_text())
    seen = set()
    for partition in ("train", "val", "test"):
        names = partitions[partition]
        if not names or len(names) != len(set(names)) or seen.intersection(names):
            raise ValueError(f"Invalid or overlapping {domain}/{partition} split")
        if any(Path(name).name != name for name in names):
            raise ValueError("Split entries must be filenames")
        seen.update(names)
    return partitions


def image_path(directory, filename):
    """Resolve the paired image by stem, allowing the source benchmark extensions."""
    direct = directory / filename
    if direct.is_file():
        return direct
    for suffix in (".jpg", ".png", ".JPG", ".PNG", ".jpeg", ".JPEG"):
        candidate = directory / (Path(filename).stem + suffix)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing paired image: {direct}")


def read_rgb(path, size=256):
    """Convert RGB to float32 [0,1] with the paper implementation's PIL resize."""
    with Image.open(path) as image:
        image = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        return pil_to_tensor(image).float().div(255)


def read_depth(path, size=256):
    """Normalize each cached monocular cue before bilinear resampling."""
    value = np.load(path, allow_pickle=False).astype(np.float32)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError(f"Depth must be a finite 2D array: {path}")
    value = (value - value.min()) / max(float(value.max() - value.min()), 1e-6)
    tensor = torch.from_numpy(value)[None, None]
    return F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)[0]


class PairedDataset(Dataset):
    """Paired RGB inputs and cached depth extracted exclusively from raw images."""

    def __init__(self, data_root, domain, split="test", size=256):
        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid partition: {split}")
        self.root = Path(data_root) / domain
        self.domain = domain
        self.size = size
        self.names = load_split(domain)[split]
        metadata_path = self.root / "depth_metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Missing {metadata_path}. Extract input depth with "
                "scripts/extract_depth_anything_v2.py before training/evaluation."
            )
        metadata = json.loads(metadata_path.read_text())
        if (metadata.get("image_source") != "raw"
                or metadata.get("normalization") != "per_image_minmax"
                or metadata.get("model") != "depth-anything/Depth-Anything-V2-Small-hf"):
            raise ValueError(f"Depth metadata does not match the released configuration: {metadata_path}")
        recorded = set(metadata.get("files", []))
        for name in self.names:
            image_path(self.root / "raw", name)
            image_path(self.root / "reference", name)
            depth_path = self.root / "depth" / (Path(name).stem + ".npy")
            if name not in recorded or not depth_path.is_file():
                raise FileNotFoundError(f"Missing input-depth record or array: {depth_path}")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        return {
            "obs": read_rgb(image_path(self.root / "raw", name), self.size),
            "gt": read_rgb(image_path(self.root / "reference", name), self.size),
            "depth": read_depth(self.root / "depth" / (Path(name).stem + ".npy"), self.size),
            "domain_id": DOMAINS.index(self.domain),
            "fname": name,
        }
