"""Offline checks for the model, fixed partitions, sample exposure and metrics."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import DOMAINS, load_split, read_depth
from data.sampling import BalancedExposureSampler
from metrics import paired_metrics, structural_metrics
from models import SmoothAlpha
from models.latent import EMATargetEncoder, LatentPredictor, latent_prediction_loss
from models.losses import perceptual_weight
from infer import input_images, restore_directory
from PIL import Image


class InstallationChecks(unittest.TestCase):
    def test_fixed_partitions_and_exposure(self):
        counts = {"water": (712, 89, 89), "fog": (467, 64, 128), "smoke": (200, 50, 50)}
        for domain, sizes in counts.items():
            partitions = load_split(domain)
            self.assertEqual(tuple(len(partitions[key]) for key in ("train", "val", "test")), sizes)
        class Domain:
            def __init__(self, name):
                self.domain = name
            def __len__(self):
                return counts[self.domain][0]
        datasets = [Domain(domain) for domain in DOMAINS]
        sampler = BalancedExposureSampler(datasets)
        first = list(sampler)
        self.assertEqual(first, list(sampler))
        self.assertEqual([sum(lo <= i < hi for i in first) for lo, hi in ((0, 712), (712, 1179), (1179, 1379))],
                         [712, 712, 712])
        sampler.set_epoch(2)
        self.assertNotEqual(first, list(sampler))
        config = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/smooth_alpha_joint.yaml").read_text())
        self.assertEqual(config["epochs"] * config["epoch_samples"], 106800)
        self.assertEqual(perceptual_weight(19, config["loss"]), 0)
        self.assertAlmostEqual(perceptual_weight(20, config["loss"]), 0.1)
        self.assertAlmostEqual(perceptual_weight(24, config["loss"]), 0.5)

    def test_model_and_clean_latent_target(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        model = SmoothAlpha().eval()
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 7905624)
        observed, depth = torch.rand(1, 3, 32, 32), torch.rand(1, 1, 32, 32)
        with torch.no_grad():
            output = model(observed, depth)
        self.assertEqual(output["x_pred"].shape, observed.shape)
        self.assertEqual(output["delta_lowres"].shape, (1, 3, 4, 4))
        self.assertTrue(torch.isfinite(output["x_pred"]).all())
        self.assertTrue((output["alpha"] > 0).all())
        self.assertTrue(((output["T"] > 0) & (output["T"] <= 1)).all())
        torch.testing.assert_close(output["x_pred"],
                                   output["proposal"] * output["T"] + output["veil"] * (1 - output["T"]),
                                   rtol=0, atol=0)
        target = EMATargetEncoder(model.detail_path)
        predictor = LatentPredictor()
        clean = torch.rand_like(observed)
        clean_target = target.encode(clean, depth)
        self.assertFalse(clean_target.requires_grad)
        latent_prediction = predictor(output["z"].detach())
        loss = latent_prediction_loss(latent_prediction, clean_target)
        loss.backward()
        self.assertTrue(torch.isfinite(predictor.conv1.weight.grad).all())
        self.assertTrue(all(parameter.grad is None for parameter in target.parameters()))

    def test_depth_and_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "depth.npy"
            np.save(path, np.ones((5, 7), dtype=np.float32))
            depth = read_depth(path, 32)
            self.assertEqual(depth.shape, (1, 32, 32))
            self.assertEqual(float(depth.abs().max()), 0.0)
            np.save(path, np.full((5, 7), np.nan, dtype=np.float32))
            with self.assertRaises(ValueError):
                read_depth(path)
        image = np.full((32, 32, 3), 0.5, dtype=np.float32)
        score = paired_metrics(image, image)
        self.assertAlmostEqual(score["psnr"], 100.0)
        self.assertAlmostEqual(score["ssim"], 1.0)
        self.assertEqual(structural_metrics(np.zeros_like(image, dtype=np.uint8),
                                            np.zeros_like(image, dtype=np.uint8)),
                         {"sift_keypoints": 0, "sift_matches": 0})

    def test_image_directory_inference(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs, depths, output = root / "input", root / "depth", root / "output"
            inputs.mkdir()
            depths.mkdir()
            image = np.zeros((40, 56, 3), dtype=np.uint8)
            image[:, :, 1] = np.arange(56, dtype=np.uint8)[None] * 4
            Image.fromarray(image).save(inputs / "scene.png")
            np.save(depths / "scene.npy", np.arange(35, dtype=np.float32).reshape(5, 7))
            checkpoint = root / "model.pth"
            torch.save({"model": SmoothAlpha().state_dict()}, checkpoint)
            self.assertEqual(restore_directory(checkpoint, inputs, output, torch.device("cpu"),
                                               depths, save_fields=True), 1)
            with Image.open(output / "scene.png") as restored:
                self.assertEqual(restored.size, (256, 256))
                self.assertEqual(restored.mode, "RGB")
            with np.load(output / "fields/scene.npz") as fields:
                self.assertEqual(set(fields.files), {"alpha", "T", "veil"})
                self.assertTrue(all(fields[key].shape == (256, 256, 3) for key in fields.files))
                self.assertTrue(all(np.isfinite(fields[key]).all() for key in fields.files))
            with self.assertRaises(FileExistsError):
                restore_directory(checkpoint, inputs, output, torch.device("cpu"), depths)
            Image.fromarray(image).save(inputs / "scene.jpg")
            with self.assertRaises(ValueError):
                input_images(inputs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
