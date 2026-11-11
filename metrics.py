"""Paired image metrics and the benchmark's UIQM/UCIQE definitions."""
import math

import cv2
import numpy as np
from skimage.metrics import structural_similarity


def _mu_alpha_trim(x, alpha=0.1):
    xs = np.sort(x.ravel())
    n = len(xs)
    lo, hi = int(alpha * n), int((1 - alpha) * n)
    return float(xs[lo:hi].mean()) if hi > lo else float(xs.mean())

def uicm(img):
    r, g, b = img[..., 0].astype(np.float64), img[..., 1].astype(np.float64), img[..., 2].astype(np.float64)
    rg, yb = r - g, (r + g) / 2.0 - b
    mu_rg, mu_yb = _mu_alpha_trim(rg), _mu_alpha_trim(yb)
    s_rg, s_yb = float(rg.var()), float(yb.var())
    return -0.0268 * math.sqrt(mu_rg ** 2 + mu_yb ** 2) + 0.1586 * math.sqrt(s_rg + s_yb)

def _eme(ch, ws=10):
    h, w = ch.shape
    bh, bw = h // ws, w // ws
    if bh == 0 or bw == 0:
        return 0.0
    acc, k = 0.0, 0
    for i in range(ws):
        for j in range(ws):
            blk = ch[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw]
            mx, mn = float(blk.max()), float(blk.min())
            if mn > 0 and mx > mn:
                acc += math.log(mx / mn)
                k += 1
    return 2.0 / max(k, 1) * acc

def uism(img):
    sob = []
    for c in range(3):
        ch = img[..., c].astype(np.float64)
        gx = cv2.Sobel(ch, cv2.CV_64F, 1, 0)
        gy = cv2.Sobel(ch, cv2.CV_64F, 0, 1)
        edge = np.hypot(gx, gy)
        sob.append(_eme(edge * (ch / 255.0), 10))
    return 0.299 * sob[0] + 0.587 * sob[1] + 0.114 * sob[2]

def uiconm(img, ws=10):
    x = img.astype(np.float64).mean(axis=2)
    h, w = x.shape
    bh, bw = h // ws, w // ws
    if bh == 0 or bw == 0:
        return 0.0
    acc, k = 0.0, 0
    for i in range(ws):
        for j in range(ws):
            blk = x[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw]
            mx, mn = float(blk.max()), float(blk.min())
            s = mx + mn
            d = mx - mn
            if s > 1e-9 and d > 1e-9:
                r = d / s
                acc += r * math.log(r)
                k += 1
    return abs(acc / max(k, 1))

def uiqm(img):
    return 0.0282 * uicm(img) + 0.2953 * uism(img) + 3.5753 * uiconm(img)

def uciqe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float64)
    l = lab[..., 0] * 100.0 / 255.0
    a = lab[..., 1] - 128.0
    b = lab[..., 2] - 128.0
    chroma = np.hypot(a, b)
    sigma_c = float(chroma.std()) / 100.0
    con_l = float(np.percentile(l, 99) - np.percentile(l, 1)) / 100.0
    sat = chroma / np.maximum(np.hypot(l, chroma), 1e-6)
    mu_s = float(sat.mean())
    return 0.4680 * sigma_c + 0.2745 * con_l + 0.2576 * mu_s


def paired_metrics(prediction, reference):
    """PSNR and channel-averaged skimage SSIM on unquantized float32 RGB."""
    mse = float(np.mean((prediction - reference) ** 2))
    psnr = 10.0 * np.log10(1.0 / max(mse, 1e-10))
    ssim = np.mean([structural_similarity(prediction[..., channel], reference[..., channel],
                                         data_range=1.0) for channel in range(3)])
    return {"psnr": float(psnr), "ssim": float(ssim)}


def structural_metrics(image, reference):
    """Report SIFT keypoint counts and ratio-tested reference matches separately."""
    sift = cv2.SIFT_create()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    target = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    _, target_descriptors = sift.detectAndCompute(target, None)
    good_matches = 0
    if descriptors is not None and target_descriptors is not None and len(target_descriptors) >= 2:
        pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(descriptors, target_descriptors, k=2)
        good_matches = sum(len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance for pair in pairs)
    return {"sift_keypoints": len(keypoints), "sift_matches": int(good_matches)}
