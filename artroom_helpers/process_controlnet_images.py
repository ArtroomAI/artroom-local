import os
import cv2
import numpy as np
import torch
from einops import rearrange

from artroom_helpers.annotator.hed import HEDdetector
from artroom_helpers.annotator.midas import MidasDetector
from artroom_helpers.annotator.openpose import OpenposeDetector
from artroom_helpers.annotator.normalbae import NormalBaeDetector
from artroom_helpers.annotator.pidinet import PidiNetDetector
from artroom_helpers.annotator.mlsd import MLSDdetector
from artroom_helpers.annotator.lineart import LineartDetector
from artroom_helpers.annotator.lineart_anime import LineartAnimeDetector
from artroom_helpers.annotator.shuffle import ContentShuffleDetector

global annotator_ckpts_path
annotator_ckpts_path = os.path.join(os.path.dirname(__file__), 'ckpts')

def apply_controlnet(image, controlnet_mode, models_dir):
    global annotator_ckpts_path
    annotator_ckpts_path = os.path.join(models_dir, 'ControlNet', 'annotators_(not_your_models)' 'ckpts')
    os.makedirs(annotator_ckpts_path, exist_ok=True)
    match controlnet_mode:
        case "canny":
            control = apply_canny(image)
        case "depth":
            control = apply_depth(image)
        case "pose":
            control = apply_pose(image)
        case "normal":  # deprecitated for normalbae
            control = apply_normalbae(image)
        case "scribble":
            control = apply_scribble(image)
        case "hed":
            control = apply_hed(image)
        case "ip2p":
            control = apply_base(image)
        case "softedge":
            control = apply_softedge(image)
        case "mlsd":
            control = apply_mlsd(image)
        case "lineart":
            control = apply_lineart(image)
        case "lineart_anime":
            control = apply_lineart_anime(image)
        case "shuffle":
            control = apply_shuffle(image)
        case "inpaint":
            control = image
        case "qrcode":
            control = apply_base(image)
        case _:
            print("Unknown control mode:", controlnet_mode)
            return None

    control = torch.stack([control for _ in range(1)], dim=0)
    control = rearrange(control, 'b h w c -> b c h w').clone()
    return control

def preview_controlnet(image, controlnet_mode, models_dir):
    global annotator_ckpts_path
    annotator_ckpts_path = os.path.join(models_dir, 'ControlNet', 'annotators_(not_your_models)' 'ckpts')
    os.makedirs(annotator_ckpts_path, exist_ok=True)
    match controlnet_mode:
        case "canny":
            control = apply_canny(image, preview=True)
        case "depth":
            control = apply_depth(image, preview=True)
        case "pose":
            control = apply_pose(image, preview=True)
        case "normal":  # deprecitated for normalbae
            control = apply_normalbae(image, preview=True)
        case "scribble":
            control = apply_scribble(image, preview=True)
        case "hed":
            control = apply_hed(image, preview=True)
        case "ip2p":
            control = apply_base(image, preview=True)
        case "softedge":
            control = apply_softedge(image, preview=True)
        case "mlsd":
            control = apply_mlsd(image, preview=True)
        case "lineart":
            control = apply_lineart(image, preview=True)
        case "lineart_anime":
            control = apply_lineart_anime(image, preview=True)
        case "shuffle":
            control = apply_shuffle(image, preview=True)
        case "inpaint":
            control = image
        case "qrcode":
            control = apply_base(image, preview=True)
        case _:
            print("Unknown control mode:", controlnet_mode)
            return None
    return control


def apply_canny(img, low_thr=100, high_thr=200, preview=False):
    img = cv2.Canny(img, low_thr, high_thr)
    img = HWC3(img)
    if preview:
        img = HWC3(img)
        return img
    control = torch.from_numpy(img.copy()).float().cuda() / 255.0
    return control


def apply_pose(img, preview=False):
    apply_openpose = OpenposeDetector(annotator_ckpts_path)
    img = apply_openpose(img, True)  # True passes full openpose
    img = HWC3(img)
    if preview:
        img = HWC3(img)
        return img
    control = torch.from_numpy(img.copy()).float().cuda() / 255.0
    return control


def apply_depth(img, preview=False):
    apply_midas = MidasDetector(annotator_ckpts_path)
    img, _ = apply_midas(img)
    img = HWC3(img)
    if preview:
        img = HWC3(img)
        return img
    control = torch.from_numpy(img.copy()).float().cuda() / 255.0
    return control


def apply_normal(img, preview=False):
    apply_midas = MidasDetector(annotator_ckpts_path)
    _, img = apply_midas(img)
    img = HWC3(img)
    img = img[:, :, ::-1]
    if preview:
        img = HWC3(img)
        return img
    control = torch.from_numpy(img.copy()).float().cuda() / 255.0
    return control


def apply_normalbae(img, preview=False):
    apply_normalbae_d = NormalBaeDetector(annotator_ckpts_path)
    img = apply_normalbae_d(img)
    img = HWC3(img)
    if preview:
        img = HWC3(img)
        return img
    control = torch.from_numpy(img.copy()).float().cuda() / 255.0
    return control


def apply_hed(img, preview=False):
    apply_hed_d = HEDdetector(annotator_ckpts_path)
    img = apply_hed_d(img)
    img = HWC3(img)
    if preview:
        img = HWC3(img)
        return img
    control = torch.from_numpy(img.copy()).float().cuda() / 255.0
    return control


def apply_scribble(img, preview=False):
    detected_map = np.zeros_like(img, dtype=np.uint8)
    detected_map[np.min(img, axis=2) < 127] = 255
    control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
    return control


def apply_softedge(img, preview=False):
    apply_pidi = PidiNetDetector(annotator_ckpts_path)
    img = apply_pidi(img)
    img = HWC3(img)
    if preview:
        img = HWC3(img)
        return img
    control = torch.from_numpy(img.copy()).float().cuda() / 255.0
    return control


def apply_mlsd(img, value_threshold=0.1, distance_threshold=0.1, preview=False):
    apply_mlsd_d = MLSDdetector(annotator_ckpts_path)
    img = apply_mlsd_d(img, value_threshold, distance_threshold)
    img = HWC3(img)
    if preview:
        img = HWC3(img)
        return img
    return img


def apply_lineart(img, coarse=False, preview=False):
    H, W, _ = img.shape
    apply_lineart_d = LineartDetector(annotator_ckpts_path)
    img = apply_lineart_d(img, coarse)
    img = HWC3(img)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    if preview:
        img = HWC3(img)
        return img
    control = 1 - torch.from_numpy(img.copy()).float().cuda() / 255.0
    return control


def apply_lineart_anime(img, preview=False):
    H, W, _ = img.shape
    apply_lineart_anime_d = LineartAnimeDetector(annotator_ckpts_path)
    img = apply_lineart_anime_d(img)
    img = HWC3(img)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    if preview:
        img = HWC3(img)
        return img
    control = 1 - torch.from_numpy(img.copy()).float().cuda() / 255.0
    return control


def apply_shuffle(img, preview=False):
    H, W, _ = img.shape
    apply_content_shuffle = ContentShuffleDetector()
    img = apply_content_shuffle(img, w=W, h=H, f=256)
    img = HWC3(img)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    if preview:
        img = HWC3(img)
        return img
    control = torch.from_numpy(img.copy()).float().cuda() / 255.0
    return control


def apply_base(img, preview=False):
    img = HWC3(img)
    if preview:
        img = HWC3(img)
        return img
    control = torch.from_numpy(img.copy()).float().cuda() / 255.0
    return control


def apply_inpaint(img, mask, preview=False):
    H, W, _ = img.shape
    mask_array = np.array(mask)
    mask_pixel = cv2.resize(mask_array, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    mask_pixel = cv2.GaussianBlur(mask_pixel, (0, 0), 8)

    detected_map = img.copy()
    detected_map[mask_pixel > 0.5] = - 255.0
    control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
    return control


def identity(img, **kwargs):
    return img


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y
