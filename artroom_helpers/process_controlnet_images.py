import gc

import cv2
import numpy as np
import torch.cuda

from artroom_helpers.annotator.hed import HEDdetector, nms
from artroom_helpers.annotator.midas import MidasDetector
from artroom_helpers.annotator.openpose import OpenposeDetector

apply_openpose, apply_midas, apply_hed_d = None, None, None


def init_cnet_stuff(controlnet_mode):
    global apply_openpose, apply_midas, apply_hed_d
    match controlnet_mode:
        case "pose":
            if apply_openpose is None:
                apply_openpose = OpenposeDetector()
        case "depth" | "normal":
            if apply_midas is None:
                apply_midas = MidasDetector()
        case "hed":
            if apply_hed_d is None:
                apply_hed_d = HEDdetector()


def deinit_cnet_stuff():
    global apply_openpose, apply_midas, apply_hed_d
    del apply_openpose, apply_midas, apply_hed_d
    apply_openpose, apply_midas, apply_hed_d = None, None, None
    gc.collect()
    torch.cuda.empty_cache()


def apply_canny(img, low_thr=100, high_thr=200):
    img = cv2.Canny(img, low_thr, high_thr)
    img = HWC3(img)
    return img


def apply_pose(img):
    img, _ = apply_openpose(img)
    img = HWC3(img)
    return img


def apply_depth(img):
    img, _ = apply_midas(img)
    img = HWC3(img)
    return img


def apply_normal(img):
    _, img = apply_midas(img)
    img = HWC3(img)
    img = img[:, :, ::-1]
    return img


def apply_hed(img):
    img = apply_hed_d(img)
    img = HWC3(img)
    return img


def apply_scribble(img):
    detected_map = np.zeros_like(img, dtype=np.uint8)
    detected_map[np.min(img, axis=2) < 127] = 255
    return detected_map


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
