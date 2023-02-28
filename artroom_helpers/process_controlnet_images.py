import cv2
import numpy as np

from artroom_helpers.annotator.hed import HEDdetector, nms
from artroom_helpers.annotator.midas import MidasDetector
from artroom_helpers.annotator.openpose import OpenposeDetector

apply_openpose = OpenposeDetector()
apply_midas = MidasDetector()
apply_hed = HEDdetector()


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


def apply_scribble(img):
    img = apply_hed(img)
    img = HWC3(img)
    img = nms(img, 127, 3.0)
    img = cv2.GaussianBlur(img, (0, 0), 3.0)
    img[img > 4] = 255
    img[img < 255] = 0
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
