import gc
import os
import cv2
import numpy as np
import torch.cuda

from artroom_helpers.annotator.hed import HEDdetector, nms
from artroom_helpers.annotator.midas import MidasDetector
from artroom_helpers.annotator.zoe import ZoeDetector
from artroom_helpers.annotator.openpose import OpenposeDetector
from artroom_helpers.annotator.normalbae import NormalBaeDetector
from artroom_helpers.annotator.pidinet import PidiNetDetector
from artroom_helpers.annotator.mlsd import MLSDdetector
from artroom_helpers.annotator.lineart import LineartDetector
from artroom_helpers.annotator.lineart_anime import LineartAnimeDetector

apply_openpose = None
apply_midas = None
apply_zoe = None
apply_pidi = None 
apply_hed_d = None
apply_normalbae_d = None
apply_mlsd_d = None
apply_lineart_d = None 
apply_lineart_anime_d = None

def apply_controlnet(image, controlnet_mode):
    match controlnet_mode:
        case "canny":
            image = apply_canny(image)
        case "pose":
            image = apply_pose(image)
        case "depth":
            image = apply_depth(image)
        case "normal": #deprecitated for normalbae
            #image = apply_normal(image)
            image = apply_normalbae(image)
        case "scribble":
            image = apply_scribble(image)
        case "hed":
            image = apply_hed(image)
        case "ip2p":
            image = HWC3(image) #I don't think the image gets changed?
        case "softedge":
            image = apply_softedge(image)
        case "mlsd":
            image = apply_mlsd(image)
        # case "inpaint":
        #     image = 
        # case "lineart":
        #     image = 
        # case "lineart_anime":
        #     image = 
        # case "ip2p":
        #     image = 

        # case "tile":
        #     image = 
        # case "shuffle":
        #     image = 
        case _:
            print("Unknown control mode:", controlnet_mode)
    return image 

def init_cnet_stuff(controlnet_mode, models_dir):
    global apply_openpose, apply_midas, apply_hed_d, apply_normalbae_d, apply_pidi, apply_zoe, apply_mlsd_d, apply_lineart_d, apply_lineart_anime_d

    annotator_ckpts_path = os.path.join(models_dir, "ControlNet", "ControlNetConfigs")

    match controlnet_mode:
        case "pose":
            if apply_openpose is None:
                apply_openpose = OpenposeDetector(annotator_ckpts_path)
        case "depth":
            if apply_midas is None:
                apply_midas = MidasDetector(annotator_ckpts_path)
            # if apply_zoe is None:
            #     apply_zoe = ZoeDetector(annotator_ckpts_path)
        case "hed":
            if apply_hed_d is None:
                apply_hed_d = HEDdetector(annotator_ckpts_path)
        case "normal":
            if apply_midas is None:
                apply_midas = MidasDetector(annotator_ckpts_path)
            if apply_normalbae_d is None:
                apply_normalbae_d = NormalBaeDetector(annotator_ckpts_path)
        case "softedge":
            if apply_pidi is None:
                apply_pidi = PidiNetDetector(annotator_ckpts_path)
        case "mlsd":
            if apply_mlsd_d is None:
                apply_mlsd_d = MLSDdetector(annotator_ckpts_path)
        case "lineart":
            if apply_lineart_d is None:
                apply_lineart_d = LineartDetector(annotator_ckpts_path)
        case "lineart_anime":
            if apply_lineart_anime_d is None:
                apply_lineart_anime_d = LineartAnimeDetector(annotator_ckpts_path)

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
    img = apply_openpose(img, True) #True passes full openpose
    img = HWC3(img)
    return img


def apply_depth(img):
    if apply_midas is not None:
       img, _ = apply_midas(img)
    else:
        img, _ = apply_zoe(img)
    img = HWC3(img)
    return img


def apply_normal(img):
    _, img = apply_midas(img)
    img = HWC3(img)
    img = img[:, :, ::-1]
    return img

def apply_normalbae(img):
    img = apply_normalbae_d(img)
    img = HWC3(img)
    return img

def apply_hed(img):
    img = apply_hed_d(img)
    img = HWC3(img)
    return img


def apply_scribble(img):
    detected_map = np.zeros_like(img, dtype=np.uint8)
    detected_map[np.min(img, axis=2) < 127] = 255
    return detected_map

def apply_softedge(img):
    img= apply_pidi(img)
    img = HWC3(img)
    return img

def apply_mlsd(img, value_threshold = 0.1, distance_threshold = 0.1):
    img= apply_mlsd_d(img, value_threshold, distance_threshold)
    img = HWC3(img)
    return img


def apply_lineart(img):
    img= apply_mlsd_d(img)
    img = HWC3(img)
    return img


def apply_mlsd(img):
    img= apply_mlsd_d(img)
    img = HWC3(img)
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
