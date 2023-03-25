import face_alignment
import torch
import numpy as np
from scipy.spatial import ConvexHull

from safe import load as safe_load
from itertools import islice
from PIL import Image, ImageDraw
from safetensors import safe_open


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(ckpt, use_safe_load=True):
    print(f"Loading model from {ckpt}")
    if ".safetensors" in ckpt:
        pl_sd = {}
        with safe_open(ckpt, framework="pt", device="cpu") as f:
            for key in f.keys():
                pl_sd[key] = f.get_tensor(key)
    elif use_safe_load:
        pl_sd = safe_load(ckpt)
    else:
        pl_sd = torch.load(ckpt, map_location="cpu")

    if "state_dict" in pl_sd:
        return pl_sd["state_dict"]
    return pl_sd


def load_mask(mask, newH, newW):
    image = np.array(mask)
    image = Image.fromarray(image).convert("RGB")
    w, h = image.size

    # resize to integer multiple of 32
    w, h = map(lambda x: x - x % 64, (w, h))
    image = image.resize((newW, newH), resample=Image.LANCZOS)

    # image = image.resize((64, 64), resample=Image.LANCZOS)
    image = np.array(image)

    image = image.astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image


def image_grid(imgs, rows, cols, path):
    print("Making image grid...")
    assert len(imgs) <= rows * cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    grid.save(path)
    print("Grid finished")


def mask_from_face(img, w, h, face_idx=0):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device="cpu")
    img = np.array(img.resize((w, h)))

    lmarks = fa.get_landmarks(img)
    if len(lmarks) > 1:
        print(f"Multiple faces found! Selecing: {face_idx}")
    lmarks = lmarks[face_idx]
    lmarks = [(int(x[0]), int(x[1])) for x in lmarks]

    hull = ConvexHull(lmarks)
    lmarks = [lmarks[x] for x in hull.vertices]

    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).polygon(lmarks, outline=1, fill="white")
    return mask


def mask_background(img, remove_background):
    from artroom_helpers.rembg import rembg
    output = rembg.remove(img, session_model=remove_background, only_mask=True)
    return output.convert('L')
