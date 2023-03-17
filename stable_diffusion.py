import threading
import warnings
import random

from scipy.spatial import ConvexHull

try:
    import face_recognition
except:
    pass
    # print("Face recognition not found, install it via: `pip install face_recognition`")
import matplotlib.pyplot as plt
import torch
import gc
import re
import time
import numpy as np
import json
import math
import os
import sys

from artroom_helpers.modules.cldm.ddim_hacked import DDIMSampler

sys.path.append("stable-diffusion/optimizedSD")
sys.path.append("artroom_helpers/modules")

from artroom_helpers.modules.lora_ext import create_network_and_apply_compvis
from artroom_helpers.process_controlnet_images import apply_pose, apply_depth, apply_canny, apply_normal, \
    apply_scribble, HWC3, apply_hed, init_cnet_stuff, deinit_cnet_stuff
from artroom_helpers.modules.cldm.model import create_model, load_state_dict

from safe import load as safe_load
from torchvision.utils import make_grid
from transformers import logging
from torch import autocast
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from itertools import islice
from einops import rearrange, repeat
from contextlib import nullcontext
from PIL import Image, ImageOps, ImageDraw
from safetensors import safe_open
from safetensors.torch import load_file

from artroom_helpers import support, inpainting
from artroom_helpers.prompt_parsing import weights_handling
from artroom_helpers.gpu_detect import get_gpu_architecture, get_device
from artroom_helpers.modules import HN

from ldm.util import instantiate_from_config

logging.set_verbosity_error()

warnings.filterwarnings("ignore", category=DeprecationWarning)


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


def mask_from_face(img, h, w, face_idx=0):
    def flatten(l):
        return [item for sublist in l for item in sublist]

    img = np.array(img.resize((w, h)))
    face_landmarks_list = face_recognition.face_landmarks(img)  # image - np array
    if len(face_landmarks_list) > 1:
        print(f"Warning: multiple faces detected: {len(face_landmarks_list)}")

    try:
        lmarks = flatten([face_landmarks_list[face_idx][x] for x in face_landmarks_list[face_idx].keys()])
    except:
        return None  # no face

    hull = ConvexHull(lmarks)
    lmarks = [lmarks[x] for x in hull.vertices]

    mask = Image.new("L", (img.shape[1], img.shape[0]), 0)
    ImageDraw.Draw(mask).polygon(lmarks, outline=1, fill="white")
    return mask


class StableDiffusion:
    def __init__(self, socketio=None, Upscaler=None):
        self.network = None
        self.config = None
        self.dtype = None
        self.Upscaler = Upscaler

        self.current_num = 0
        self.total_num = 0
        self.running = False

        self.artroom_path = None

        self.model = None
        self.modelCS = None
        self.modelFS = None

        self.ckpt = ''
        self.vae = ''
        self.loras = []
        self.controlnet_path = None

        self.can_use_half = get_gpu_architecture() == 'NVIDIA'
        self.device = get_device()
        self.speed = "Max"
        self.socketio = socketio
        self.v1 = False
        self.cc = self.get_cc()
        self.intermediate_path = ''
        # Generation Runtime Parameters

    def load_img(self, image, h0, w0, inpainting=False, controlnet_mode=None):
        w, h = image.size
        if not inpainting and h0 != 0 and w0 != 0:
            h, w = h0, w0

        # resize to integer multiple of 32
        w, h = map(lambda x: x - x % 64, (w, h))
        print(f"New image size ({w}, {h})")
        image = image.resize((w, h), resample=Image.LANCZOS)

        if controlnet_mode is not None:
            image = HWC3(np.array(image))
            match controlnet_mode:
                case "canny":
                    image = apply_canny(image)
                case "pose":
                    image = apply_pose(image)
                case "depth":
                    image = apply_depth(image)
                case "normal":
                    image = apply_normal(image)
                case "scribble":
                    image = apply_scribble(image)
                case "hed":
                    image = apply_hed(image)
                case _:
                    print("Unknown control mode:", controlnet_mode)

            control = torch.from_numpy(image.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(1)], dim=0)
            control = rearrange(control, 'b h w c -> b c h w').clone()
            return control

        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return 2. * image - 1.

    def get_cc(self):
        try:
            cc = torch.cuda.get_device_capability()
            cc = cc[0] * 10 + cc[1]
            return cc
        except:
            # probably no cuda gpus
            return 0

    def get_steps(self):
        if self.model:
            return self.current_num, self.total_num, self.model.current_step, self.model.total_steps
        else:
            return 0, 0, 0, 0

    def clean_up(self):
        self.total_num = 0
        self.current_num = 0
        if self.model:
            self.model.current_step = 0
            self.model.total_steps = 0

        self.running = False
        if self.device.type == "cuda" and self.v1:
            mem = torch.cuda.memory_allocated() / 1e6
            self.modelFS.to("cpu")
            while torch.cuda.memory_allocated() / 1e6 >= mem:
                time.sleep(1)

    def loaded_models(self):
        return self.model is not None

    def load_hypernet(self, path: str, safe_load_=True):
        hn_sd = load_model_from_config(path, safe_load_)
        hn = HN(hn_sd)
        return hn

    def inject_lora(self, path: str, weight_tenc=1.1, weight_unet=4):
        print(f'Loading Lora file :{path} with weight {weight_tenc}')
        du_state_dict = load_file(path)
        text_encoder = self.modelCS.cond_stage_model.to(self.device, dtype=self.dtype)
        # text_encoder = model.cond_stage_model.transformer.to(device, dtype=model.dtype)
        # text_encoder = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14')
        assert text_encoder is not None, "Text encoder is Null"

        network, info, state_dict = create_network_and_apply_compvis(
            du_state_dict, weight_tenc, weight_unet, text_encoder, unet=self.model)
        self.network = network.to(self.device, dtype=self.dtype)
        self.network.enable_loras(True)

    def deinject_lora(self, delete=True):
        self.network.enable_loras(False)
        self.network.restore(text_encoder=self.modelCS.cond_stage_model, unet=self.model)
        if delete:
            del self.network
            self.network = None
            torch.cuda.empty_cache()

    def load_vae(self, vae_path: str, safe_load_=True):
        self.modelFS.to(torch.float32)
        vae = load_model_from_config(vae_path, safe_load_)
        vae = {k: v for k, v in vae.items() if

               ("loss" not in k) and
               (k not in ['quant_conv.weight', 'quant_conv.bias', 'post_quant_conv.weight',
                          'post_quant_conv.bias'])}
        vae = {k.replace("encoder", "first_stage_model.encoder")
               .replace("decoder", "first_stage_model.decoder"): v for k, v in vae.items()}
        self.modelFS.load_state_dict(vae, strict=False)

    def load_ckpt(self, ckpt, speed, vae, loras=[], controlnet_path=None):
        assert ckpt != '', 'Checkpoint cannot be empty'
        if self.ckpt != ckpt or self.speed != speed or self.vae != vae or self.controlnet_path != controlnet_path:
            try:
                print("Setting up model...")
                self.set_up_models(ckpt, speed, vae, controlnet_path)
                print("Successfully set up model")
            except Exception as e:
                print(f"Setting up model failed: {e}")
                self.model = None
                self.modelCS = None
                self.modelFS = None
                return False
        try:
            if self.network:
                self.deinject_lora()
            if len(loras) > 0:
                for lora in loras:
                    self.inject_lora(path=lora['path'], weight_tenc=lora['weight'], weight_unet=lora['weight'])
        except Exception as e:
            print(f"Failed to load in Lora! {e}")
        return True

    def inject_controlnet(self, ckpt, path_sd15, path_sd15_with_control):
        print("Injecting controlnet..")

        def get_state_dict(d):
            return d.get('state_dict', d)

        def load_state_dict(ckpt_path, location='cpu'):
            _, extension = os.path.splitext(ckpt_path)
            if extension.lower() == ".safetensors":
                import safetensors.torch
                state_dict = safetensors.torch.load_file(ckpt_path, device=location)
            else:
                state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
            state_dict = get_state_dict(state_dict)
            return state_dict

        def get_node_name(name, parent_name):
            if len(name) <= len(parent_name):
                return False, ''
            p = name[:len(parent_name)]
            if p != parent_name:
                return False, ''
            return True, name[len(parent_name):]

        sd15_state_dict = load_state_dict(path_sd15)
        sd15_with_control_state_dict = load_state_dict(path_sd15_with_control)
        input_state_dict = load_state_dict(ckpt)
        keys = sd15_with_control_state_dict.keys()

        final_state_dict = {}
        for key in keys:
            is_first_stage, _ = get_node_name(key, 'first_stage_model')
            is_cond_stage, _ = get_node_name(key, 'cond_stage_model')
            if is_first_stage or is_cond_stage:
                final_state_dict[key] = input_state_dict[key]
                continue
            p = sd15_with_control_state_dict[key]
            is_control, node_name = get_node_name(key, 'control_')
            if is_control:
                sd15_key_name = 'model.diffusion_' + node_name
            else:
                sd15_key_name = key
            if sd15_key_name in input_state_dict:
                p_new = p + input_state_dict[sd15_key_name] - sd15_state_dict[sd15_key_name]
            else:
                p_new = p
            final_state_dict[key] = p_new
        del sd15_with_control_state_dict, sd15_state_dict, input_state_dict

        return final_state_dict

    def set_up_models(self, ckpt, speed, vae, controlnet_path=None):
        speed = speed if self.device.type != 'privateuseone' else "High"
        self.socketio.emit('get_status', {'status': "Loading Model"})
        try:
            del self.model
            del self.modelFS
            del self.modelCS
            self.model = None
            self.modelFS = None
            self.modelCS = None
        except:
            pass
        torch.cuda.empty_cache()
        gc.collect()

        sd = load_model_from_config(f"{ckpt}")
        if sd:
            print("Model safety check passed")
        else:
            print("Model safety check died midways")
            return

        print("Setting up config...")
        parameterization = "eps"
        if sd['model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight'].shape[1] == 1024:
            print("Detected v2 model")
            self.config = os.path.splitext(ckpt)[0] + ".yaml"
            if os.path.exists(self.config):
                # so, we can't select their config because our is modified to our implementation
                # still, there are only two types of configs, the ones with parameterization in them and without
                if "parameterization" in "".join(open(self.config, "r").readlines()):
                    parameterization = "v"
                    self.config = 'stable-diffusion/optimizedSD/configs/v2/v2-inference-v.yaml'
                else:
                    self.config = 'stable-diffusion/optimizedSD/configs/v2/v2-inference.yaml'
            else:
                self.config = 'stable-diffusion/optimizedSD/configs/v2/v2-inference.yaml'
            print(f"v2 conf: {self.config}")
            config = OmegaConf.load(f"{self.config}")
            self.model = instantiate_from_config(config.model)
            _, _ = self.model.load_state_dict(sd, strict=False)
            self.model.eval()
            self.model.parameterization = parameterization
            self.model.cdevice = self.device
            self.model.to(self.device)
            self.modelCS = self.model  # just link without a copy
            self.modelFS = self.model  # just link without a copy
            self.v1 = False
        else:
            self.v1 = True
            if sd['model.diffusion_model.input_blocks.0.0.weight'].shape[1] == 9:
                print("Detected runwayml inpainting model")
                if speed == 'Low':
                    self.config = 'stable-diffusion/optimizedSD/configs/runway/v1-inference_lowvram.yaml'
                elif speed == 'Medium':
                    self.config = 'stable-diffusion/optimizedSD/configs/runway/v1-inference_lowvram.yaml'
                elif speed == 'High':
                    self.config = 'stable-diffusion/optimizedSD/configs/runway/v1-inference.yaml'
                elif speed == 'Max':
                    self.config = 'stable-diffusion/optimizedSD/configs/runway/v1-inference_xformer.yaml'
                else:
                    print(f"Dafuq is {speed}")
                    self.config = 'stable-diffusion/optimizedSD/configs/runway/v1-inference.yaml'
            elif sd['model.diffusion_model.input_blocks.0.0.weight'].shape[1] == 8:
                print("Detected pix2pix model")
                if speed == 'Low':
                    self.config = 'stable-diffusion/optimizedSD/configs/instruct_pix2pix/v1-inference_lowvram.yaml'
                elif speed == 'Medium':
                    self.config = 'stable-diffusion/optimizedSD/configs/instruct_pix2pix/v1-inference_lowvram.yaml'
                elif speed == 'High':
                    self.config = 'stable-diffusion/optimizedSD/configs/instruct_pix2pix/v1-inference.yaml'
                elif speed == 'Max':
                    self.config = 'stable-diffusion/optimizedSD/configs/instruct_pix2pix/v1-inference_xformer.yaml'
                else:
                    print(f"Dafuq is {speed}")
                    self.config = 'stable-diffusion/optimizedSD/configs/instruct_pix2pix/v1-inference.yaml'
            else:
                print("Loading ordinary model")
                if 60 <= self.cc <= 86 and self.device.type == "cuda":
                    print("Congrats, your gpu supports xformers, autoselecting it")
                    self.config = 'stable-diffusion/optimizedSD/configs/v1/v1-inference_xformer.yaml'
                else:
                    print("Using speed mode from artroom settings")
                    if speed == 'Low':
                        self.config = 'stable-diffusion/optimizedSD/configs/v1/v1-inference_lowvram.yaml'
                    elif speed == 'Medium':
                        self.config = 'stable-diffusion/optimizedSD/configs/v1/v1-inference_lowvram.yaml'
                    elif speed == 'High':
                        self.config = 'stable-diffusion/optimizedSD/configs/v1/v1-inference.yaml'
                    elif speed == 'Max':
                        self.config = 'stable-diffusion/optimizedSD/configs/v1/v1-inference_xformer.yaml'
                    else:
                        print(f"Not recognized speed: {speed}")
                        self.config = 'stable-diffusion/optimizedSD/configs/v1/v1-inference.yaml'
            li = []
            lo = []
            for key, value in sd.items():
                sp = key.split('.')
                if (sp[0]) == 'model':
                    if 'input_blocks' in sp:
                        li.append(key)
                    elif 'middle_block' in sp:
                        li.append(key)
                    elif 'time_embed' in sp:
                        li.append(key)
                    else:
                        lo.append(key)
            for key in li:
                sd['model1.' + key[6:]] = sd.pop(key)
            for key in lo:
                sd['model2.' + key[6:]] = sd.pop(key)
            config = OmegaConf.load(f"{self.config}")
            self.model = instantiate_from_config(config.modelUNet)
            _, _ = self.model.load_state_dict(sd, strict=False)
            self.model.eval()
            self.model.cdevice = self.device
            self.model.unet_bs = 1  # unet_bs=1

            self.model.turbo = (speed != 'Low')

            self.modelCS = instantiate_from_config(config.modelCondStage)
            _, _ = self.modelCS.load_state_dict(sd, strict=False)
            self.modelCS.eval()
            self.modelCS.cond_stage_model.device = self.device

            self.modelFS = instantiate_from_config(config.modelFirstStage)
            _, _ = self.modelFS.load_state_dict(sd, strict=False)
            self.modelFS.eval()

        if self.can_use_half:
            self.model.half()
            self.modelCS.half()
            self.modelFS.half()
            # torch.set_default_tensor_type(torch.HalfTensor)
        else:
            self.model.to(torch.float32)
            self.modelCS.to(torch.float32)
            self.modelFS.to(torch.float32)

        if controlnet_path is not None:
            self.control_model = create_model("stable-diffusion/optimizedSD/configs/cnet/cldm_v15.yaml").cpu()
            sd = self.inject_controlnet(ckpt, os.path.dirname(ckpt) + "/model.ckpt", controlnet_path)
            self.control_model.load_state_dict(sd)
        else:
            self.control_model = None

        del sd

        # if self.controlnet_path is not None:
        #     self.hack_everything(clip_skip=2)

        self.ckpt = ckpt.replace(os.sep, '/')
        self.speed = speed
        self.vae = vae
        self.controlnet_path = controlnet_path

        print("Model loading finished")
        print("Loading vae")
        if '.vae' in vae:
            try:
                self.load_vae(vae)
                print("Loading vae finished")
            except:
                print("Failed to load vae")
        self.socketio.emit('get_status', {'status': "Finished Loading Model"})

    def get_image(self, init_image_str, mask_b64):
        if len(init_image_str) == 0:
            return None

        if init_image_str[:4] == 'data':
            print("Loading image from b64")
            init_image = support.b64_to_image(init_image_str)
        else:
            print(f"Loading from path {init_image_str}")
            init_image = Image.open(init_image_str)

        if len(mask_b64) > 0:
            try:
                init_image = inpainting.infill_patchmatch(init_image)
            except Exception as e:
                print(f"Failed to outpaint the alpha layer {e}")
        return init_image.convert("RGB")

    def load_image(self, image, h0, w0, inpainting=False):
        w, h = image.size
        if not inpainting and h0 != 0 and w0 != 0:
            h, w = h0, w0

        # resize to integer multiple of 32
        w, h = map(lambda x: x - x % 64, (w, h))
        print(f"New image size ({w}, {h})")
        image = image.resize((w, h), resample=Image.LANCZOS)

        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        init_image = 2. * image - 1.
        init_image = init_image.to(self.device)
        _, _, H, W = image.shape
        if self.can_use_half:
            init_image = init_image.half()

        return init_image, H, W

    def callback_fn(self, x, enabled=True):
        if not enabled:
            return

        current_num, total_num, current_step, total_steps = self.get_steps()

        self.socketio.emit('get_progress',
                           {'current_step': current_step + 1, 'total_steps': total_steps, 'current_num': current_num,
                            'total_num': total_num})

        def send_intermediates(x):
            def float_tensor_to_pil(tensor: torch.Tensor):
                """aka torchvision's ToPILImage or DiffusionPipeline.numpy_to_pil
                (Reproduced here to save a torchvision dependency in this demo.)
                """
                tensor = (((tensor + 1) / 2)
                          .clamp(0, 1)  # change scale from -1..1 to 0..1
                          .mul(0xFF)  # to 0..255
                          .byte())
                tensor = rearrange(tensor, 'c h w -> h w c')
                return Image.fromarray(tensor.cpu().numpy())

            x = x.detach().cpu()
            x = make_grid(x, nrow=x.shape[0]).to(torch.float32)
            x = float_tensor_to_pil(torch.einsum('...lhw,lr -> ...rhw', x, torch.tensor([
                #   R        G        B
                [0.298, 0.207, 0.208],  # L1
                [0.187, 0.286, 0.173],  # L2
                [-0.158, 0.189, 0.264],  # L3
                [-0.184, -0.271, -0.473],  # L4
            ], dtype=torch.float32)))

            x = x.resize((x.width * 8, x.height * 8))
            # x.save(os.path.join(self.intermediate_path, f'{current_step:04}.png'), "PNG")
            self.socketio.emit('intermediate_image', {'b64': support.image_to_b64(x)})

        if self.show_intermediates and current_step % 5 == 0:  # every 5 steps
            threading.Thread(target=send_intermediates, args=(x,)).start()

    def interrupt(self):
        if self.running and self.model:
            self.model.interrupted_state = True
            self.running = False

    def generate(
            self, text_prompts="", negative_prompts="", init_image_str="", mask_b64="",
            invert=False, txt_cfg_scale=1.5, steps=50, H=512, W=512, strength=0.75, cfg_scale=7.5, seed=-1,
            sampler="ddim", C=4, ddim_eta=0.0, f=8, n_iter=4, batch_size=1, ckpt="", vae="", loras=None,
            image_save_path="", speed="High", skip_grid=False, palette_fix=False, batch_id=0, highres_fix=False,
            long_save_path=False, show_intermediates=False, controlnet=None, auto_mask_face=False
    ):

        if loras is None:
            loras = []
        self.show_intermediates = show_intermediates

        controlnet_ckpts = {
            "canny": os.path.join(os.path.dirname(ckpt), "ControlNet", "control_sd15_canny.pth"),
            "depth": os.path.join(os.path.dirname(ckpt), "ControlNet", "control_sd15_depth.pth"),
            "normal": os.path.join(os.path.dirname(ckpt), "ControlNet", "control_sd15_normal.pth"),
            "pose": os.path.join(os.path.dirname(ckpt), "ControlNet", "control_sd15_openpose.pth"),
            "scribble": os.path.join(os.path.dirname(ckpt), "ControlNet", "control_sd15_scribble.pth"),
            "hed": os.path.join(os.path.dirname(ckpt), "ControlNet", "control_sd15_hed.pth"),
            "None": None,
            "none": None
        }

        # controlnet_ckpts = {
        #     "canny": os.path.join(os.path.dirname(ckpt), "ControlNet", "controlnetPreTrained_cannyV10.safetensors"),
        #     "depth": os.path.join(os.path.dirname(ckpt), "ControlNet", "controlnetPreTrained_depthV10.safetensors"),
        #     "normal": os.path.join(os.path.dirname(ckpt), "ControlNet", "controlnetPreTrained_normalV10.safetensors"),
        #     "pose": os.path.join(os.path.dirname(ckpt), "ControlNet", "controlnetPreTrained_openposeV10.safetensors"),
        #     "scribble": os.path.join(os.path.dirname(ckpt), "ControlNet", "controlnetPreTrained_scribbleV10.safetensors"),
        #     "hed": os.path.join(os.path.dirname(ckpt), "ControlNet", "controlnetPreTrained_hedV10.safetensors"),
        #     "None": None,
        #     "none": None
        # }

        print(f"Using controlnet {controlnet}")
        controlnet_path = controlnet_ckpts[controlnet]
        if controlnet_path is None:
            deinit_cnet_stuff()
            controlnet = None
        else:
            init_cnet_stuff(controlnet)

        self.running = True

        self.dtype = torch.float16 if self.can_use_half else torch.float32

        if palette_fix:
            print("Using palette fix")
            padding = 32
        else:
            padding = 0

        if batch_id == 0:
            batch_id = random.randint(1, 922337203685)

        W += padding * 2
        H += padding * 2
        oldW, oldH = W, H

        if W * H >= 1024 * 1024 and highres_fix:
            highres_fix_steps = math.ceil((W * H) / (512 * 512) / 4)
            W, H = W // highres_fix_steps, H // highres_fix_steps
            W = math.floor(W / 64) * 64
            H = math.floor(H / 64) * 64
        else:
            highres_fix_steps = 1

        print("Starting generate process...")

        torch.cuda.empty_cache()
        gc.collect()
        seed_everything(seed)

        if (len(init_image_str) > 0 and sampler == 'plms') or (len(mask_b64) > 0 and sampler == 'plms'):
            if len(mask_b64) > 0:
                print("Currently, only DDIM works with masks. Switching samplers to DDIM")
            sampler = 'ddim'

        ddim_steps = int(steps / strength)

        self.load_ckpt(ckpt, speed, vae, loras, controlnet_path)
        if not self.model:
            print("Setting up model failed")
            return 'Failure'

        print("Generating...")
        self.socketio.emit('get_status', {'status': "Generating"})
        os.makedirs(image_save_path, exist_ok=True)

        if len(init_image_str) > 0:
            if init_image_str[:4] == 'data':
                print("Loading image from b64")
                image = support.b64_to_image(init_image_str)
            else:
                print(f"Loading from path {init_image_str}")
                image = Image.open(init_image_str)

            if padding > 0:
                w, h = image.size
                # Create a white image with the desired padding size
                padding_img = Image.fromarray(
                    (np.random.rand(h + 2 * padding, w + 2 * padding, 3) * 255).astype(np.uint8), "RGB")
                # Paste the original image onto the white image
                padding_img.paste(image, (padding, padding))
                # Update the image variable to be the padded image
                image = padding_img

            if len(mask_b64) > 0:
                try:
                    image = inpainting.infill_patchmatch(image)
                except Exception as e:
                    print(f"Failed to outpaint the alpha layer {e}")

            init_image = self.load_img(image.convert('RGB'), H, W, inpainting=(len(mask_b64) > 0),
                                       controlnet_mode=controlnet).to(self.device)
            if controlnet_path is not None:
                control = init_image.clone()
            else:
                control = None
            _, _, H, W = init_image.shape
            init_image = init_image.to(self.dtype)
        else:
            image = None
            init_image = None
            control = None

        mode = "default" if not self.v1 or (
                self.v1 and self.model.model1.diffusion_model.input_blocks[0][0].weight.shape[1] == 4) else (
            "runway" if self.model.model1.diffusion_model.input_blocks[0][0].weight.shape[1] == 9 else "pix2pix"
        )

        if mode == "pix2pix":
            sampler = "ddim"
            ddim_steps = steps

        if mode != "default":
            highres_fix_steps = 1

        print("Prompt:", text_prompts)
        data = [batch_size * text_prompts]
        print("Negative Prompt:", negative_prompts)
        negative_prompts_data = [batch_size * negative_prompts]

        if long_save_path:
            sample_path = os.path.join(image_save_path, re.sub(
                r'\W+', '', "_".join(text_prompts.split())))[:150]
        else:
            sample_path = image_save_path

        # self.intermediate_path = os.path.join(sample_path, 'intermediates/', f'{batch_id}/')
        # os.makedirs(self.intermediate_path, exist_ok=True)
        base_count = len(os.listdir(sample_path))

        if init_image is not None:
            if self.v1:
                self.modelFS.to(self.device)

        self.total_num = n_iter * highres_fix_steps
        all_samples = []
        precision_scope = autocast if self.can_use_half else nullcontext
        with torch.no_grad():
            for n in range(n_iter):
                if not self.running:
                    self.clean_up()
                    return

                if long_save_path:
                    save_name = f"{base_count:05}_seed_{str(seed)}.png"
                else:
                    prompt_name = re.sub(
                        r'\W+', '', '_'.join(text_prompts.split()))[:100]
                    save_name = f"{base_count:05}_{prompt_name}_seed_{str(seed)}.png"

                for prompts in data:
                    with precision_scope(self.device.type):
                        if self.v1:
                            self.modelCS.to(self.device)
                        uc = None
                        if cfg_scale != 1.0:
                            uc = self.modelCS.get_learned_conditioning(
                                negative_prompts_data)
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        weighted_prompt = weights_handling(prompts)
                        if type(weighted_prompt) == str:
                            weighted_prompt = [prompts]
                        print(f"Weighted prompts: {weighted_prompt}")
                        if len(weighted_prompt) > 1:
                            c = torch.zeros_like(uc)
                            weights_greater_than_zero = sum([wp[1] - 1 for wp in weighted_prompt if wp[1] > 1]) + 1
                            weighted_prompt_joined = ", ".join([wp[0] for wp in weighted_prompt])
                            c = self.modelCS.get_learned_conditioning(weighted_prompt_joined).to(self.device)
                            c /= weights_greater_than_zero
                            for i in range(len(weighted_prompt)):
                                weight = weighted_prompt[i][1]
                                if weight > 1:
                                    c_weighted = self.modelCS.get_learned_conditioning(weighted_prompt[i][0]).to(
                                        self.device)
                                    c = torch.add(c, c_weighted, alpha=(weight - 1) / weights_greater_than_zero)
                        else:
                            c = self.modelCS.get_learned_conditioning(prompts).to(self.device)
                        shape = [batch_size, C, H // f, W // f]

                        x0 = None
                        for ij in range(1, highres_fix_steps + 1):
                            self.current_num = n * highres_fix_steps + ij - 1
                            self.model.current_step = 0
                            self.model.total_steps = steps * highres_fix_steps

                            if ij > 1:
                                strength = 0.1
                                ddim_steps = int(steps / strength)
                            if init_image is not None:
                                init_image = init_image.to(self.device)
                                init_image = repeat(
                                    init_image, '1 ... -> b ...', b=batch_size)
                                init_latent_1stage = self.modelFS.encode_first_stage(init_image)
                                init_latent_1stage = init_latent_1stage.mode() if mode == "pix2pix" else init_latent_1stage
                                init_latent = self.modelFS.get_first_stage_encoding(init_latent_1stage).to(self.device)

                                x0 = self.model.stochastic_encode(
                                    init_latent,
                                    torch.tensor(
                                        [steps] * batch_size).to(self.device),
                                    seed,
                                    ddim_eta,
                                    ddim_steps,
                                )
                            if auto_mask_face and image is not None:
                                mask_image = mask_from_face(image.convert('RGB'), H, W)
                            elif len(mask_b64) > 0:
                                if mask_b64[:4] == 'data':
                                    print("Loading mask from b64")
                                    mask_image = support.b64_to_image(mask_b64).convert('L')
                                elif os.path.exists(mask_b64):
                                    mask_image = Image.open(mask_b64).convert("L")
                            else:
                                mask_image = None

                            if mask_image is not None:
                                if invert:
                                    mask_image = ImageOps.invert(mask_image)

                                if padding > 0:
                                    w, h = mask_image.size

                                    # Create a white image with the desired padding size
                                    padding_img = Image.new("RGB", (w + 2 * padding, h + 2 * padding), (255, 255, 255))
                                    # Paste the original image onto the white image
                                    padding_img.paste(mask_image, (padding, padding))
                                    # Update the image variable to be the padded image
                                    mask_image = padding_img

                                mask = load_mask(mask_image, init_latent.shape[2], init_latent.shape[3]) \
                                    .to(self.device)
                                mask = mask[0][0].unsqueeze(0).repeat(4, 1, 1).unsqueeze(0)
                                mask = repeat(mask, '1 ... -> b ...', b=batch_size)
                                x_T = init_latent
                            else:
                                mask = None
                                x_T = None

                            x0 = x0 if (init_image is None or "ddim" in sampler.lower()) else init_latent
                            x0 = init_latent_1stage if mode == "pix2pix" else x0

                            if controlnet is not None and controlnet.lower() != "none" and control is not None:
                                # control = torch.load("control.torch")
                                c = {"c_concat": [control], "c_crossattn": [c]}
                                uc = {"c_concat": [control], "c_crossattn": [uc]}
                                self.control_model.control_scales = [1.0] * 13
                                self.control_model.to(self.device)
                                ddim_sampler = DDIMSampler(self.control_model)
                                x0 = ddim_sampler.sample(
                                    steps,
                                    batch_size,
                                    tuple(shape[1:]),
                                    c,
                                    verbose=False,
                                    eta=ddim_eta,
                                    unconditional_guidance_scale=cfg_scale,
                                    callback=self.callback_fn,
                                    unconditional_conditioning=uc)
                                try:
                                    self.control_model.cpu()
                                except:  # means it's still under usage by another thread
                                    pass
                            else:
                                x0 = self.model.sample(
                                    S=steps,
                                    conditioning=c,
                                    x0=x0,
                                    S_ddim_steps=ddim_steps,
                                    unconditional_guidance_scale=cfg_scale,
                                    txt_scale=txt_cfg_scale,
                                    unconditional_conditioning=uc,
                                    eta=ddim_eta,
                                    sampler=sampler,
                                    shape=shape,
                                    batch_size=batch_size,
                                    seed=seed,
                                    mask=mask,
                                    x_T=x_T,
                                    callback=self.callback_fn,
                                    mode=mode
                                )
                            if self.v1:
                                self.modelFS.to(self.device)

                            self.model.to(torch.float32)
                            self.modelCS.to(torch.float32)
                            self.modelFS.to(torch.float32)
                            # torch.set_default_tensor_type(torch.FloatTensor)

                            x_samples_ddim = self.modelFS.decode_first_stage(
                                x0[0].to(torch.float32).unsqueeze(0))

                            x_sample = torch.clamp(
                                (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            x_sample = 255. * \
                                       rearrange(
                                           x_sample[0].cpu().numpy(), 'c h w -> h w c')
                            out_image = Image.fromarray(
                                x_sample.astype(np.uint8))

                            if self.can_use_half:
                                self.model.half()
                                self.modelCS.half()
                                self.modelFS.half()
                                # torch.set_default_tensor_type(torch.HalfTensor)

                            if ij < highres_fix_steps - 1:
                                init_image = self.load_img(
                                    out_image, H * (ij + 1), W * (ij + 1), inpainting=(len(mask_b64) > 0),
                                    controlnet_mode=None).to(self.device).to(self.dtype)  # we only encode cnet 1 time
                            elif ij == highres_fix_steps - 1:
                                init_image = self.load_img(out_image, oldH, oldW, inpainting=(len(mask_b64) > 0),
                                                           controlnet_mode=None).to(self.device).to(self.dtype)
                            if padding > 0:
                                w, h = out_image.size
                                out_image = out_image.crop((padding, padding, w - padding, h - padding))
                            elif mask is not None:
                                if init_image_str[:4] == 'data':
                                    original_init_image = support.b64_to_image(init_image_str).convert('RGB')
                                else:
                                    original_init_image = Image.open(init_image_str).convert('RGB')
                                out_image = support.repaste_and_color_correct(result=out_image,
                                                                              init_image=original_init_image,
                                                                              init_mask=mask_image, mask_blur_radius=8)
                            if not self.running:
                                break

                        exif_data = out_image.getexif()
                        # Does not include Mask, ImageB64, or if Inverted. Only settings for now
                        settings_data = {
                            "text_prompts": text_prompts,
                            "negative_prompts": negative_prompts,
                            "steps": steps,
                            "H": H,
                            "W": W,
                            "strength": strength,
                            "cfg_scale": cfg_scale,
                            "seed": seed,
                            "sampler": sampler,
                            "ckpt": os.path.basename(ckpt),
                            "vae": os.path.basename(vae)
                        }
                        # 0x9286 Exif Code for UserComment
                        exif_data[0x9286] = json.dumps(settings_data)
                        if not self.model.interrupted_state:
                            out_image.save(
                                os.path.join(sample_path, save_name), "PNG", exif=exif_data)

                            self.socketio.emit('get_images', {'b64': support.image_to_b64(out_image),
                                                              'path': os.path.join(sample_path, save_name),
                                                              'batch_id': batch_id})

                        base_count += 1
                        seed += 1
                        if len(init_image_str) == 0:  # Resets highres_fix starting image
                            init_image = None
                        if not skip_grid and n_iter > 1:
                            all_samples.append(out_image)

            if not skip_grid and n_iter > 1:
                # additionally, save as grid
                rows = int(np.sqrt(len(all_samples)))
                cols = int(np.ceil(len(all_samples) / rows))
                os.makedirs(sample_path + "/grids", exist_ok=True)
                image_grid(all_samples, rows, cols, path=os.path.join(
                    sample_path + "/grids", f'grid-{len(os.listdir(sample_path + "/grids")):04}.png'))
        self.clean_up()
