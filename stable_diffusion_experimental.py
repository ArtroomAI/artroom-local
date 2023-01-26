from safe import load as safe_load
import warnings
from artroom_helpers.prompt_parsing import weights_handling, split_weighted_subprompts
from artroom_helpers.gpu_detect import get_gpu_architecture
from skimage import exposure
import cv2
import random
from transformers import logging
from torch import autocast
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from itertools import islice
from einops import rearrange, repeat
from contextlib import nullcontext
from PIL import Image, ImageOps
import torch
import gc
import re
import time
import numpy as np
import json
import math
import os
import sys
from safetensors import safe_open
from artroom_helpers import support, inpainting

sys.path.append("stable-diffusion/optimizedSD")
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


class StableDiffusion:
    def __init__(self, socketio=None, Upscaler = None):
        self.Upscaler = Upscaler

        self.current_num = 0
        self.total_num = 0
        self.stage = ''
        self.running = False

        self.artroom_path = None
        self.latest_images_part1 = []
        self.latest_images_part2 = []
        self.latest_images_id = 0

        self.model = None
        self.modelCS = None
        self.modelFS = None

        self.ckpt = ''
        self.vae = ''
        self.image_save_path = os.environ['USERPROFILE'] + '/Desktop/'
        self.long_save_path = False
        self.highres_fix = False
        self.is_nvidia = get_gpu_architecture() == 'NVIDIA'
        self.device = 'cpu' if get_gpu_architecture() == 'None' else "cuda"
        self.speed = "Max"
        self.socketio = socketio
        self.v1 = False
        self.cc = self.get_cc()

        #Generation Runtime Parameters
        
    def get_cc(self):
        try:
            cc = torch.cuda.get_device_capability()
            cc = cc[0] * 10 + cc[1]
            return cc
        except:
            # probably no cuda gpus
            return 0

    def set_artroom_path(self, path):
        print("Setting up artroom path")
        self.artroom_path = path
        # loaded = False
        if os.path.exists(f"{self.artroom_path}/artroom/settings/sd_settings.json"):
            print("Loading model from sd_settings.json")
            sd_settings = json.load(
                open(f"{self.artroom_path}/artroom/settings/sd_settings.json"))
            self.image_save_path = sd_settings['image_save_path']
            self.long_save_path = sd_settings['long_save_path']
            self.highres_fix = sd_settings['highres_fix']

            print("Welcome to Artroom!")

    def get_steps(self):
        if self.model:
            return self.current_num, self.total_num, self.model.current_step, self.model.total_steps
        else:
            return 0, 0, 0, 0

    def get_latest_images(self):
        return self.latest_images_part1 + self.latest_images_part2

    def get_latest_image(self):
        latest_images = self.get_latest_images()
        if len(latest_images) > 0:
            return support.image_to_b64(Image.open(latest_images[-1]).convert('RGB'))
        else:
            return ''

    def add_to_latest(self, new_image: Image.Image, path=""):
        self.latest_images_part2.append({"b64": support.image_to_b64(new_image.convert('RGB')), "path": path})

    def clean_up(self):
        self.total_num = 0
        self.current_num = 0
        if self.model:
            self.model.current_step = 0
            self.model.total_steps = 0

        self.stage = ""
        self.running = False
        if self.device != "cpu" and self.v1:
            mem = torch.cuda.memory_allocated() / 1e6
            self.modelFS.to("cpu")
            while torch.cuda.memory_allocated() / 1e6 >= mem:
                time.sleep(1)

    def loaded_models(self):
        return self.model is not None

    def load_vae(self, vae_path, safe_load_=True):
        vae = load_model_from_config(vae_path, safe_load_)
        vae = {k: v for k, v in vae.items() if
        
               ("loss" not in k) and
               (k not in ['quant_conv.weight', 'quant_conv.bias', 'post_quant_conv.weight',
                          'post_quant_conv.bias'])}
        vae = {k.replace("encoder", "first_stage_model.encoder")
                   .replace("decoder", "first_stage_model.decoder"): v for k, v in vae.items()}
        self.modelFS.load_state_dict(vae, strict=False)

    def load_ckpt(self, ckpt, speed, vae):
        print(f"Attempting to load {ckpt}, speed: {speed}")
        assert ckpt != '', 'Checkpoint cannot be empty'
        if self.ckpt != ckpt or self.speed != speed or self.vae != vae:
            try:
                print("Setting up model...")
                self.set_up_models(ckpt, speed, vae)
                print("Successfully set up model")
                return True
            except Exception as e:
                print(f"Setting up model failed: {e}")
                self.stage = ""
                self.model = None
                self.modelCS = None
                self.modelFS = None
                return False

    def set_up_models(self, ckpt, speed, vae):
        print("Loading in model...")
        self.stage = "Loading Model"
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

        print("Loading model from config")
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
            else:
                print("Loading ordinary model")
                if 60 <= self.cc <= 86:
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
        if self.is_nvidia:
            self.model.half()
            self.modelCS.half()
            self.modelFS.half()
            torch.set_default_tensor_type(torch.HalfTensor)
        del sd
        self.ckpt = ckpt.replace(os.sep, '/')
        self.speed = speed
        self.vae = vae
        self.stage = "Finished Loading Model"
        print("Model loading finished")
        print("Loading vae")
        if '.vae' in vae:
            try:
                self.load_vae(vae)
                print("Loading vae finished")
            except:
                print("Failed to load vae")

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
        init_image =  2. * image - 1.
        init_image = init_image.to(self.device)
        _, _, H, W = image.shape
        if self.is_nvidia:
            init_image = init_image.half()

        return init_image, H, W

    @torch.no_grad()
    def generate_image(
        self, 
        prompts_data="", 
        negative_prompts_data="", 
        precision_scope=None, 
        starting_image = None,        
        mask_b64="",
        invert=False,
        steps=50, 
        H=512, 
        W=512, 
        cfg_scale=7.5, 
        seed=-1,
        sampler="ddim", 
        C=4, 
        ddim_eta=0.0, 
        f=8, 
        ddim_steps = 0,
        batch_size=1):

        if starting_image:
            init_image, H, W = self.load_image(starting_image, H, W, inpainting=(len(mask_b64) > 0))
        else:
            init_image = None
        for prompts in prompts_data:
            with precision_scope(self.device):
                if self.v1:
                    self.modelCS.to(self.device)
                uc = None
                if cfg_scale != 1.0:
                    uc = self.modelCS.get_learned_conditioning(
                        negative_prompts_data)
                if isinstance(prompts, tuple):
                    prompts = list(prompts)
                weighted_prompt = weights_handling([prompts])
                print(f"Weighted prompts: {weighted_prompt}")
                if len(weighted_prompt) > 1:
                    c = torch.zeros_like(uc)
                    # normalize each "sub prompt" and add it
                    for i in range(len(weighted_prompt)):
                        weight = weighted_prompt[i][1]
                        # if not skip_normalize:
                        c = torch.add(c, self.modelCS.get_learned_conditioning(
                            weighted_prompt[i][0]), alpha=weight)
                else:
                    c = self.modelCS.get_learned_conditioning(prompts)
                shape = [batch_size, C, H // f, W // f]

                print("Sampler", sampler)
                x0 = None

                if init_image is not None:
                    init_image = repeat(
                        init_image, '1 ... -> b ...', b=batch_size)
                    init_latent = self.modelFS.get_first_stage_encoding(
                        self.modelFS.encode_first_stage(init_image)).to(
                        self.device)  # move to latent space

                    x0 = self.model.stochastic_encode(
                        init_latent,
                        torch.tensor(
                            [steps] * batch_size).to(self.device),
                        seed,
                        ddim_eta,
                        ddim_steps,
                    )
                if len(mask_b64) > 0 and init_image is not None:
                    if mask_b64[:4] == 'data':
                        print("Loading mask from b64")
                        mask_image = support.b64_to_image(mask_b64).convert('L')
                    else:
                        mask_image = Image.open(mask_b64).convert("L")
                    if invert:
                        mask_image = ImageOps.invert(mask_image)

                    mask = load_mask(mask_image, init_latent.shape[2], init_latent.shape[3]).to(self.device)
                    mask = mask[0][0].unsqueeze(0).repeat(4, 1, 1).unsqueeze(0)
                    mask = repeat(mask, '1 ... -> b ...', b=batch_size)
                    if self.v1:
                        if self.model.model1.diffusion_model.input_blocks[0][0].weight.shape[1] == 9:
                            init_latent = torch.cat(
                                (init_latent, x0, mask[:, :1, :, :]), dim=1)  # yeah basically
                    x_T = init_latent
                else:
                    mask_image = None
                    mask = None
                    x_T = None

                x0 = self.model.sample(
                    S=steps,
                    conditioning=c,
                    x0=(x0 if init_image is None or "ddim" in sampler.lower(
                    ) else init_latent),
                    S_ddim_steps=ddim_steps,
                    unconditional_guidance_scale=cfg_scale,
                    unconditional_conditioning=uc,
                    eta=ddim_eta,
                    sampler=sampler,
                    shape=shape,
                    batch_size=batch_size,
                    seed=seed,
                    mask=mask,
                    x_T=x_T
                )
                if self.v1:
                    self.modelFS.to(self.device)

                x_samples_ddim = self.modelFS.decode_first_stage(
                    x0[0].unsqueeze(0))
                x_sample = torch.clamp(
                    (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                # print(x_sample.shape)  # 1, 3, 768, 768
                if abs(float(x_sample.flatten().sum())) <= 1.0:  # black square
                    self.modelFS.to(torch.float32)
                    x_samples_ddim = self.modelFS.decode_first_stage(
                        x0[0].to(torch.float32).unsqueeze(0))
                    x_sample = torch.clamp(
                        (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).half()
                    self.modelFS.half()
                x_sample = 255. * \
                            rearrange(
                                x_sample[0].cpu().numpy(), 'c h w -> h w c')
                out_image = Image.fromarray(x_sample.astype(np.uint8))
                return out_image

    def generate(self, text_prompts="", negative_prompts="", batch_name="", init_image_str="", mask_b64="",
                 invert=False,
                 steps=50, H=512, W=512, strength=0.75, cfg_scale=7.5, seed=-1, sampler="ddim", C=4, ddim_eta=0.0, f=8,
                 n_iter=4, batch_size=1, ckpt="", vae="", image_save_path="", speed="High", skip_grid=False):


        self.highres_fix = False
        self.running = True
        print("Starting generate process...")

        self.latest_images_part1 = self.latest_images_part2
        self.latest_images_part2 = []

        torch.cuda.empty_cache()
        gc.collect()
        seed_everything(seed)

        if len(init_image_str) > 0 and sampler == 'plms' or len(mask_b64) > 0:
            if len(mask_b64) > 0:
                print("Currently, only DDIM works with masks. Switching samplers to DDIM")
            sampler = 'ddim'

        self.image_save_path = image_save_path
        ddim_steps = steps

        print("Setting up models...")
        self.load_ckpt(ckpt, speed, vae)
        if not self.model:
            print("Setting up model failed")
            return 'Failure'

        print("Generating...")
        self.stage = "Generating"
        outdir = os.path.join(self.image_save_path, batch_name)
        os.makedirs(outdir, exist_ok=True)

        starting_image = self.get_image(init_image_str, mask_b64)  
        
        print("Prompt:", text_prompts)
        prompts_data = [batch_size * text_prompts]
        print("Negative Prompt:", negative_prompts)
        negative_prompts_data = [batch_size * negative_prompts]

        if self.long_save_path:
            sample_path = os.path.join(outdir, re.sub(
                r'\W+', '', "_".join(text_prompts.split())))[:150]
        else:
            sample_path = outdir

        os.makedirs(sample_path, exist_ok=True)
        base_count = len(os.listdir(sample_path))

        if starting_image is not None:
            if self.v1:
                self.modelFS.to(self.device)

            steps = int(strength * steps)
            if steps <= 0:
                steps = 1

        if self.is_nvidia:
            precision_scope = autocast
        else:
            precision_scope = nullcontext

        self.total_num = n_iter
        all_samples = []

        with torch.no_grad():
            for n in range(n_iter):
                if not self.running:
                    self.clean_up()
                    return

                if self.long_save_path:
                    save_name = f"{base_count:05}_seed_{str(seed)}.jpg"
                else:
                    prompt_name = re.sub(
                        r'\W+', '', '_'.join(text_prompts.split()))[:100]
                    save_name = f"{base_count:05}_{prompt_name}_seed_{str(seed)}.jpg"

                self.current_num = n
                self.model.current_step = 0
                self.model.total_steps = steps
            
                if self.highres_fix: 
                    if min(W, H) > 512:
                        scale = min(W, H) / 512
                        print(f"Hires Scale: {scale}")
                        W, H = (int(W/scale), int(H/scale))
                    
                    out_image = self.generate_image(prompts_data, negative_prompts_data, precision_scope, starting_image, mask_b64, invert, steps, H, W, cfg_scale, seed, sampler, C, ddim_eta, f, ddim_steps)
                
                    out_image.convert("RGB").save("TEST_FIRST.jpg")

                    starting_image = self.Upscaler.upscale(images = ["C:/Users/artad/Documents/GitHub/ArtroomAI/artroom-frontend/TEST_FIRST.jpg"], upscaler="RealESRGAN", upscale_factor=scale, upscale_dest=os.path.join("C:/Users/artad/Documents/GitHub/ArtroomAI/artroom-frontend/"))["content"]["output_images"][0].convert("RGB")
                    starting_image.save("TEST_UPSCALE.jpg")

                    out_image = self.generate_image(prompts_data, negative_prompts_data, precision_scope, starting_image, mask_b64, invert, steps, starting_image.size[1], starting_image.size[0], cfg_scale, seed, sampler, C, ddim_eta, f, ddim_steps)
                    out_image.convert("RGB").save("TEST_FINAL.jpg")
                else:
                    out_image = self.generate_image(prompts_data, negative_prompts_data, precision_scope, starting_image, mask_b64, invert, steps, H, W, cfg_scale, seed, sampler, C, ddim_eta, f, ddim_steps)
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
                out_image.save(
                    os.path.join(sample_path, save_name), "JPEG", exif=exif_data)
                self.latest_images_part2.append(
                    {"b64": support.image_to_b64(out_image), "path": os.path.join(sample_path, save_name)})

                self.socketio.emit('message', {'data': 'testInside'})
                while True:
                    newrand = random.randint(1, 922337203685)
                    if newrand != self.latest_images_id:
                        self.latest_images_id = newrand
                        break

                base_count += 1
                seed += 1
                if not skip_grid and n_iter > 1:
                    all_samples.append(out_image)

            if not skip_grid and n_iter > 1:
                # additionally, save as grid
                rows = int(np.sqrt(len(all_samples)))
                cols = int(np.ceil(len(all_samples) / rows))
                os.makedirs(sample_path + "/grids", exist_ok=True)
                image_grid(all_samples, rows, cols, path=os.path.join(
                    sample_path + "/grids", f'grid-{len(os.listdir(sample_path + "/grids")):04}.jpg'))
        self.clean_up()
