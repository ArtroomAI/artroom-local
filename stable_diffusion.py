from safe import load as safe_load
import warnings
from artroom_helpers.prompt_parsing import weights_handling, split_weighted_subprompts
from skimage import exposure
import cv2
import random
from io import BytesIO
import base64
from transformers import logging
from tqdm import tqdm, trange
from torch import autocast
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from itertools import islice
from einops import rearrange, repeat
from contextlib import nullcontext
from PIL import Image
import torch
import gc
import re
import time
import numpy as np
import json
import math
import os
import sys

sys.path.append("stable-diffusion/optimizedSD/")
from ldm.util import instantiate_from_config


logging.set_verbosity_error()

warnings.filterwarnings("ignore", category=DeprecationWarning)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def setup_color_correction(image):
    correction_target = cv2.cvtColor(
        np.asarray(image.copy()), cv2.COLOR_RGB2LAB)
    return correction_target


def apply_color_correction(correction, image):
    image = Image.fromarray(cv2.cvtColor(exposure.match_histograms(
        cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2LAB), correction, channel_axis=2
    ), cv2.COLOR_LAB2RGB).astype("uint8"))

    return image


def load_model_from_config(ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = safe_load(ckpt)
    # pl_sd = torch.load(ckpt, map_location="cpu")
    # if "global_step" in pl_sd:
    #     print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    return sd


def load_img(image, h0, w0):
    w, h = image.size
    if h0 != 0 and w0 != 0:
        h, w = h0, w0

    # resize to integer multiple of 32
    w, h = map(lambda x: x - x % 64, (w, h))

    print(f"New image size ({w}, {h})")
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.


def load_mask(mask, h0, w0, newH, newW, invert=False):
    image = np.array(mask)
    if invert:
        image = np.clip(image, 254, 255) + 1
    else:
        image = np.clip(image + 1, 0, 1) - 1
    image = Image.fromarray(image).convert("RGB")
    w, h = image.size
    print(f"loaded input mask of size ({w}, {h})")
    if h0 is not None and w0 is not None:
        h, w = h0, w0

    # resize to integer multiple of 32
    w, h = map(lambda x: x - x % 64, (w, h))

    print(f"New mask size ({w}, {h})")
    image = image.resize((newW, newH), resample=Image.LANCZOS)
    # image = image.resize((64, 64), resample=Image.LANCZOS)
    image = np.array(image)

    # if invert:
    #     print("inverted")
    #     where_0, where_1 = np.where(image == 0), np.where(image == 255)
    #     image[where_0], image[where_1] = 255, 0
    image = image.astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image


def image_to_b64(image):
    image_file = BytesIO()
    image.save(image_file, format='JPEG')
    im_bytes = image_file.getvalue()  # im_bytes: image in binary format.
    imgb64 = base64.b64encode(im_bytes)
    return 'data:image/jpeg;base64,' + str(imgb64)[2:-1]


def b64_to_image(b64):
    image_data = re.sub('^data:image/.+;base64,', '', b64)
    return Image.open(BytesIO(base64.b64decode(image_data))).convert('RGB')


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
    def __init__(self):
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
        self.image_save_path = os.environ['USERPROFILE'] + '/Desktop/'
        self.long_save_path = False
        self.highres_fix = False

        self.device = "cuda"
        self.precision = "autocast"
        self.speed = "High"

    def set_artroom_path(self, path):
        print("Setting up artroom path")
        self.artroom_path = path
        # First load of ckpt
        loaded = False
        if os.path.exists(f"{self.artroom_path}/artroom/settings/sd_settings.json"):
            print("Loading model from sd_settings.json")
            sd_settings = json.load(
                open(f"{self.artroom_path}/artroom/settings/sd_settings.json"))
            self.image_save_path = sd_settings['image_save_path']
            self.long_save_path = sd_settings['long_save_path']
            self.highres_fix = sd_settings['highres_fix']

            model_ckpt = sd_settings['ckpt_dir'] + \
                '/' + os.path.basename(sd_settings['ckpt'])
            model_ckpt = model_ckpt.replace(os.sep, '/')
            speed = sd_settings['speed']
            precision = sd_settings['precision']

            if os.path.exists(model_ckpt):
                loaded = self.load_ckpt(model_ckpt, speed, precision)
                if loaded:
                    print("Model successfully loaded")
                else:
                    print("Failed to load model from sd_settings.json")

        if not loaded:
            print("Loading default model form artroom path...")
            if os.path.exists(f"{self.artroom_path}/artroom/model_weights/model.ckpt"):
                loaded = self.load_ckpt(
                    f"{self.artroom_path}/artroom/model_weights/model.ckpt", self.speed, self.precision)
                if loaded:
                    print("Loaded default model from artroom path")
                else:
                    print("Failed to load model from artroom path")

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
            return image_to_b64(Image.open(latest_images[-1]).convert('RGB'))
        else:
            return ''

    def clean_up(self):
        self.total_num = 0
        self.current_num = 0
        if self.model:
            self.model.current_step = 0
            self.model.total_steps = 0

        self.stage = ""
        self.running = False
        if self.device != "cpu":
            mem = torch.cuda.memory_allocated() / 1e6
            self.modelFS.to("cpu")
            while torch.cuda.memory_allocated() / 1e6 >= mem:
                time.sleep(1)

    def loaded_models(self):
        return self.model is not None

    def load_ckpt(self, ckpt, speed, precision):
        print(
            f"Attempting to load {ckpt}, speed: {speed}, precision: {precision}, device: {self.device}")
        assert ckpt != '', 'Checkpoint cannot be empty'
        if self.ckpt != ckpt or self.speed != speed or self.precision != precision:
            try:
                print("Setting up model...")
                self.set_up_models(ckpt, speed, precision)
                print("Successfully set up model")
                return True
            except Exception as e:
                print(f"Setting up model failed: {e}")
                self.stage = ""
                self.model = None
                self.modelCS = None
                self.modelFS = None
                return False

    def set_up_models(self, ckpt, speed, precision):
        print("Loading in model...")
        self.stage = "Loading Model"
        print("Loading model from config")
        sd = load_model_from_config(f"{ckpt}")
        if sd:
            print("Model safety check passed")
        else:
            print("Model safety check died midways")
            return
        print("Setting up config...")
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
            if speed == 'Low':
                self.config = 'stable-diffusion/optimizedSD/configs/v1-inference_lowvram.yaml'
            elif speed == 'Medium':
                self.config = 'stable-diffusion/optimizedSD/configs/v1-inference_lowvram.yaml'
            elif speed == 'High':
                self.config = 'stable-diffusion/optimizedSD/configs/v1-inference.yaml'
            elif speed == 'Max':
                self.config = 'stable-diffusion/optimizedSD/configs/v1-inference_xformer.yaml'
            else:
                print(f"Not recognized speed: {speed}")
                self.config = 'stable-diffusion/optimizedSD/configs/v1-inference.yaml'
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
        del sd
        if self.device != "cpu" and precision == "autocast":
            self.model.half()
            self.modelCS.half()
            self.modelFS.half()
            torch.set_default_tensor_type(torch.HalfTensor)

        self.ckpt = ckpt.replace(os.sep, '/')
        self.speed = speed
        self.precision = precision
        self.stage = "Finished Loading Model"
        print("Model loading finished")

    def generate(self, text_prompts="", negative_prompts="", batch_name="",          init_image_str="", mask_b64="",
                 invert=False,
                 steps=50, H=512, W=512, strength=0.75, cfg_scale=7.5, seed=-1, sampler="ddim", C=4, ddim_eta=0.0, f=8,
                 n_iter=4, batch_size=1, ckpt="", image_save_path="", speed="High", device='cuda', precision='autocast',
                 skip_grid=False):

        self.running = True

        oldW, oldH = W, H
        if W * H > 1024 * 1024 and self.highres_fix:
            highres_fix_steps = math.ceil((W * H) / (1024 * 1024))
            W, H = W // highres_fix_steps, H // highres_fix_steps
            W = math.floor(W / 64) * 64
            H = math.floor(H / 64) * 64
            print(f"Using a highres fix of {highres_fix_steps}")
        else:
            highres_fix_steps = 1

        print("Starting generate process...")

        self.latest_images_part1 = self.latest_images_part2
        self.latest_images_part2 = []

        torch.cuda.empty_cache()
        gc.collect()
        seed_everything(seed)

        if len(init_image_str) > 0 and sampler == 'plms':
            sampler = 'ddim'

        self.image_save_path = image_save_path
        ddim_steps = steps

        self.device = device

        print("Setting up models...")
        self.load_ckpt(ckpt, speed, precision)
        if not self.model:
            print("Setting up model failed")
            return 'Failure'

        print("Generating...")
        self.stage = "Generating"
        outdir = self.image_save_path + batch_name
        os.makedirs(outdir, exist_ok=True)

        if len(init_image_str) > 0:
            if init_image_str[:4] == 'data':
                print("Loading image from b64")
                image = b64_to_image(init_image_str).convert('RGB')
            else:
                print(f"Loading from path {init_image_str}")
                image = Image.open(init_image_str).convert('RGB')
            init_image = load_img(image, H, W).to(self.device)
            _, _, H, W = init_image.shape
            if self.device != "cpu" and self.precision == "autocast":
                init_image = init_image.half()
        else:
            init_image = None

        print("Prompt:", text_prompts)
        data = [batch_size * text_prompts]
        print("Negative Prompt:", negative_prompts)
        negative_prompts_data = [batch_size * negative_prompts]

        if self.long_save_path:
            sample_path = os.path.join(outdir, re.sub(
                r'\W+', '', "_".join(text_prompts.split())))[:150]
        else:
            sample_path = outdir

        os.makedirs(sample_path, exist_ok=True)
        base_count = len(os.listdir(sample_path))

        if init_image is not None:
            self.modelFS.to(self.device)
            steps = int(strength * steps)
            if steps <= 0:
                steps = 1

        if self.precision == "autocast" and self.device != "cpu":
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
                    save_name = f"{base_count:05}_seed_{str(seed)}.png"
                else:
                    prompt_name = re.sub(
                        r'\W+', '', '_'.join(text_prompts.split()))[:100]
                    save_name = f"{base_count:05}_{prompt_name}_seed_{str(seed)}.png"

                self.current_num = n
                self.model.current_step = 0
                self.model.total_steps = steps
                for prompts in data:
                    with precision_scope(self.device):
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
                        for ij in range(1, highres_fix_steps + 1):
                            if init_image is not None:
                                init_image = init_image.to(self.device)
                                if self.device != "cpu" and self.precision == "autocast":
                                    init_image = init_image.half()
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
                                    mask = b64_to_image(mask_b64).convert('L')
                                else:
                                    mask = Image.open(mask).convert("L")
                                mask = load_mask(mask, H, W, init_latent.shape[2], init_latent.shape[3], invert).to(
                                    self.device)
                                mask = mask[0][0].unsqueeze(
                                    0).repeat(4, 1, 1).unsqueeze(0)
                                mask = repeat(
                                    mask, '1 ... -> b ...', b=batch_size)
                                if self.model.model1.diffusion_model.input_blocks[0][0].weight.shape[1] == 9:
                                    init_latent = torch.cat(
                                        (init_latent, x0, mask[:, :1, :, :]), dim=1)  # yeah basically
                                x_T = init_latent
                                color_correction = setup_color_correction(
                                    image)
                            else:
                                mask = None
                                x_T = None
                                color_correction = None
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
                            self.modelFS.to(self.device)
                            x_samples_ddim = self.modelFS.decode_first_stage(
                                x0[0].unsqueeze(0))
                            x_sample = torch.clamp(
                                (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            x_sample = 255. * \
                                rearrange(
                                    x_sample[0].cpu().numpy(), 'c h w -> h w c')
                            out_image = Image.fromarray(
                                x_sample.astype(np.uint8))
                            if ij < highres_fix_steps - 1:
                                init_image = load_img(
                                    out_image, H * (ij + 1), W * (ij + 1))
                                if self.device != "cpu" and self.precision == "autocast":
                                    init_image = init_image.half()
                            elif ij == highres_fix_steps - 1:
                                init_image = load_img(out_image, oldH, oldW)
                                if self.device != "cpu" and self.precision == "autocast":
                                    init_image = init_image.half()

                        if mask is not None:
                            out_image = apply_color_correction(
                                color_correction, out_image)
                        out_image.save(
                            os.path.join(sample_path, save_name))
                        self.latest_images_part2.append(out_image)
                        self.latest_images_id = random.randint(1, 922337203685)

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
                    sample_path + "/grids", f'grid-{len(os.listdir(sample_path + "/grids")):04}.png'))
        self.clean_up()
