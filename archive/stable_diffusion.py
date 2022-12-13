import argparse
import json
import os
import re
import time
import traceback
from contextlib import nullcontext
from itertools import islice

import numpy as np
import torch
from PIL import Image
from einops import rearrange, repeat
from omegaconf import OmegaConf
from optimUtils import split_weighted_subprompts
from pytorch_lightning import seed_everything
from torch import autocast
from tqdm import tqdm, trange
from transformers import logging

from handle_errs import process_error_trace
from ldm.util import instantiate_from_config

logging.set_verbosity_error()
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(ckpt, verbose=False):
    # print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    # if "global_step" in pl_sd:
    #     print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    return sd


def load_img(path, h0, w0):
    image = Image.open(path).convert("RGB")
    w, h = image.size

    print(f"loaded input image of size ({w}, {h}) from {path}")
    if (h0 != 0 and w0 != 0):
        h, w = h0, w0

    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 32

    print(f"New image size ({w}, {h})")
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.


def load_mask(mask, h0, w0, newH, newW, invert=False):
    image = Image.open(mask).convert("L")
    image = np.array(image)
    if invert:
        image = np.clip(image, 254, 255) + 1
    else:
        image = np.clip(image + 1, 0, 1) - 1
    image = Image.fromarray(image).convert("RGB")
    w, h = image.size
    print(f"loaded input mask of size ({w}, {h})")
    if h0 is not None and w0 is not None:
        h, w = h0, w0

    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 32

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


def image_grid(imgs, rows, cols, path):
    assert len(imgs) <= rows * cols
    imgs = [Image.fromarray(img) for img in imgs]
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    grid.save(path)


parser = argparse.ArgumentParser()

parser.add_argument(
    "--prompt",
    type=str,
    nargs="?",
    default="",
    help="the prompt to render"
)
parser.add_argument(
    "--negative_prompt",
    type=str,
    nargs="?",
    default="",
    help="the prompt to render"
)
parser.add_argument(
    "--outdir",
    type=str,
    nargs="?",
    help="dir to write results to",
    default="outputs/img2img-samples"
)
parser.add_argument(
    "--skip_grid",
    action='store_true',
    help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
)
parser.add_argument(
    "--skip_save",
    action='store_true',
    help="do not save individual samples. For speed measurements.",
)
parser.add_argument(
    "--ddim_steps",
    type=int,
    default=50,
    help="number of ddim sampling steps",
)
parser.add_argument(
    "--ddim_eta",
    type=float,
    default=0.0,
    help="ddim eta (eta=0.0 corresponds to deterministic sampling",
)
parser.add_argument(
    "--n_iter",
    type=int,
    default=1,
    help="sample this often",
)
parser.add_argument(
    "--H",
    type=int,
    default=None,
    help="image height, in pixel space",
)
parser.add_argument(
    "--W",
    type=int,
    default=None,
    help="image width, in pixel space",
)
parser.add_argument(
    "--strength",
    type=float,
    default=0.75,
    help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
)
parser.add_argument(
    "--C",
    type=int,
    default=4,
    help="latent channels",
)
parser.add_argument(
    "--f",
    type=int,
    default=8,
    help="downsampling factor",
)
parser.add_argument(
    "--n_samples",
    type=int,
    default=5,
    help="how many samples to produce for each given prompt. A.k.a. batch size",
)
parser.add_argument(
    "--n_rows",
    type=int,
    default=0,
    help="rows in the grid (default: n_samples)",
)
parser.add_argument(
    "--scale",
    type=float,
    default=7.5,
    help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="the seed (for reproducible sampling)",
)
parser.add_argument(
    "--small_batch",
    action='store_true',
    help="Reduce inference time when generate a smaller batch of images",
)
parser.add_argument(
    "--precision",
    type=str,
    help="evaluate at this precision",
    choices=["full", "autocast"],
    default="autocast"
)
parser.add_argument(
    "--ckpt",
    type=str,
    help="path to checkpoint of model",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="CPU or GPU (cuda/cuda:0/cuda:1/...)",
)
parser.add_argument(
    "--unet_bs",
    type=int,
    default=1,
    help="Slightly reduces inference time at the expense of high VRAM (value > 1 not recommended )",
)
parser.add_argument(
    "--sampler",
    type=str,
    help="sampler",
    choices=["ddim", "plms", "lms", "euler", "euler_a", "dpm", "dpm_a", "heun"],
    default="ddim",
)
parser.add_argument(
    "--turbo",
    action="store_true",
    help="Reduces inference time on the expense of 1GB VRAM",
)
parser.add_argument(
    "--superfast",
    action="store_true",
    help="Reduces inference time on the expense of 1GB VRAM",
)
parser.add_argument(
    "--init_image",
    type=str,
    nargs="?",
    default="",
    help="path to the input image"
)
parser.add_argument(
    "--mask",
    type=str,
    nargs="?",
    default="",
    help="path to the input mask"
)
parser.add_argument(
    "--invert",
    action="store_true",
    help="Invert mask",
)

opt = parser.parse_args()
userprofile = os.environ['USERPROFILE']
queue_json = json.load(open(f'{userprofile}/artroom/settings/queue.json'))
settings = queue_json["Queue"][0]
delay = queue_json["Delay"]
opt.precision = settings['precision']
opt.ckpt = settings['ckpt']
opt.device = settings['device']
opt.turbo = settings['turbo']
opt.superfast = settings['superfast']
opt.seed = settings['seed']

ckpt = opt.ckpt
if opt.superfast:
    config = "optimizedSD/v1-inference.yaml"
else:
    config = "optimizedSD/v1-inference_lowvram.yaml"

print("Seed: ", opt.seed)
seed_everything(opt.seed)

sd = load_model_from_config(f"{ckpt}")
li = []
lo = []
for key, value in sd.items():
    sp = key.split('.')
    if (sp[0]) == 'model':
        if ('input_blocks' in sp):
            li.append(key)
        elif ('middle_block' in sp):
            li.append(key)
        elif ('time_embed' in sp):
            li.append(key)
        else:
            lo.append(key)
for key in li:
    sd['model1.' + key[6:]] = sd.pop(key)
for key in lo:
    sd['model2.' + key[6:]] = sd.pop(key)

config = OmegaConf.load(f"{config}")
model = instantiate_from_config(config.modelUNet)
_, _ = model.load_state_dict(sd, strict=False)
model.eval()
model.cdevice = opt.device
model.unet_bs = opt.unet_bs
model.turbo = opt.turbo

modelCS = instantiate_from_config(config.modelCondStage)
_, _ = modelCS.load_state_dict(sd, strict=False)
modelCS.eval()
modelCS.cond_stage_model.device = opt.device

modelFS = instantiate_from_config(config.modelFirstStage)
_, _ = modelFS.load_state_dict(sd, strict=False)
modelFS.eval()
del sd
if opt.device != "cpu" and opt.precision == "autocast":
    model.half()
    modelCS.half()
    modelFS.half()
    torch.set_default_tensor_type(torch.HalfTensor)

while (True):
    try:
        tic = time.time()
        queue_json = json.load(open(f'{userprofile}/artroom/settings/queue.json'))
        settings = queue_json["Queue"][0]
        delay = queue_json["Delay"]
        opt.prompt = settings['prompt']
        opt.negative_prompt = settings['negative_prompt']
        opt.outdir = settings['outdir']
        opt.skip_grid = settings['skip_grid']
        opt.ddim_steps = settings['ddim_steps']
        opt.n_samples = settings['n_samples']
        opt.n_iter = settings['n_iter']
        opt.H = settings['H']
        opt.W = settings['W']
        opt.scale = settings['scale']
        opt.seed = settings['seed']
        opt.sampler = settings['sampler']
        opt.init_image = settings['init_image']
        opt.strength = settings['strength']
        opt.mask = settings['mask']
        opt.invert = settings['invert']
        outpath = opt.outdir

        os.makedirs(opt.outdir, exist_ok=True)

        if len(opt.init_image) > 0:
            assert os.path.isfile(opt.init_image)
            init_image = load_img(opt.init_image, opt.H, opt.W).to(opt.device)
            _, _, H, W = init_image.shape
            if opt.device != "cpu" and opt.precision == "autocast":
                init_image = init_image.half()
        else:
            init_image = None
            H = opt.H
            W = opt.W

        batch_size = opt.n_samples
        n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
        print("Prompt:", opt.prompt)
        data = [batch_size * [opt.prompt]]
        print("Negative Prompt:", opt.negative_prompt)
        negative_prompt_data = [batch_size * opt.negative_prompt]

        sample_path = os.path.join(outpath, re.sub(r'\W+', '', "_".join(opt.prompt.split())))[:150]
        os.makedirs(sample_path, exist_ok=True)
        base_count = len(os.listdir(sample_path))

        modelFS.to(opt.device)
        if init_image is not None:
            init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
            init_latent = modelFS.get_first_stage_encoding(
                modelFS.encode_first_stage(init_image))  # move to latent space
            assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
            steps = int(opt.strength * opt.ddim_steps)
        else:
            steps = opt.ddim_steps

        if len(opt.mask) > 0:
            mask = load_mask(opt.mask, H, W, init_latent.shape[2], init_latent.shape[3], opt.invert).to(opt.device)
            mask = mask[0][0].unsqueeze(0).repeat(4, 1, 1).unsqueeze(0)
            mask = repeat(mask, '1 ... -> b ...', b=batch_size)
            x_T = init_latent
        else:
            mask = None
            x_T = None

        if opt.device != "cpu":
            mem = torch.cuda.memory_allocated() / 1e6
            modelFS.to("cpu")
            while (torch.cuda.memory_allocated() / 1e6 >= mem):
                time.sleep(1)

        if opt.precision == "autocast" and opt.device != "cpu":
            precision_scope = autocast
        else:
            precision_scope = nullcontext

        with torch.no_grad():
            all_samples = list()
            for n in trange(opt.n_iter, desc="Sampling"):
                for prompts in tqdm(data, desc="data"):
                    with precision_scope("cuda"):
                        modelCS.to(opt.device)
                        uc = None
                        if opt.scale != 1.0:
                            uc = modelCS.get_learned_conditioning(negative_prompt_data)
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)

                        subprompts, weights = split_weighted_subprompts(prompts[0])
                        if len(subprompts) > 1:
                            c = torch.zeros_like(uc)
                            totalWeight = sum(weights)
                            # normalize each "sub prompt" and add it
                            for i in range(len(subprompts)):
                                weight = weights[i]
                                # if not skip_normalize:
                                weight = weight / totalWeight
                                c = torch.add(c, modelCS.get_learned_conditioning(subprompts[i]), alpha=weight)
                        else:
                            c = modelCS.get_learned_conditioning(prompts)

                        shape = [opt.n_samples, opt.C, H // opt.f, W // opt.f]
                        if opt.device != "cpu":
                            mem = torch.cuda.memory_allocated() / 1e6
                            modelCS.to("cpu")
                            while (torch.cuda.memory_allocated() / 1e6 >= mem):
                                time.sleep(1)
                        if init_image is not None:
                            # encode (scaled latent)
                            if opt.sampler == "ddim":
                                x0 = model.stochastic_encode(
                                    init_latent,
                                    torch.tensor([steps] * batch_size).to(opt.device),
                                    opt.seed,
                                    opt.ddim_eta,
                                    opt.ddim_steps,
                                )
                            else:
                                x0 = init_latent
                        else:
                            x0 = None
                        # decode it
                        samples_ddim = model.sample(
                            S=steps,
                            conditioning=c,
                            x0=x0,
                            unconditional_guidance_scale=opt.scale,
                            unconditional_conditioning=uc,
                            eta=opt.ddim_eta,
                            sampler=opt.sampler,
                            shape=shape,
                            batch_size=batch_size,
                            seed=opt.seed,
                            mask=mask,
                            x_T=x_T
                        )

                        modelFS.to(opt.device)
                        for i in range(batch_size):
                            x_samples_ddim = modelFS.decode_first_stage(samples_ddim[i].unsqueeze(0))
                            x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            x_sample = 255. * rearrange(x_sample[0].cpu().numpy(), 'c h w -> h w c')
                            Image.fromarray(x_sample.astype(np.uint8)).save(
                                os.path.join(sample_path, "seed_" + str(opt.seed) + "_" + f"{base_count:05}.png"))
                            Image.fromarray(x_sample.astype(np.uint8)).save(
                                os.path.join(sample_path, "latest.png"))
                            base_count += 1
                            opt.seed += 1
                            if not opt.skip_grid:
                                all_samples.append(x_sample.astype(np.uint8))

                        if opt.device != "cpu":
                            mem = torch.cuda.memory_allocated() / 1e6
                            modelFS.to("cpu")
                            while (torch.cuda.memory_allocated() / 1e6 >= mem):
                                time.sleep(1)

                        del samples_ddim
                        # print("memory_final = ", torch.cuda.memory_allocated()/1e6)

            if not opt.skip_grid:
                # additionally, save as grid
                # grid = torch.from_numpy(np.stack(all_samples,0))
                # grid = torch.stack(all_samples,0)
                rows = int(np.sqrt(len(all_samples)))
                cols = int(np.ceil(len(all_samples) / rows))
                os.makedirs(sample_path + "/grids", exist_ok=True)
                image_grid(all_samples, rows, cols, path=os.path.join(sample_path + "/grids",
                                                                      f'grid-{len(os.listdir(sample_path + "/grids")):04}.png'))

    except Exception as err:
        queue_json = json.load(open(f'{userprofile}/artroom/settings/queue.json'))
        queue_json["Queue"] = queue_json["Queue"][1:]
        queue_json["Running"] = False
        with open(f'{userprofile}/artroom/settings/queue.json', 'w') as outfile:
            json.dump(queue_json, outfile, indent=4)
        process_error_trace(traceback.format_exc(), err, opt.prompt, outpath)

    toc = time.time()

    time_taken = (toc - tic) / 60.0

    print(("Your samples are ready in {0:.2f} minutes and waiting for you here \n" + sample_path).format(time_taken))
    queue_json["Queue"] = queue_json["Queue"][1:]
    with open(f'{userprofile}/artroom/settings/queue.json', 'w') as outfile:
        json.dump(queue_json, outfile, indent=4)

    while (True):
        queue_json = json.load(open(f'{userprofile}/artroom/settings/queue.json'))
        keep_warm = queue_json['Keep_Warm']
        if not keep_warm:
            break
        if len(queue_json["Queue"]) == 0:
            time.sleep(5)
        else:
            try:
                time.sleep(int(delay))
            except:
                time.sleep(5)
            break

    if not keep_warm:
        break

queue_json = json.load(open(f'{userprofile}/artroom/settings/queue.json'))
queue_json["Running"] = False
with open(f'{userprofile}/artroom/settings/queue.json', 'w') as outfile:
    json.dump(queue_json, outfile, indent=4)
