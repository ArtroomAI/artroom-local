from functools import partial
import io
import re
import threading
import warnings
import gc
import sys
from einops import rearrange
from glob import glob

sys.path.append("backend/ComfyUI/")
from artroom_helpers.gpu_detect import get_device, get_gpu_architecture
from transformers import logging as tflogging
from upscale import Upscaler

from backend.ComfyUI.nodes import *
from backend.ComfyUI.comfy.cli_args import args
from backend.ComfyUI.comfy.sd import model_lora_keys_unet, model_lora_keys_clip, load_lora
from backend.ComfyUI.comfy import utils

from artroom_helpers.generation.preprocess import mask_from_face, mask_background
from artroom_helpers.process_controlnet_images import apply_controlnet, HWC3, apply_inpaint
from artroom_helpers import support, inpainting
from artroom_helpers.toast_status import toast_status

tflogging.set_verbosity_error()
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import traceback


class NodeModules:
    def __init__(self):
        # self.CheckpointLoaderSimple = CheckpointLoaderSimple()  # doesn't work because it has hardcoded paths
        self.CLIPTextEncode = CLIPTextEncode()
        self.LoadImage = LoadImage()
        self.LoadImageMask = LoadImageMask()
        self.KSampler = KSampler()
        self.KSamplerAdvanced = KSamplerAdvanced()
        self.SetLatentNoiseMask = SetLatentNoiseMask()
        self.VAEDecode = VAEDecode()
        self.VAEDecodeTiled = VAEDecodeTiled()
        self.LoraLoader = LoraLoader()
        self.ControlNetLoader = ControlNetLoader()
        self.ControlNetApply = ControlNetApply()
        self.VAELoader = VAELoader()
        self.VAEEncodeForInpaint = VAEEncodeForInpaint()
        self.VAEEncode = VAEEncode()
        self.EmptyLatentImage = EmptyLatentImage()


class Mute:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class Model:
    def __init__(self, ckpt, socketio, device='cuda:0'):
        self.controlnet_path = None
        self.model = None
        self.clip = None
        self.vae = None
        self.clipvision = None
        self.controlnet = None
        self.loras = []

        self.nodes = NodeModules()
        self.device = get_device()

        self.ckpt = ckpt
        self.socketio = socketio

        self.steps = 0
        self.current_num = 0
        self.total_num = 0
        self.total_steps = 0
        if ckpt != '':
            self.load_model()

        # self.prompt_appendices = []

    def load_model(self):
        with Mute():
            self.model, self.clip, self.vae, clipvision = comfy.sd.load_checkpoint_guess_config(
                self.ckpt,
                output_vae=True,
                output_clip=True,
                embedding_directory=folder_paths.get_folder_paths("embeddings"))
        # if hasattr(self.clip.cond_stage_model, "clip_l"):  # sdxl
        #     self.can_use_half_vae = False
        del clipvision

    def load_textual_inversions(self, textual_inversion_list):
        pass
        # for textual_inversion in textual_inversion_list:
        #     textual_inversion = os.path.basename(textual_inversion)
        #     self.prompt_appendices.append(f"embedding:{textual_inversion}")

    def setup_controlnet(self, controlnet_path=None):
        self.controlnet_path = controlnet_path
        try:
            if self.controlnet_path is not None:
                self.inject_controlnet(controlnet_path)
        except Exception as e:
            self.socketio.emit('status', toast_status(title=f"Controlnet failed to load {e}", status="error"))

    def setup_lora(self, loras=None):
        if loras is None:
            loras = []

        if len(loras) > 0:
            for lora in loras:
                if lora not in self.loras:
                    try:
                        self.inject_lora(path=lora['path'], weight_tenc=lora['weight'], weight_unet=lora['weight'])
                    except Exception as e:
                        self.socketio.emit('status',
                                           toast_status(title=f"Failed to load in Lora {lora} {e}", status="error"))

        self.loras = loras

    def get_steps(self):
        if self.model is None:
            return 0, 0, 0, 0
        return self.current_num, self.total_num, self.model.current_step + self.steps * (self.current_num - 1), self.total_steps

    def inject_lora(self, path: str, weight_tenc=1.1, weight_unet=4):
        lora = comfy.utils.load_torch_file(path, safe_load=True)
        key_map = model_lora_keys_unet(self.model.model)
        key_map = model_lora_keys_clip(self.clip.cond_stage_model, key_map)
        loaded = load_lora(lora, key_map)

        self.model.add_patches(loaded, weight_unet)
        self.clip.add_patches(loaded, weight_tenc)

    def deinject_lora(self, delete=True):
        self.model.unpatch_model()
        self.clip.unpatch_model()

    def inject_controlnet(self, controlnet_path, existing=False):
        self.controlnet = comfy.sd.load_controlnet(controlnet_path)

    def deinject_controlnet(self, delete=True):
        del self.controlnet
        self.controlnet = None 

    def load_vae(self, vae_path: str):
        try:
            if os.path.basename(vae_path) != 'none':
                self.vae = comfy.sd.VAE(ckpt_path=vae_path, device=self.device)
        except Exception as e:
            print("Failed to load vae")

    def to(self, device, dtype=torch.float16):
        self.vae.device = device
        self.vae.first_stage_model.to(device).to(torch.float32)

        # self.model.model.to(device).to(dtype)
        # self.model.model_patches_to(device)
        # self.model.model_patches_to(dtype)
        
        # self.clip.device = device
        # self.clip.cond_stage_model.device = device
        # self.clip.patcher.model.to(device).to(dtype)
        # self.clip.cond_stage_model.to(device).to(dtype)

        # if hasattr(self.clip.cond_stage_model, "clip_l"):
        #     self.clip.cond_stage_model.clip_l.device = device
        #     self.clip.cond_stage_model.clip_g.device = device
        #     self.dtype = dtype

        if self.controlnet is not None:
            self.controlnet.control_model.to(device)
            self.controlnet.device = device


class StableDiffusion:
    def __init__(self, socketio=None, Upscaler=None):
        self.ready = False

        self.upscaler = Upscaler
        self.socketio = socketio
        self.nodes = NodeModules()
        self.highres_fix = False
        self.running = False
        self.device = get_device()
        self.gpu_architecture = get_gpu_architecture()  #
        self.active_model = Model(ckpt='', socketio=self.socketio, device=self.device)

    def get_image(self, init_image_str, mask_image, job_id):
        init_image = support.b64_to_image(init_image_str)

        # If init_image is all black, return None. This happens when comes from inpainting
        if np.all(np.array(init_image) == 0):
            return None, None

        if init_image.mode == 'RGBA' and mask_image is None:
            # If it's completely empty, comes from paint tab with nothing there. Send back empty array
            # mask_image = init_image.split()[-1]
            mask_image = None

        if mask_image is not None:
            try:
                init_image = inpainting.infill_patchmatch(init_image)
            except Exception as e:
                print(f"Failed to outpaint the alpha layer {e}")

        # return init_image.convert("RGB"), mask_image
        return init_image.convert("RGB"), mask_image

    def load_image(self, image, h0, w0, mask_image=None, inpainting=False):
        w, h = image.size
        if not inpainting and h0 != 0 and w0 != 0:
            h, w = h0, w0

        w, h = map(lambda x: x - x % 64, (w, h))
        # print(f"New image size ({w}, {h})")
        image = image.resize((w, h), resample=Image.LANCZOS)

        image = np.array(image).astype(np.float32) / 255.0
        init_image = torch.from_numpy(image)[None,]
        init_image = init_image.to(self.device)
        _, H, W, _ = init_image.shape
        # init_image = init_image.half()

        return init_image.to(self.device), H, W

    def load_control_image(self, image, h0, w0, mask_image=None, inpainting=False, controlnet_mode="none", use_preprocessed_controlnet=False, models_dir=""):
        w, h = image.size
        if not inpainting and h0 != 0 and w0 != 0:
            h, w = h0, w0

        w, h = map(lambda x: x - x % 64, (w, h))
        # print(f"New image size ({w}, {h})")
        image = image.resize((w, h), resample=Image.LANCZOS)

        if controlnet_mode in ['scribble']:
            image = support.invert_rgb_mask(image)
        image = HWC3(np.array(image))
        if not use_preprocessed_controlnet:
            if controlnet_mode == 'inpaint' and mask_image:
                control = apply_inpaint(image, mask_image)
            else:
                control = apply_controlnet(image, controlnet_mode, models_dir)
        else:
            control = 1 - torch.from_numpy(image.copy()).float() / 255.0
            control = torch.stack([control for _ in range(1)], dim=0)
            control = rearrange(control, 'b h w c -> b c h w').clone()
        if self.gpu_architecture == 'NVIDIA':
            control = control.cuda()
        return control.to(self.device)

    def diffusion_upscale(
            self,
            job_id=0,
            n=0,
            positive=None,
            negative=None,
            image=None,
            mask_image=None,
            highres_steps=10,
            highres_strength=0.2,
            highres_multiplier=2,
            H=512,
            W=512,
            cfg_scale=7.5,
            seed=-1,
            sampler="ddim",
            mode="default",
            clip_skip=1,
            batch_size=1,
            keep_size=False,
            models_dir="",
            callback=None,
    ):
        # try:
        if keep_size:
            print("Running experimental diffusion touchup")
            original_W, original_H = image.size
            W = int(W * highres_multiplier)
            H = int(H * highres_multiplier)

        original_image = image.copy()

        os.makedirs(os.path.join(models_dir, 'upscale_highres'), exist_ok=True)

        temp_save_path = os.path.join(models_dir, 'upscale_highres', f'diffusion_touchup.png')
        image.save(temp_save_path)

        print("Upscaling image")
        try:
            upscaler_model = "UltraSharp"
            image = self.upscaler.upscale(models_dir=models_dir,
                                          images=[temp_save_path], upscaler=upscaler_model,
                                          upscale_factor=highres_multiplier,
                                          upscale_dest=os.path.join(models_dir, 'upscale_highres'))['content'][
                'output_images'][0]
        except Exception as e:
            print("Failed with UltraSharp. Make sure the UltraSharp pth is correct. Retrying with RealESGRGAN")
            upscaler_model = "RealESRGAN"
            image = self.upscaler.upscale(models_dir=models_dir,
                                          images=[temp_save_path], upscaler=upscaler_model,
                                          upscale_factor=highres_multiplier,
                                          upscale_dest=os.path.join(models_dir, 'upscale_highres'))['content'][
                'output_images'][0]
        image.resize((W, H))  # Ensures final dimensions are correct

        # image.save(f'{job_id}_{n}_upscaled.png')

        # generate the next version using the upscaled image as the new input
        highres_init_image, H, W = self.load_image(image.convert('RGB'), H, W)

        print("Generating upscaled image")
        image = self.generate_image(
            job_id=job_id,
            positive=positive,
            negative=negative,
            init_image=highres_init_image.to(self.device),
            mask_image=mask_image,
            steps=highres_steps,
            H=H,
            W=W,
            cfg_scale=cfg_scale,
            seed=seed,
            sampler=sampler,
            batch_size=batch_size,
            clip_skip=clip_skip,
            callback=callback,
            denoise=highres_strength
        )

        try:
            os.remove(temp_save_path)
            pass
        except Exception as e:
            print('Failed to delete diffusion touchup {e}')

        if mask_image is not None:
            image = support.repaste_and_color_correct(result=image,
                                                      init_image=original_image,
                                                      init_mask=mask_image, mask_blur_radius=8)

        # Used for adding details without changing resolution
        if keep_size:
            image = image.resize((original_W, original_H))
        return image

    @torch.no_grad()
    def generate_image(
            self,
            job_id="",
            positive=None,
            negative=None,
            init_image=None,
            mask_image=None,
            steps=50,
            H=512,
            W=512,
            cfg_scale=7.5,
            seed=-1,
            sampler="ddim",
            batch_size=1,
            device='cuda:0',
            clip_skip=1,
            callback=None,
            denoise=1.0,
            disable_noise=False,
            start_step=None, 
            last_step=None, 
            force_full_denoise=False
    ):

        if mask_image is not None:
            mask_image = np.array(mask_image).astype(np.float32) / 255.0
            mask_image = 1. - torch.from_numpy(mask_image).to(torch.float32).to(init_image.device)
            latent = self.nodes.VAEEncode.encode(self.active_model.vae, init_image)[0]
            latent = self.nodes.SetLatentNoiseMask.set_mask(latent, mask_image)[0]
        elif init_image is not None:
            latent = self.nodes.VAEEncode.encode(self.active_model.vae, init_image)[0]
        else:
            latent = self.nodes.EmptyLatentImage.generate(width=W, height=H, batch_size=batch_size)[0]

        self.active_model.to(device)

        scheduler = "karras"  # ["normal", "karras", "exponential", "simple", "ddim_uniform"]

        latent_image = latent["samples"]

        if disable_noise:
            noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
        else:
            batch_inds = latent["batch_index"] if "batch_index" in latent else None
            noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        samples = comfy.sample.sample(
            self.active_model.model, 
            noise, 
            steps, 
            cfg_scale, 
            sampler, 
            scheduler, 
            positive, 
            negative, 
            latent_image,
            denoise=denoise, 
            disable_noise=disable_noise, 
            start_step=start_step, 
            last_step=last_step,
            force_full_denoise=force_full_denoise, noise_mask=noise_mask, 
            callback=callback, 
            seed=seed)
    
        out = latent.copy()
        out["samples"] = samples.to(torch.float16)

        out = self.nodes.VAEDecode.decode(self.active_model.vae, out)[0]

        out = 255. * out[0].cpu().numpy()
        out = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
        return out

    def generate(self,
                 job_id="",
                 image_save_path="",
                 highres_fix=False,
                 long_save_path=False,
                 show_intermediates=False,
                 use_preprocessed_controlnet=False,
                 remove_background=False,
                 use_removed_background=False,
                 models_dir="",
                 text_prompts="",
                 negative_prompts="",
                 init_image_str="",
                 mask_str="",
                 invert=False,
                 steps=50,
                 H=512,
                 W=512,
                 strength=0.75,
                 cfg_scale=7.5,
                 seed=-1,
                 sampler="ddim",
                 n_iter=4,
                 batch_size=1,
                 ckpt="",
                 vae="",
                 loras=None,
                 controlnet="none",
                 controlnet_strength=1,
                 background_removal_type="none",
                 clip_skip=1,
                 generation_mode='',
                 highres_steps=10,
                 highres_strength=0.1
                 ):

        if long_save_path:
            sample_path = os.path.join(image_save_path, re.sub(
                r'\W+', '', "_".join(text_prompts.split())))[:150]
        else:
            sample_path = image_save_path

        if loras is None:
            loras = []

        if "sd_xl" in ckpt:
            controlnet = "none"
            vae = "none"
            loras = []

        if len(init_image_str) == 0:
            controlnet = 'none'

        if vae is not None and len(vae) > 0:
            vae = os.path.join(models_dir, "Vae", vae)
        else:
            vae = "none"
            
        def get_contronet_path(controlnet):
            controlnet_folder = os.path.join(models_dir, "ControlNet")

            # Search for files with the specified string in their base filename in the controlnet_folder
            matching_files = glob(os.path.join(controlnet_folder, f"*{controlnet}*"))
            if controlnet == 'lineart':
                matching_files = [file for file in matching_files if 'lineart_anime' not in file]

            if matching_files:
                print(f"Found matching controlnet, using {matching_files[0]}")
                return matching_files[0]
            else:
                print(f"Controlnet not found at {controlnet_folder}")
                return None

        if 'none' not in controlnet:
            try:
                controlnet_path = get_contronet_path(controlnet)
            except:
                controlnet_path = None
        else:
            controlnet_path = None

        # Loads from cache
        # use_controlnet = controlnet_path is not None
        # model_key = f'{ckpt}_controlnet_{use_controlnet}'
        model_key = f'{ckpt}'
        self.socketio.emit('status', toast_status(
            id="loading-model", title="Loading model...", status="info",
            position="bottom-right", duration=None, isClosable=False))

        if model_key != self.active_model.ckpt or not support.check_array_dict_equality(self.active_model.loras, loras):
            self.active_model = Model(ckpt, socketio=self.socketio, device=self.device)
            self.active_model.setup_lora(loras)

        self.active_model.load_vae(vae)
        self.active_model.setup_controlnet(controlnet_path)
        self.active_model.to(self.device)

        self.socketio.emit('status', toast_status(
            id="loading-model", title="Finished Loading Model",
            status="info", position="bottom-right", duration=2000))
        print(f"Generating... {job_id} dim: {W}x{H}, Prompt: {text_prompts}")
        self.socketio.emit('status', toast_status(
            id="loading-model", title="Generating",
            status="info", position="bottom-right", duration=2000))

        def resize_dims_balanced(width, height, base=512):
            # Calculate the aspect ratio of the original dimensions
            aspect_ratio = width / height

            # Scale the width and height so their sum is 2048
            new_width = ((base * 2) * aspect_ratio) / (aspect_ratio + 1)
            new_height = ((base * 2) * 1) / (aspect_ratio + 1)

            # Adjust to the nearest multiple of 64
            new_width = round(new_width / 64) * 64
            new_height = round(new_height / 64) * 64
            print(f"New WxH: {new_width}x{new_height}")
            return int(new_width), int(new_height)

        starting_image = None
        init_image = None
        mask_image = None
        control_image = None

        if generation_mode == 'highresfix':
            highres_fix = True

        if highres_fix:
            old_H = H
            old_W = W
            if "sd_xl" in ckpt:
                base = 1024
            else:
                base = 768
            if W > base or H > base:
                W, H = resize_dims_balanced(W, H, base)
            else:
                highres_fix = False 

        if not highres_fix:
            highres_steps = 0

        if len(mask_str) > 0 and len(init_image_str) > 0:
            mask_image = support.b64_to_image(mask_str).convert('L')

        # Get image and extract mask from image if needed
        if len(init_image_str) > 0:
            starting_image, mask_image = self.get_image(init_image_str, mask_image, job_id)

        if background_removal_type != 'none' and starting_image is not None:
            if background_removal_type == 'face':
                mask_image = mask_from_face(starting_image.convert('RGB'), W, H)
            else:
                mask_image = mask_background(starting_image.convert('RGB'),
                                             remove_background=background_removal_type)
        if mask_image is not None:
            mask_image = mask_image.resize((W, H))
            if invert:
                mask_image = ImageOps.invert(mask_image)
        original_mask = mask_image

        if starting_image is not None:
            starting_image = starting_image.resize((W, H))
            init_image, H, W = self.load_image(starting_image, H, W, mask_image=mask_image,
                                               inpainting=(mask_image is not None or background_removal_type != 'none'))
            if controlnet != "none":
                control_image = self.load_control_image(
                    starting_image,
                    H,
                    W,
                    mask_image=mask_image,
                    inpainting=(
                            mask_image is not None or background_removal_type != 'none'),
                    controlnet_mode=controlnet.lower(),
                    use_preprocessed_controlnet=use_preprocessed_controlnet,
                    models_dir=models_dir
                )

        if self.gpu_architecture == 'NVIDIA':
            torch.cuda.empty_cache()
        gc.collect()

        prompts_data = [batch_size * text_prompts]
        negative_prompts_data = [batch_size * negative_prompts]

        self.active_model.to(self.device)

        total_steps = (steps+highres_steps) * n_iter

        self.active_model.model.job_id = job_id
        self.active_model.model.current_step = 0
        self.active_model.steps = steps
        self.active_model.total_steps = total_steps
        self.active_model.total_num = n_iter

        print(f'''Generating on gpu {self.device} image job {job_id} with settings 
            Job Id: {job_id}
            Text Prompts: {text_prompts}, 
            Negative Prompts: {negative_prompts}, 
            Init Image: {init_image_str[:50]}, 
            Mask: {mask_str[:50]}, 
            Invert: {invert}, 
            Steps: {steps}, 
            H: {H}, 
            W: {W}, 
            Strength: {strength}, 
            CFG: {cfg_scale}, 
            Seed: {seed},
            Sampler: {sampler}, 
            N_Iter: {n_iter}, 
            Batch Size: {batch_size}, 
            Ckpt: {ckpt}, 
            Clip Skip: {clip_skip}, 
            Vae: {vae}, 
            Loras: {loras}, 
            Controlnet: {controlnet}''')

        base_count = len(os.listdir(sample_path))
        self.running = True
        if isinstance(prompts_data, list):
            prompts_data = prompts_data[0]

        if isinstance(negative_prompts_data, list):
            negative_prompts_data = negative_prompts_data[0]

        self.active_model.to(self.device)

        positive_cond = self.nodes.CLIPTextEncode.encode(self.active_model.clip, prompts_data)[0]

        negative_prompt = self.nodes.CLIPTextEncode.encode(self.active_model.clip, negative_prompts_data)[0]

        if controlnet is not None and controlnet != "none":
            control_image = control_image.permute(0, 2, 3, 1)
            positive_cond = self.nodes.ControlNetApply.apply_controlnet(
                positive_cond, self.active_model.controlnet, control_image, controlnet_strength)[0]

        preview_format = "JPEG"
        if preview_format not in ["JPEG", "PNG"]:
            preview_format = "JPEG"

        previewer = latent_preview.get_previewer(self.device, self.active_model.model.model.latent_format)

        def callback(step, x0, x, total_steps, current_num, total_num, step_check=5, start_step=0, job_total_steps=0, show_intermediates= False):
            def send_intermediates(x0):
                if previewer and show_intermediates:
                    preview_bytes = previewer.decode_latent_to_preview_image("JPEG", x0)
                    print('PREVIEW BYTES', preview_bytes)
                    self.socketio.emit('intermediate_image', {'b64': support.image_to_b64(Image.open(io.BytesIO(preview_bytes)))})

                self.socketio.emit('get_progress', {
                    'current_step': step+start_step,
                    'total_steps': job_total_steps,
                    'current_num': current_num-1,
                    'total_num': total_num
                })

            if step % step_check == 0:
                threading.Thread(target=send_intermediates, args=(x0,)).start()


        with torch.no_grad():
            for n in range(1, n_iter + 1):
                if not self.running:
                    self.clean_up()
                    return
                self.active_model.current_num = n
                try:
                    print(f'Generating on gpu {self.device} image {n}')
                    if generation_mode != 'highresfix':
                        out_image = self.generate_image(
                            job_id=job_id,
                            positive=positive_cond,
                            negative=negative_prompt,
                            init_image=init_image,
                            mask_image=mask_image,
                            steps=steps,
                            H=H,
                            W=W,
                            cfg_scale=cfg_scale,
                            seed=seed,
                            sampler=sampler,
                            batch_size=batch_size,
                            clip_skip=clip_skip,
                            callback=partial(
                                callback, 
                                current_num=n,
                                total_num=n_iter,
                                start_step=0, 
                                job_total_steps=(steps+highres_steps),
                                show_intermediates=show_intermediates
                                ),
                            denoise=strength if starting_image is not None else 1.0
                        )
                        self.socketio.emit('get_progress', {
                            'current_step': steps,
                            'total_steps': steps+highres_steps,
                            'current_num': n,
                            'total_num': n_iter
                        })
                    else:
                        out_image = starting_image

                    if mask_image is not None:
                        out_image = support.repaste_and_color_correct(
                            result=out_image,
                            init_image=starting_image,
                            init_mask=original_mask,
                            mask_blur_radius=8
                        )

                    # if controlnet == 'none' and mask_image is None:  # Do not apply touchup for inpainting, messes with the rest of the image, unless we decide to pass the mask too. TODO
                    if highres_fix:
                        out_image = self.diffusion_upscale(
                            job_id=job_id,
                            n=n,
                            positive=positive_cond,
                            negative=negative_prompt,
                            image=out_image,
                            mask_image=mask_image,
                            highres_steps=highres_steps,
                            highres_strength=highres_strength,
                            highres_multiplier=min(old_H / H, old_W / W),
                            cfg_scale=cfg_scale,
                            H=old_H,
                            W=old_W,
                            seed=seed,
                            sampler=sampler,
                            clip_skip=clip_skip,
                            keep_size=False,
                            models_dir=models_dir,
                            callback=partial(
                            callback, 
                            current_num=n, 
                            total_num=n_iter,
                            start_step=steps, 
                            step_check=1, 
                            job_total_steps=steps+highres_steps,
                            show_intermediates=show_intermediates)
                        )
                        self.socketio.emit('get_progress', {
                            'current_step': steps+highres_steps,
                            'total_steps': steps+highres_steps,
                            'current_num': n,
                            'total_num': n_iter
                        })
                    exif_data = out_image.getexif()
                    # Does not include Mask, ImageB64, or if Inverted. Only settings for now
                    settings_data = {
                        "text_prompts": text_prompts,
                        "negative_prompts": negative_prompts,
                        "steps": steps,
                        "height": H,
                        "width": W,
                        "strength": strength,
                        "cfg_scale": cfg_scale,
                        "seed": seed,
                        "sampler": sampler,
                        "ckpt": os.path.basename(ckpt),
                        "vae": os.path.basename(vae),
                        "controlnet": controlnet,
                        "loras": [{'name': os.path.basename(lora['path']), 'weight': lora['weight']} for lora in loras],
                        "clip_skip": clip_skip
                    }
                    # 0x9286 Exif Code for UserComment
                    exif_data[0x010E] = "\nArtroom Settings:\n" + json.dumps(settings_data, indent=4) + "\nEND"

                    if long_save_path:
                        save_name = f"{base_count:05}_seed_{str(seed)}.png"
                    else:
                        prompt_name = re.sub(
                            r'\W+', '', '_'.join(text_prompts.split()))[:100]
                        save_name = f"{base_count:05}_{prompt_name}_seed_{str(seed)}.png"

                    out_image.save(
                        os.path.join(sample_path, save_name), "PNG", exif=exif_data)

                    self.socketio.emit('get_images', {'b64': support.image_to_b64(out_image),
                                                      'path': os.path.join(sample_path, save_name),
                                                      'batch_id': job_id})
                except Exception as e:
                    tb = traceback.extract_tb(e.__traceback__)
                    file_name, line_number, _, _ = tb[-1]
                    print(f"Failure in file {file_name} at line {line_number}: {e}")
                    self.socketio.emit('status', toast_status(title=f"Failed to generate image {e}", status="error"))
                seed += 1
                self.clean_up()
                
        self.clean_up()
        self.running = False
        self.active_model.deinject_controlnet()
        self.active_model.deinject_lora()

    def clean_up(self):
        if self.gpu_architecture == 'NVIDIA':
            torch.cuda.empty_cache()
        gc.collect()

    def interrupt(self):
        self.running = False
