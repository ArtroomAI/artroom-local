import warnings
import gc
import sys

from glob import glob
from transformers import logging as tflogging
from pytorch_lightning import seed_everything
from upscale import Upscaler

sys.path.append("ComfyUI/")
from ComfyUI.nodes import *

from artroom_helpers.generation.preprocess import load_model_from_config, mask_from_face, mask_background, load_mask
from artroom_helpers.process_controlnet_images import apply_controlnet, HWC3, apply_inpaint
from artroom_helpers import support, inpainting

tflogging.set_verbosity_error()
warnings.filterwarnings("ignore", category=DeprecationWarning)

class NodeModules:
    def __init__(self):
        self.CLIPTextEncode = CLIPTextEncode()
        self.LoadImage = LoadImage()
        self.LoadImageMask = LoadImageMask()
        self.KSampler = KSampler()
        self.KSamplerAdvanced = KSamplerAdvanced()
        self.VAEDecode = VAEDecode()
        self.VAEDecodeTiled = VAEDecodeTiled()
        self.LoraLoader = LoraLoader()
        self.ControlNetLoader = ControlNetLoader()
        self.ControlNetApply = ControlNetApply()
        self.VAELoader = VAELoader()
        self.VAEEncodeForInpaint = VAEEncodeForInpaint()
        self.VAEEncode = VAEEncode()
        self.EmptyLatentImage = EmptyLatentImage()

class Model:
    def __init__(self, ckpt):
        self.controlnet_path = None
        self.model = None
        self.clip = None
        self.vae = None
        self.clipvision = None
        self.controlnet = None
        self.can_use_half_vae = True

        self.vae_path = None

        self.nodes = NodeModules()

        self.ckpt = ckpt

        self.steps = 0
        self.current_num = 0
        self.total_num = 0
        self.total_steps = 0
        self.load_model()

    def load_model(self):
        self.model, self.clip, self.vae, clipvision = comfy.sd.load_checkpoint_guess_config(self.ckpt,
                                                                                            output_vae=True,
                                                                                            output_clip=True,
                                                                                            embedding_directory=folder_paths.get_folder_paths(
                                                                                                "embeddings"))
        # if hasattr(self.clip.cond_stage_model, "clip_l"):  # sdxl
        #     self.can_use_half_vae = False
        del clipvision  # because dafuq

    def load_textual_inversions(self, textual_inversion_list):
        pass
        # for textual_inversion in textual_inversion_list:
        #     textual_inversion = os.path.basename(textual_inversion)
        #     self.prompt_appendices.append(f"embedding:{textual_inversion}")

    def setup(self, vae, loras=None, controlnet_path=None, device='cuda:0'):
        if loras is None:
            loras = []

        self.controlnet_path = controlnet_path
        try:
            if self.controlnet_path is not None:
                self.inject_controlnet(controlnet_path)
        except Exception as e:
            support.report_error("Controlnet failed to load", e)

        if vae != self.vae_path:
            try:
                if '.vae' in vae:
                    self.load_vae(vae, device=device)
            except:
                print("Failed to load vae")
        self.vae_path = vae

        if len(loras) > 0:
            for lora in loras:
                try:
                    self.inject_lora(path=lora['path'], weight_tenc=lora['weight'], weight_unet=lora['weight'],
                                     controlnet=(controlnet_path is not None), device=device)
                except Exception as e:
                    support.report_error("Failed to load in Lora {lora}", e)
        print("device: ", device)
        self.to(device)

    def get_steps(self):
        return self.current_num, self.total_num, self.model.current_step + self.steps * (
                self.current_num - 1), self.total_steps

    def inject_lora(self, path: str, weight_tenc=1.1, weight_unet=4, controlnet=False, device='cuda:0'):
        lora = comfy.utils.load_torch_file(path, safe_load=True)
        self.model, self.clip = comfy.sd.load_lora_for_models(self.model, self.clip, lora, weight_unet, weight_tenc)

    def deinject_lora(self, delete=True):
        self.load_model()

    def inject_controlnet(self, controlnet_path, existing=False):
        self.controlnet = comfy.sd.load_controlnet(controlnet_path)

    def deinject_controlnet(self, delete=True):
        del self.controlnet
    def load_vae(self, vae_path: str, device='cuda:0', safe_load_=True, original=False, controlnet=False):
        self.vae = comfy.sd.VAE(ckpt_path=vae_path, device=device)

    def to(self, device, dtype=torch.float32):
        self.vae.device = device
        self.vae.first_stage_model.to(device).to(torch.float32)

#         self.model.model.to(device).to(dtype)

        # self.clip.device = device
        # self.clip.cond_stage_model.device = device
        # self.clip.patcher.model.to(device).to(dtype)
        # self.clip.cond_stage_model.to(device).to(dtype)

#         if hasattr(self.clip.cond_stage_model, "clip_l"):
#             print("found")
#             self.clip.cond_stage_model.clip_l.device = device
#             self.clip.cond_stage_model.clip_g.device = device
#         else:
#             print(dir(self.clip.cond_stage_model))

#         # self.clipvision.to(device)
#         if self.controlnet is not None:
#             self.controlnet.to(device)


class StableDiffusion:
    def __init__(self, QM=None, producer=None):
        self.ready = False

        self.QM = QM
        self.upscaler = Upscaler()
        self.producer = producer
        self.nodes = NodeModules()

        self.active_model = None
        self.highres_fix = False

        #self.ait = AITModule()

    def send_job_status(self, progressType, job_id, n, currentStep=0, totalSteps=0, isNSFW=False):
        bucket_path = f"jobs/generate/{job_id}/{n}.jpg"

        job_status = {
            "jobId": job_id,
            "progressType": progressType,
            "modelProgress": {},
            "jobProgress": {
                "totalSteps": totalSteps,
                "currentStep": currentStep
            },
            "newImage": {
                "storagePath": bucket_path,
                "number": n,
                "isNSFW": isNSFW
            },
            "hasErrors": False,
            "errorsDetails": {
                "items": [
                    {
                        "userDisplayedErrorMessage": "string",
                        "internalErrorMessage": "string"
                    }
                ]
            }
        }
        if self.producer:
            self.producer.send('job_status', value=job_status)
        # print(f'job_status: {job_status}')

    def callback_fn(self, job_id, x0=None, enabled=True):
        if not enabled:
            return

        current_num, total_num, current_step, total_steps = self.active_model.get_steps()
        if current_step % 5 == 0:
            self.send_job_status(2, job_id, current_num, current_step, total_steps)
            # print(f"JOB ID: {job_id} {current_step}/{total_steps}")

    def get_image(self, init_image_str, mask_image, job_id):
        if init_image_str[:4] == 'data':
            init_image = support.b64_to_image(init_image_str)
        else:
            print(f"Downloading from drive {init_image_str}")
            support.download_image_from_google(init_image_str, job_id)
            init_image = Image.open(f'{job_id}_{os.path.basename(init_image_str)}')

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

    def load_image(self, image, h0, w0, mask_image=None, inpainting=False, device='cuda:0'):
        w, h = image.size
        if not inpainting and h0 != 0 and w0 != 0:
            h, w = h0, w0

        w, h = map(lambda x: x - x % 64, (w, h))
        # print(f"New image size ({w}, {h})")
        image = image.resize((w, h), resample=Image.LANCZOS)

        image = np.array(image).astype(np.float32) / 255.0
        init_image = torch.from_numpy(image)[None,]
        init_image = init_image.to(device)
        _, H, W, _ = init_image.shape
        #init_image = init_image.half()

        return init_image.to(device), H, W

    def load_control_image(self, image, h0, w0, mask_image=None, inpainting=False, controlnet_mode="none",
                           use_preprocessed_controlnet=False, device='cuda:0'):
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
                control = apply_controlnet(image, controlnet_mode)

        return control.to(device)

    def diffusion_upscale(
            self,
            job_id=0,
            n=0,
            prompts_data=None,
            negative_prompts_data=None,
            image=None,
            mask_image=None,
            control_image=None,
            controlnet="none",
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
            device='cuda:0',
            keep_size=False
    ):
        # try:
        if keep_size:   
            print("Running experimental diffusion touchup")
            original_W, original_H = image.size
            W = int(W * highres_multiplier)
            H = int(H * highres_multiplier)
        
        original_image = image.copy()
        upscaler_model = "RealESRGAN"

        temp_save_path = f'{job_id}_{n}_diffusion_touchup.png'
        image.save(temp_save_path)



        print("Upscaling image")
        image = \
            self.upscaler.upscale(images=[temp_save_path], upscaler=upscaler_model,
                                  upscale_factor=highres_multiplier)[
                0]

        image.resize((W, H))  # Ensures final dimensions are correct

        # image.save(f'{job_id}_{n}_upscaled.png')

        # generate the next version using the upscaled image as the new input
        highres_init_image, H, W = self.load_image(image.convert('RGB'), H, W, device=device)

        print("Generating upscaled image")
        image = self.generate_image(
            job_id=job_id,
            prompts_data=prompts_data,
            negative_prompts_data=negative_prompts_data,
            control_image=control_image,
            init_image=highres_init_image.to(device),
            mask_image=mask_image,
            steps=highres_steps,
            H=H,
            W=W,
            cfg_scale=cfg_scale,
            seed=seed,
            sampler=sampler,
            batch_size=batch_size,
            controlnet=controlnet,
            clip_skip=clip_skip,
            callback_fn=self.callback_fn,
            device=device,
            strength=highres_strength
        )     

        # except Exception as e:
        #     if "CUDA out of memory" in str(e):
        #         torch.cuda.empty_cache()  # free up GPU memory
        #         print("CUDA out of memory error occurred, skipping this image resolution")
        #     print(f"Failed to touchup generate image for resultion {H}x{W}: {e}")

        try:
            os.remove(temp_save_path)
            pass
        except Exception as e:
            print('Failed to delete diffusion touchup {e}')

        if mask_image is not None:
            image = support.repaste_and_color_correct(result=image,
                                                      init_image=original_image,
                                                      init_mask=mask_image, mask_blur_radius=8)

        if keep_size:
            image = image.resize((original_W, original_H))
        return image

    @torch.no_grad()
    def generate_image(
            self,
            job_id="",
            prompts_data=None,
            negative_prompts_data=None,
            control_image=None,
            init_image=None,
            mask_image=None,
            steps=50,
            H=512,
            W=512,
            controlnet_strength=1,
            cfg_scale=7.5,
            txt_cfg_scale=1.5,
            seed=-1,
            sampler="ddim",
            ddim_eta=0.0,
            batch_size=1,
            controlnet="none",
<<<<<<< Updated upstream
            control=None,
            precision_scope=None,
            highres_fix_steps=1,
            use_removed_background=False,
            remove_background="none",
            clip_skip=1
    ):

        for prompts in prompts_data:
            with precision_scope(self.device.type):
                if self.v1:
                    self.modelCS.to(self.device)
                if self.control_model:
                    self.model.switch_devices(diffusion_loop=False)

                uc = None
                if cfg_scale != 1.0:
                    uc = self.modelCS.get_learned_conditioning(negative_prompts_data, clip_skip=clip_skip)
                if isinstance(prompts, tuple):
                    prompts = list(prompts)
                c = self.modelCS.get_learned_conditioning(batch_size * [prompts], clip_skip=clip_skip)

                shape = [batch_size, C, H // f, W // f]
                if self.control_model:
                    self.model.switch_devices(diffusion_loop=True)
                    self.modelCS.cpu()

                x0 = None
                for ij in range(1, highres_fix_steps + 1):
                    self.current_num = n * highres_fix_steps + ij - 1
                    self.model.current_step = 0
                    self.model.total_steps = steps * highres_fix_steps

                    if ij > 1:
                        strength = 0.15
                        steps = 5
                        ddim_steps = int(steps / strength)

                    if init_image is not None and not self.control_model:
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

                    if use_removed_background and image is not None:
                        if remove_background == 'face':
                            mask_image = mask_from_face(image.convert('RGB'), W, H)
                        else:
                            mask_image = mask_background(image.convert('RGB'),
                                                         remove_background=remove_background)
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

                    if self.control_model and controlnet is not None and controlnet.lower() != "none" and control is not None:
                        # control = torch.load("control.torch")
                        c = {"c_concat": [control], "c_crossattn": [c]}
                        uc = {"c_concat": [control], "c_crossattn": [uc]}
                        self.model.control_scales = [1.0] * 13
                        ddim_sampler = DDIMSampler(self.model)
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
                        if self.control_model:
                            self.model.switch_devices(diffusion_loop=False)
                    else:
                        x0 = x0 if (init_image is None or "ddim" in sampler.lower()) else init_latent
                        x0 = init_latent_1stage if mode == "pix2pix" else x0
                        gen_kwargs = {
                            "S": steps,
                            "conditioning": c,
                            "x0": x0,
                            "S_ddim_steps": ddim_steps,
                            "unconditional_guidance_scale": cfg_scale,
                            "txt_scale": txt_cfg_scale,
                            "unconditional_conditioning": uc,
                            "eta": ddim_eta,
                            "sampler": sampler,
                            "shape": shape,
                            "batch_size": batch_size,
                            "seed": seed,
                            "mask": mask,
                            "x_T": x_T,
                            "callback": self.callback_fn,
                            "mode": mode}
                        x0 = self.model.sample(**gen_kwargs)
                    if self.v1:
                        self.modelFS.to(self.device)

                    x_samples_ddim = self.modelFS.decode_first_stage(
                        x0[0].unsqueeze(0))

                    if x_samples_ddim.sum().isnan():  # black square fix
                        print("Black square detected, repeating on full precision")
                        try:
                            self.model.to(torch.float32)
                            self.modelFS.to(torch.float32)
                            gen_kwargs = {"S": steps, "conditioning": c.to(torch.float32), "x0": x0.to(torch.float32),
                                          "S_ddim_steps": ddim_steps, "unconditional_guidance_scale": cfg_scale,
                                          "txt_scale": txt_cfg_scale, "unconditional_conditioning": uc, "eta": ddim_eta,
                                          "sampler": sampler, "shape": shape, "batch_size": batch_size, "seed": seed,
                                          "mask": mask, "x_T": x_T, "callback": self.callback_fn, "mode": mode}

                            x0 = self.model.sample(**gen_kwargs)
                            x_samples_ddim = self.modelFS.decode_first_stage(x0[0].unsqueeze(0))
                        except:
                            pass
                        if self.can_use_half:
                            self.modelFS.half()
                            self.model.half()
                            x_samples_ddim = x_samples_ddim.half()
                    x_sample = torch.clamp(
                        (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_sample = 255. * \
                               rearrange(
                                   x_sample[0].cpu().numpy(), 'c h w -> h w c')
                    out_image = Image.fromarray(
                        x_sample.astype(np.uint8))

                    if self.can_use_half:
                        self.modelFS.half()
                        # torch.set_default_tensor_type(torch.HalfTensor)

                    if ij < highres_fix_steps - 1:
                        init_image = self.load_img(
                            out_image, H * (ij + 1), W * (ij + 1), inpainting=(len(mask_b64) > 0)).to(
                            self.device).to(self.dtype)  # we only encode cnet 1 time
                    elif ij == highres_fix_steps - 1:
                        init_image = self.load_img(out_image, oldH, oldW, inpainting=(len(mask_b64) > 0)).to(
                            self.device).to(self.dtype)
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
            return out_image

    def generate(
            self,
            text_prompts="",
            negative_prompts="",
            init_image_str="",
            mask_b64="",
            invert=False,
            txt_cfg_scale=1.5,
            steps=50,
            H=512,
            W=512,
            strength=0.75,
            cfg_scale=7.5,
            seed=-1,
=======
>>>>>>> Stashed changes
            clip_skip=1,
            device='cuda:0',
            callback_fn=None,
            strength=1.0
    ):
        
        if isinstance(prompts_data, list):
            prompts_data = prompts_data[0]
        # prompts_data += " ".join(self.active_model.prompt_appendices)

        if isinstance(negative_prompts_data, list):
            negative_prompts_data = negative_prompts_data[0]

        dtype = torch.float32

        positive_cond = self.nodes.CLIPTextEncode.encode(self.active_model.clip, prompts_data)[0]
        negative_prompt = self.nodes.CLIPTextEncode.encode(self.active_model.clip, negative_prompts_data)[0]

        if controlnet is not None and controlnet != "none":
            control_image = control_image.permute(0, 2, 3, 1)
            positive_cond = self.nodes.ControlNetApply.apply_controlnet(
                positive_cond, self.active_model.controlnet, control_image, controlnet_strength)[0]

        if mask_image is not None:
            mask_image = np.array(mask_image).astype(np.float32) / 255.0
            mask_image = 1. - torch.from_numpy(mask_image).to(dtype).to(init_image.device)
            with init_image.device, torch.autocast("cuda"):
                if init_image.shape[1] == 3:
                    init_image = init_image.permute(0, 2, 3, 1)
                init_image = self.nodes.VAEEncodeForInpaint.encode(self.active_model.vae, init_image, mask_image)[0]
        elif init_image is not None:
            init_image = self.nodes.VAEEncode.encode(self.active_model.vae, init_image)[0]        
        else:
            init_image = self.nodes.EmptyLatentImage.generate(width=W, height=H, batch_size=batch_size)[0]
        
        self.active_model.to(device)
        scheduler = "karras"  # ["normal", "karras", "exponential", "simple", "ddim_uniform"]
        out_image = self.nodes.KSampler.sample(self.active_model.model, seed, steps, cfg_scale, sampler, scheduler,
                                               positive_cond, negative_prompt, init_image, denoise=strength)[0]
        out_image["samples"] = out_image["samples"].to(dtype)

        out_image = self.nodes.VAEDecode.decode(self.active_model.vae, out_image)[0]
        out_image = 255. * out_image[0].cpu().numpy()
        out_image = Image.fromarray(np.clip(out_image, 0, 255).astype(np.uint8))
        return out_image

    def generate(self,
                 job_id="",
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
                 background_removal_type="none",
                 clip_skip=1,
                 palette_fix=False,
                 device='cuda:0'
                 ):
        use_ait = False
        if loras is None:
            loras = []

        if "sd_xl" in ckpt:
            controlnet = "none"
            vae = "none"
            loras = []

        if len(init_image_str) == 0:
            controlnet = 'none'

        if vae is not None and len(vae) > 0:
            vae = os.path.join("models", "vae", vae)

        def get_contronet_path(controlnet):
            controlnet_folder = os.path.join("models", "ControlNet")

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
        if model_key not in self.QM.cpu_cache:
            self.QM.setup_cache(ckpt, controlnet_path)
        seed_everything(seed)

        
        print(f"Generating... {job_id} dim: {W}x{H}, Prompt: {text_prompts}")
        
        def resize_dims_balanced(width, height):
            # Calculate the aspect ratio of the original dimensions
            aspect_ratio = width / height

            # Scale the width and height so their sum is 2048
            new_width = (2048 * aspect_ratio) / (aspect_ratio + 1)
            new_height = (2048 * 1) / (aspect_ratio + 1)

            # Adjust to the nearest multiple of 64
            new_width = round(new_width / 64) * 64
            new_height = round(new_height / 64) * 64
            print(f"New WxH: {new_width}x{new_height}")
            return int(new_width), int(new_height)
        
        starting_image = None
        init_image = None
        mask_image = None
        control_image = None
        highres_fix = False
        
        if W*H > 1024*1024:
            highres_fix = True
            old_H = H
            old_W = W
            W, H = resize_dims_balanced(W,H)
        
        
        if len(mask_str) > 0 and len(init_image_str) > 0:
            if mask_str[:4] == 'data':
                mask_image = support.b64_to_image(mask_str).convert('L')
            else:
                support.download_image_from_google(mask_str, job_id)
                mask_image = Image.open(f'{job_id}_{os.path.basename(mask_str)}').convert('L')

        # Get image and extract mask from image if needed
        if len(init_image_str) > 0:
            starting_image, mask_image = self.get_image(init_image_str, mask_image, job_id)

        if background_removal_type != 'none' and starting_image is not None:
            if background_removal_type == 'face':
                mask_image = mask_from_face(starting_image.convert('RGB'), W, H)
            else:
                mask_image = mask_background(starting_image.convert('RGB'),
                                             remove_background=background_removal_type)

        if starting_image is not None:
            starting_image = starting_image.resize((W, H))
            if mask_image is not None:
                mask_image = mask_image.resize((W, H))
                if invert:
                    mask_image = ImageOps.invert(mask_image)
            init_image, H, W = self.load_image(starting_image, H, W, mask_image=mask_image,
                                               inpainting=(mask_image is not None or background_removal_type != 'none'),
                                               device=device)
            if controlnet != "none":
                control_image = self.load_control_image(starting_image, H, W, mask_image=mask_image,
                                                        inpainting=(
                                                                mask_image is not None or background_removal_type != 'none'),
                                                        controlnet_mode=controlnet, device=device)

        self.active_model = Model(ckpt)
        self.active_model.setup(vae, loras, controlnet_path, device)

        torch.cuda.empty_cache()
        gc.collect()

        prompts_data = [batch_size * text_prompts]
        negative_prompts_data = [batch_size * negative_prompts]

        self.active_model.to(device)
        total_steps = steps * n_iter

        self.active_model.model.job_id = job_id
        self.active_model.model.current_step = 0
        self.active_model.steps = steps
        self.active_model.total_steps = total_steps
        self.active_model.total_num = n_iter

        print(f'''Generating
            Job Id: {job_id}
            Text Prompts: {text_prompts}, 
            Negative Prompts: {negative_prompts}, 
            Init Image: {init_image_str}, 
            Mask: {mask_str}, 
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

        with torch.no_grad():
            for n in range(1, n_iter + 1):
                self.active_model.current_num = n
                try:
                    print(f'Generating on gpu {device} image job {job_id} image {n}')
                    out_image = self.generate_image(
                        job_id=job_id,
                        prompts_data=prompts_data,
                        negative_prompts_data=negative_prompts_data,
                        control_image=control_image,
                        init_image=init_image,
                        mask_image=mask_image,
                        steps=steps,
                        H=H,
                        W=W,
                        cfg_scale=cfg_scale,
                        seed=seed,
                        sampler=sampler,
                        batch_size=batch_size,
                        controlnet=controlnet,
                        clip_skip=clip_skip,
                        callback_fn=self.callback_fn,
                        device=device,
                        strength=strength if starting_image is not None else 1.0
                    )
                    
                    if highres_fix:
                        out_image = self.diffusion_upscale(
                            job_id=job_id,
                            n=n,
                            prompts_data=prompts_data,
                            negative_prompts_data=negative_prompts_data,
                            image=out_image,
                            mask_image=mask_image,
                            highres_steps=15,
                            highres_strength=0.2,
                            highres_multiplier=min(old_H/H,old_W/W),
                            cfg_scale=cfg_scale,
                            H=old_H,
                            W=old_W,
                            seed=seed,
                            sampler=sampler,
                            clip_skip=clip_skip,
                            device=device,
                            keep_size=False
                        )

                    out_image.save(filename)
                except Exception as e:
                    support.report_error("Failed to generate image", e)
                    traceback.print_exc()
                seed += 1
        self.clean_up()

    def clean_up(self):
        torch.cuda.empty_cache()
        gc.collect()
