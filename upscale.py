import json
import os
import time
import subprocess
from glob import glob
import shutil
import sys
from artroom_helpers.gpu_detect import get_gpu_architecture
import cv2
import numpy as np
from PIL import Image
from basicsr.utils.download_util import load_file_from_url

class Upscaler():
    def __init__(self):
        self.artroom_path = None
        self.upscale_queue_path = None
        self.running = False
        self.images = []

    def add_images(self, images):
        # Filter non-images
        self.images = [image for image in images if (
            ".jpg" in image or ".png" in image or ".jpeg" in image)]

    def get_dest_path(self, upscale_dest, upscaler, image_path):
        if upscale_dest == "":
            upscale_dest = os.path.dirname(image_path)
        elif upscale_dest[-1] == "/":
            upscale_dest = upscale_dest[:-1]
            
        upscale_dest += f"/{upscaler.replace(' ','')}"
        return upscale_dest

    def upscale(self, models_dir, images, upscaler, upscale_factor, upscale_dest, upscale_strength=None):
        self.running = True

        self.artroom_path = models_dir  
        upscale_dest = self.get_dest_path(upscale_dest, upscaler, images[0])
        self.upscale_queue_path = f"{upscale_dest}/temp"

        try:
            self.add_images(images)
            os.makedirs(upscale_dest, exist_ok=True)
            if "GFPGAN" in upscaler or "RestoreFormer" in upscaler:
                output_images, save_paths = self.GFPGAN(upscaler, upscale_factor, upscale_dest)
            elif "RealESRGAN" in upscaler:
                output_images, save_paths = self.RealESRGAN(upscaler, upscale_factor, upscale_dest)

            print("Running cleanup")
            # Clean up
            #shutil.rmtree(self.upscale_queue_path)
            self.running = False
            return {"status": "Success", "status_message": "", "content": {"output_images": output_images, "save_paths": save_paths}}
        except Exception as e:
            self.running = False
            print(f"Upscale Failed, error: {e}")
            return {"status": "Failure", "status_message": f"Error: {e}", "content": {"output_images": output_images, "save_paths": save_paths}}

    def download_upscaler(self, url):
        model_dest = f'{self.artroom_path}/upscalers'
        print(model_dest)
        model_name = os.path.basename(url)
        model_path = os.path.join(model_dest, model_name)
        if not os.path.isfile(model_path):
            model_path = load_file_from_url(
                url=url, model_dir=model_dest, progress=True, file_name=None)
        return model_path

    def GFPGAN(self, upscaler, upscale_factor, upscale_dest):
        import torch
        from basicsr.utils import imwrite
        from gfpgan import GFPGANer

        if "1.3" in upscaler:
            version = "1.3"  # GFPGANv1.3
        elif "1.4" in upscaler:
            version = "1.4"  # GFPGANv1.4
        else:
            version = upscaler  # RestoreFormer

        bg_upsampler = "realesrgan"
        bg_tile = 400
        suffix = "_upscaled"
        ext = "auto"

        # ------------------------ set up background upsampler ------------------------
        if bg_upsampler == 'realesrgan' and torch.cuda.is_available():
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                            num_block=23, num_grow_ch=32, scale=2)
            use_half = get_gpu_architecture() == 'NVIDIA'
            model_path = self.download_upscaler(
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth')

            bg_upsampler = RealESRGANer(
                scale=2,
                model_path=model_path,
                model=model,
                tile=bg_tile,
                tile_pad=10,
                pre_pad=0,
                half=use_half)
        else:
            bg_upsampler = None

        if not bg_upsampler:
            print("BG Upsampler not found")

        if version == '1.3':
            arch = 'clean'
            channel_multiplier = 2
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
        elif version == '1.4':
            arch = 'clean'
            channel_multiplier = 2
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
        elif version == 'RestoreFormer':
            arch = 'RestoreFormer'
            channel_multiplier = 2
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'

        model_path = self.download_upscaler(url)

        try:
            restorer = GFPGANer(
                model_path=model_path,
                upscale=upscale_factor,
                arch=arch,
                channel_multiplier=channel_multiplier,
                bg_upsampler=bg_upsampler)
        except PermissionError:
            print("Downloading internal GFPGAN models need additional permissions. Restart Artroom with admin privilage")
            return [], []

        output_images = []
        save_paths = []
        # ------------------------ restore ------------------------
        for img_path in self.images:
            # read image
            img_name = os.path.basename(img_path)
            print(f'Processing {img_name} ...')
            basename, ext = os.path.splitext(img_name)
            input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

            print("Restoring face...")
            # restore faces and background if necessary
            cropped_faces, restored_faces, restored_img = restorer.enhance(
                input_img,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
                weight=0.5)
            print("Saving face...")
            if len(cropped_faces) == 0:
                print("no faces were restored. If there were faces in the image, please restart artroom as administrator.")

            # save faces
            for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_faces)):
                # save cropped face
                save_crop_path = os.path.join(
                    upscale_dest, 'cropped_faces', f'{basename}_{idx:02d}.png')
                print(save_crop_path)
                imwrite(cropped_face, save_crop_path)
                # save restored face
                save_face_name = f'{basename}_{idx:02d}_{suffix}.png'
                save_restore_path = os.path.join(
                    upscale_dest, 'restored_faces', save_face_name)
                imwrite(restored_face, save_restore_path)
                # save comparison image
                cmp_img = np.concatenate((cropped_face, restored_face), axis=1)
                imwrite(cmp_img, os.path.join(
                    upscale_dest, 'cmp', f'{basename}_{idx:02d}.png'))

            # save restored img
            if restored_img is not None:
                if ext == 'auto':
                    extension = ext[1:]
                else:
                    extension = ext

                save_restore_path = os.path.join(
                    upscale_dest, 'restored_imgs', f'{basename}_{suffix}.{extension}')
                imwrite(restored_img, save_restore_path)
                output_images.append(Image.fromarray(cv2.cvtColor(restored_img, cv2.COLOR_BGRA2RGB)))
                save_paths.append(save_restore_path)
        return output_images, save_paths
        
    def RealESRGAN(self, upscaler, upscale_factor, upscale_dest):
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        if upscaler == "RealESRGAN":
            model_name = "RealESRGAN_x4plus"
            suffix = "upscaled_R"
        elif "Anime" in upscaler:
            model_name = "RealESRGAN_x4plus_anime_6B"
            suffix = "upscaled_anime"

        if model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                            num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
        elif model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                            num_block=6, num_grow_ch=32, scale=4)
            netscale = 4
            url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth'

        model_path = self.download_upscaler(url)
        use_half = get_gpu_architecture() == 'NVIDIA'

        # restorer
        upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            model=model,
            tile=400,
            tile_pad=10,
            pre_pad=0,
            half=use_half)

        output_images = []
        save_paths = []
        for path in self.images:
            imgname, extension = os.path.splitext(os.path.basename(path))
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            try:
                output, _ = upsampler.enhance(img, outscale=upscale_factor)
            except RuntimeError as error:
                print('Error', error)
                print(
                    'If you encounter CUDA out of memory, try to set --tile with a smaller number.')
            else:
                save_path = os.path.join(
                    upscale_dest, f'{imgname}_{suffix}.{extension[1:]}')
                print(save_path)
                cv2.imwrite(save_path, output)
                output_images.append(Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGRA2RGB)))
                save_paths.append(save_path)
        return output_images, save_paths
