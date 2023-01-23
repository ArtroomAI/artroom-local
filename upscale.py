import json
import os
import time
import subprocess
from glob import glob
import shutil
import sys
import warnings
from artroom_helpers.gpu_detect import get_gpu_architecture

warnings.filterwarnings("ignore")


class Upscaler():
    def __init__(self):
        self.artroom_path = None
        self.upscale_queue_path = None
        self.running = False

    def set_artroom_path(self, path):
        self.artroom_path = path
        self.upscale_queue_path = f"{self.artroom_path}/artroom/settings/upscale_queue/"
        # sys.path.append(f"{self.artroom_path}/artroom/upscalers/RealESRGAN")
        sys.path.append(f"{self.artroom_path}/artroom/upscalers/GFPGAN")

    def upscale(self, images, upscaler, upscale_factor, upscale_dest, upscale_strength=None):
        self.running = True
        if upscale_dest == "":
            upscale_dest = os.path.dirname(images[0])
        elif upscale_dest[-1] == "/":
            upscale_dest = upscale_dest[:-1]
            
        upscale_dest += f"/{upscaler.replace(' ','')}/"

        try:
            self.add_images(images)
            if "GFPGAN" in upscaler or "RestoreFormer" in upscaler:
                output_images, save_paths = self.GFPGAN(upscaler, upscale_factor, upscale_dest)
            elif "RealESRGAN" in upscaler:
                output_images, save_paths = self.RealESRGAN(upscaler, upscale_factor, upscale_dest)

            print("Running cleanup")
            # Clean up
            shutil.rmtree(self.upscale_queue_path)
            self.running = False
            return {"status": "Success", "status_message": "", "content": {"output_images": output_images, "save_paths": save_paths}}
        except Exception as e:
            self.running = False
            print(f"Upscale Failed, error: {e}")
            return {"status": "Failure", "status_message": f"Error: {e}", "content": {"output_images": output_images, "save_paths": save_paths}}

    def GFPGAN(self, upscaler, upscale_factor, upscale_dest):
        # sys.path.append(f"{self.artroom_path}/artroom/upscalers/GFPGAN")\
        import cv2
        import numpy as np
        import torch
        from basicsr.utils import imwrite
        from gfpgan import GFPGANer
        from basicsr.utils.download_util import load_file_from_url
        from PIL import Image

        # Potentially problematic
        torch.set_default_tensor_type(torch.FloatTensor)
        input = self.upscale_queue_path
        upscale = upscale_factor
        if "1.3" in upscaler:
            version = "1.3"  # GFPGANv1.3
        elif "1.4" in upscaler:
            version = "1.4"  # GFPGANv1.4
        else:
            version = upscaler  # RestoreFormer

        outdir = upscale_dest
        bg_upsampler = "realesrgan"
        bg_tile = 400
        suffix = "_upscaled"
        ext = "auto"

        if input.endswith('/'):
            input = input[:-1]
        if os.path.isfile(input):
            img_list = [input]
        else:
            img_list = sorted(glob(os.path.join(input, '*')))
        os.makedirs(outdir, exist_ok=True)

        # ------------------------ set up background upsampler ------------------------
        if bg_upsampler == 'realesrgan':
            if not torch.cuda.is_available():  # CPU
                bg_upsampler = None
            else:
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from realesrgan import RealESRGANer
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                                num_block=23, num_grow_ch=32, scale=2)

                model_name = "RealESRGAN_x2plus"
                url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'
                model_dest = f'{self.artroom_path}/artroom/upscalers/Real-ESRGAN/weights/'

                model_path = os.path.join(model_dest, model_name + '.pth')
                if not os.path.isfile(model_path):
                    model_path = load_file_from_url(
                        url=url, model_dir=model_dest, progress=True, file_name=None)

                bg_upsampler = RealESRGANer(
                    scale=2,
                    model_path=model_path,
                    model=model,
                    tile=bg_tile,
                    tile_pad=10,
                    pre_pad=0,
                    half=False)  # need to set False in CPU mode
        else:
            bg_upsampler = None

        if not bg_upsampler:
            print("BG Upsampler not found")

        if version == '1.3':
            arch = 'clean'
            channel_multiplier = 2
            model_name = 'GFPGANv1.3'
            file_url = [
                'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth']
            model_dest = f'{self.artroom_path}/artroom/upscalers/GFPGAN/experiments/pretrained_models/'
        elif version == '1.4':
            arch = 'clean'
            channel_multiplier = 2
            model_name = 'GFPGANv1.4'
            file_url = [
                'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth']
            model_dest = f'{self.artroom_path}/artroom/upscalers/GFPGAN/experiments/pretrained_models/'
        elif version == 'RestoreFormer':
            arch = 'RestoreFormer'
            channel_multiplier = 2
            model_name = 'RestoreFormer'
            file_url = [
                'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth']
            model_dest = f'{self.artroom_path}/artroom/upscalers/GFPGAN/experiments/pretrained_models/'

        # determine model paths
        # model_path = os.path.join(f'{self.artroom_path}/artroom/model_weights/upscalers', model_name + '.pth')
        # if not os.path.isfile(model_path):
        #     # download pre-trained models from url
        #     model_path = url

        model_path = os.path.join(model_dest, model_name + '.pth')
        if not os.path.isfile(model_path):
            for url in file_url:
                model_path = load_file_from_url(
                    url=url, model_dir=f'{self.artroom_path}/artroom/model_weights/upscalers', progress=True, file_name=None)

        restorer = GFPGANer(
            model_path=model_path,
            upscale=upscale,
            arch=arch,
            channel_multiplier=channel_multiplier,
            bg_upsampler=bg_upsampler)

        output_images = []
        save_paths = []
        # ------------------------ restore ------------------------
        for img_path in img_list:
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
            # save faces
            for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_faces)):
                # save cropped face
                save_crop_path = os.path.join(
                    outdir, 'cropped_faces', f'{basename}_{idx:02d}.png')
                imwrite(cropped_face, save_crop_path)
                # save restored face
                if suffix is not None:
                    save_face_name = f'{basename}_{idx:02d}_{suffix}.png'
                else:
                    save_face_name = f'{basename}_{idx:02d}.png'
                save_restore_path = os.path.join(
                    outdir, 'restored_faces', save_face_name)
                imwrite(restored_face, save_restore_path)
                # save comparison image
                cmp_img = np.concatenate((cropped_face, restored_face), axis=1)
                imwrite(cmp_img, os.path.join(
                    outdir, 'cmp', f'{basename}_{idx:02d}.png'))

            # save restored img
            if restored_img is not None:
                if ext == 'auto':
                    extension = ext[1:]
                else:
                    extension = ext

                if suffix is not None:
                    save_restore_path = os.path.join(
                        outdir, 'restored_imgs', f'{basename}_{suffix}.{extension}')
                else:
                    save_restore_path = os.path.join(
                        outdir, 'restored_imgs', f'{basename}.{extension}')
                imwrite(restored_img, save_restore_path)
                output_images.append(Image.fromarray(cv2.cvtColor(restored_img, cv2.COLOR_BGRA2RGB)))
                save_paths.append(save_restore_path)
        tensor = torch.HalfTensor if get_gpu_architecture() == 'NVIDIA' else torch.FloatTensor
        torch.set_default_tensor_type(tensor)
        return output_images, save_paths
        
    def RealESRGAN(self, upscaler, upscale_factor, upscale_dest):
        import cv2
        import numpy as np
        from PIL import Image
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from basicsr.utils.download_util import load_file_from_url
        from realesrgan import RealESRGANer
        
        outdir = upscale_dest
        outscale = upscale_factor
        input = self.upscale_queue_path

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
            file_url = [
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
            model_dest = f'{self.artroom_path}/artroom/upscalers/Real-ESRGAN/experiments/pretrained_models/'
        elif model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                            num_block=6, num_grow_ch=32, scale=4)
            netscale = 4
            file_url = [
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
            model_dest = f'{self.artroom_path}/artroom/upscalers/Real-ESRGAN/experiments/pretrained_models/'

        # determine model paths
        #model_path = os.path.join(f'{self.artroom_path}/artroom/model_weights/upscalers', model_name + '.pth')
        model_path = os.path.join(model_dest, model_name + '.pth')
        if not os.path.isfile(model_path):
            for url in file_url:
                model_path = load_file_from_url(
                    url=url, model_dir=f'{self.artroom_path}/artroom/model_weights/upscalers', progress=True, file_name=None)

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

        os.makedirs(outdir, exist_ok=True)

        if input.endswith('/'):
            input = input[:-1]
        if os.path.isfile(input):
            img_list = [input]
        else:
            img_list = sorted(glob(os.path.join(input, '*')))
        output_images = []
        save_paths = []
        for idx, path in enumerate(img_list):
            imgname, extension = os.path.splitext(os.path.basename(path))
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            try:
                output, _ = upsampler.enhance(img, outscale=outscale)
            except RuntimeError as error:
                print('Error', error)
                print(
                    'If you encounter CUDA out of memory, try to set --tile with a smaller number.')
            else:
                save_path = os.path.join(
                    outdir, f'{imgname}_{suffix}.{extension[1:]}')
                print(save_path)
                cv2.imwrite(save_path, output)
                output_images.append(Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGRA2RGB)))
                save_paths.append(save_path)
        return output_images, save_paths

    def add_images(self, images):
        # Filter non-images
        images = [image for image in images if (
            ".jpg" in image or ".png" in image or ".jpeg" in image)]
        if os.path.exists(self.upscale_queue_path):
            shutil.rmtree(self.upscale_queue_path)

        os.makedirs(self.upscale_queue_path, exist_ok=True)
        for image in images:
            shutil.copy(image, self.upscale_queue_path)


if __name__ == "__main__":
    US = Upscaler()
    US.set_artroom_path(os.environ['USERPROFILE'])
    US.upscale()
