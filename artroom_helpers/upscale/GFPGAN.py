import os
from artroom_helpers.gpu_detect import get_gpu_architecture
import cv2
import numpy as np
from PIL import Image

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
