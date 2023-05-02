import os
from basicsr.utils.download_util import load_file_from_url
from artroom_helpers.gpu_detect import get_gpu_architecture
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import cv2
import os 
from PIL import Image 
import torch
import math
import numpy as np 
from collections import namedtuple

Grid = namedtuple("Grid", ["tiles", "tile_w", "tile_h", "image_w", "image_h", "overlap"])

def split_grid(image, tile_w=512, tile_h=512, overlap=64):
    w = image.width
    h = image.height

    non_overlap_width = tile_w - overlap
    non_overlap_height = tile_h - overlap

    cols = math.ceil((w - overlap) / non_overlap_width)
    rows = math.ceil((h - overlap) / non_overlap_height)

    dx = (w - tile_w) / (cols - 1) if cols > 1 else 0
    dy = (h - tile_h) / (rows - 1) if rows > 1 else 0

    grid = Grid([], tile_w, tile_h, w, h, overlap)
    for row in range(rows):
        row_images = []

        y = int(row * dy)

        if y + tile_h >= h:
            y = h - tile_h

        for col in range(cols):
            x = int(col * dx)

            if x + tile_w >= w:
                x = w - tile_w

            tile = image.crop((x, y, x + tile_w, y + tile_h))

            row_images.append([x, tile_w, tile])

        grid.tiles.append([y, tile_h, row_images])

    return grid

def combine_grid(grid):
    def make_mask_image(r):
        r = r * 255 / grid.overlap
        r = r.astype(np.uint8)
        return Image.fromarray(r, 'L')

    mask_w = make_mask_image(np.arange(grid.overlap, dtype=np.float32).reshape((1, grid.overlap)).repeat(grid.tile_h, axis=0))
    mask_h = make_mask_image(np.arange(grid.overlap, dtype=np.float32).reshape((grid.overlap, 1)).repeat(grid.image_w, axis=1))

    combined_image = Image.new("RGB", (grid.image_w, grid.image_h))
    for y, h, row in grid.tiles:
        combined_row = Image.new("RGB", (grid.image_w, h))
        for x, w, tile in row:
            if x == 0:
                combined_row.paste(tile, (0, 0))
                continue

            combined_row.paste(tile.crop((0, 0, grid.overlap, h)), (x, 0), mask=mask_w)
            combined_row.paste(tile.crop((grid.overlap, 0, w, h)), (x + grid.overlap, 0))

        if y == 0:
            combined_image.paste(combined_row, (0, 0))
            continue

        combined_image.paste(combined_row.crop((0, 0, combined_row.width, grid.overlap)), (0, y), mask=mask_h)
        combined_image.paste(combined_row.crop((0, grid.overlap, combined_row.width, h)), (0, y + grid.overlap))

    return combined_image

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
            else:
                output_images, save_paths = self.ESRGAN(upscaler, upscale_factor, upscale_dest)

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
        
    def RealESRGAN(self, upscaler, upscale_factor, upscale_dest):
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
            tile=256,
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

    def GFPGAN(self, upscaler, upscale_factor, upscale_dest):
        from basicsr.utils import imwrite
        from gfpgan import GFPGANer

        if "1.3" in upscaler:
            version = "1.3"  # GFPGANv1.3
        elif "1.4" in upscaler:
            version = "1.4"  # GFPGANv1.4
        else:
            version = upscaler  # RestoreFormer

        bg_upsampler = "realesrgan"
        bg_tile = 256
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


    def ESRGAN(self, upscaler, upscale_factor, upscale_dest):
        import artroom_helpers.upscale.esrgan_model_arch as arch
        if "UltraSharp" in upscaler:
            filename = os.path.join(self.artroom_path,"upscalers", "4x-UltraSharp.pth")
        else:
            filename = None

        def mod2normal(state_dict):
            # this code is copied from https://github.com/victorca25/iNNfer
            if 'conv_first.weight' in state_dict:
                crt_net = {}
                items = []
                for k, v in state_dict.items():
                    items.append(k)

                crt_net['model.0.weight'] = state_dict['conv_first.weight']
                crt_net['model.0.bias'] = state_dict['conv_first.bias']

                for k in items.copy():
                    if 'RDB' in k:
                        ori_k = k.replace('RRDB_trunk.', 'model.1.sub.')
                        if '.weight' in k:
                            ori_k = ori_k.replace('.weight', '.0.weight')
                        elif '.bias' in k:
                            ori_k = ori_k.replace('.bias', '.0.bias')
                        crt_net[ori_k] = state_dict[k]
                        items.remove(k)

                crt_net['model.1.sub.23.weight'] = state_dict['trunk_conv.weight']
                crt_net['model.1.sub.23.bias'] = state_dict['trunk_conv.bias']
                crt_net['model.3.weight'] = state_dict['upconv1.weight']
                crt_net['model.3.bias'] = state_dict['upconv1.bias']
                crt_net['model.6.weight'] = state_dict['upconv2.weight']
                crt_net['model.6.bias'] = state_dict['upconv2.bias']
                crt_net['model.8.weight'] = state_dict['HRconv.weight']
                crt_net['model.8.bias'] = state_dict['HRconv.bias']
                crt_net['model.10.weight'] = state_dict['conv_last.weight']
                crt_net['model.10.bias'] = state_dict['conv_last.bias']
                state_dict = crt_net
            return state_dict

        def resrgan2normal(state_dict, nb=23):
            # this code is copied from https://github.com/victorca25/iNNfer
            if "conv_first.weight" in state_dict and "body.0.rdb1.conv1.weight" in state_dict:
                re8x = 0
                crt_net = {}
                items = []
                for k, v in state_dict.items():
                    items.append(k)

                crt_net['model.0.weight'] = state_dict['conv_first.weight']
                crt_net['model.0.bias'] = state_dict['conv_first.bias']

                for k in items.copy():
                    if "rdb" in k:
                        ori_k = k.replace('body.', 'model.1.sub.')
                        ori_k = ori_k.replace('.rdb', '.RDB')
                        if '.weight' in k:
                            ori_k = ori_k.replace('.weight', '.0.weight')
                        elif '.bias' in k:
                            ori_k = ori_k.replace('.bias', '.0.bias')
                        crt_net[ori_k] = state_dict[k]
                        items.remove(k)

                crt_net[f'model.1.sub.{nb}.weight'] = state_dict['conv_body.weight']
                crt_net[f'model.1.sub.{nb}.bias'] = state_dict['conv_body.bias']
                crt_net['model.3.weight'] = state_dict['conv_up1.weight']
                crt_net['model.3.bias'] = state_dict['conv_up1.bias']
                crt_net['model.6.weight'] = state_dict['conv_up2.weight']
                crt_net['model.6.bias'] = state_dict['conv_up2.bias']

                if 'conv_up3.weight' in state_dict:
                    # modification supporting: https://github.com/ai-forever/Real-ESRGAN/blob/main/RealESRGAN/rrdbnet_arch.py
                    re8x = 3
                    crt_net['model.9.weight'] = state_dict['conv_up3.weight']
                    crt_net['model.9.bias'] = state_dict['conv_up3.bias']

                crt_net[f'model.{8+re8x}.weight'] = state_dict['conv_hr.weight']
                crt_net[f'model.{8+re8x}.bias'] = state_dict['conv_hr.bias']
                crt_net[f'model.{10+re8x}.weight'] = state_dict['conv_last.weight']
                crt_net[f'model.{10+re8x}.bias'] = state_dict['conv_last.bias']

                state_dict = crt_net
            return state_dict

        def infer_params(state_dict):
            # this code is copied from https://github.com/victorca25/iNNfer
            scale2x = 0
            scalemin = 6
            n_uplayer = 0
            plus = False

            for block in list(state_dict):
                parts = block.split(".")
                n_parts = len(parts)
                if n_parts == 5 and parts[2] == "sub":
                    nb = int(parts[3])
                elif n_parts == 3:
                    part_num = int(parts[1])
                    if (part_num > scalemin
                        and parts[0] == "model"
                        and parts[2] == "weight"):
                        scale2x += 1
                    if part_num > n_uplayer:
                        n_uplayer = part_num
                        out_nc = state_dict[block].shape[0]
                if not plus and "conv1x1" in block:
                    plus = True

            nf = state_dict["model.0.weight"].shape[0]
            in_nc = state_dict["model.0.weight"].shape[1]
            out_nc = out_nc
            scale = 2 ** scale2x

            return in_nc, out_nc, nf, nb, plus, scale

        def load_model(path: str):
            filename = path
            #TODO Implement downloading of Ultrax4
            # if not os.path.exists(filename) or filename is None:
            #     filename = load_file_from_url(url=self.model_url, model_dir=self.model_path,
            #                                                 file_name="%s.pth" % self.model_name,
            #                                                 progress=True)

            state_dict = torch.load(filename)

            if "params_ema" in state_dict:
                state_dict = state_dict["params_ema"]
            elif "params" in state_dict:
                state_dict = state_dict["params"]
                num_conv = 16 if "realesr-animevideov3" in filename else 32
                model = arch.SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=num_conv, upscale=4, act_type='prelu')
                model.load_state_dict(state_dict)
                model.eval()
                return model

            if "body.0.rdb1.conv1.weight" in state_dict and "conv_first.weight" in state_dict:
                nb = 6 if "RealESRGAN_x4plus_anime_6B" in filename else 23
                state_dict = resrgan2normal(state_dict, nb)
            elif "conv_first.weight" in state_dict:
                state_dict = mod2normal(state_dict)
            elif "model.0.weight" not in state_dict:
                raise Exception("The file is not a recognized ESRGAN model.")

            in_nc, out_nc, nf, nb, plus, mscale = infer_params(state_dict)

            model = arch.RRDBNet(in_nc=in_nc, out_nc=out_nc, nf=nf, nb=nb, upscale=mscale, plus=plus)
            model.load_state_dict(state_dict)
            model.eval()

            return model
        
        use_half = get_gpu_architecture() == 'NVIDIA'
        suffix = "upscaled_e"

        def upscale_without_tiling(model, img):
            img = np.array(img)
            img = img[:, :, ::-1]
            img = np.ascontiguousarray(np.transpose(img, (2, 0, 1))) / 255
            img = torch.from_numpy(img).float()
            img = img.unsqueeze(0).to('cuda')
            model = model.to('cuda')
            with torch.no_grad():
                output = model(img)
            output = output.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = 255. * np.moveaxis(output, 0, 2)
            output = output.astype(np.uint8)
            output = output[:, :, ::-1]
            return Image.fromarray(output, 'RGB')

        # restorer
        def upscale(model, img, scale_factor, tile=192, tile_overlap=8):
            img = Image.fromarray(img, 'RGB')
            grid = split_grid(img, tile, tile, tile_overlap)
            newtiles = []
            scale_factor=1
            for y, h, row in grid.tiles:
                newrow = []
                for tiledata in row:
                    x, w, tile = tiledata

                    output = upscale_without_tiling(model, tile)
                    scale_factor = output.width // tile.width

                    newrow.append([x * scale_factor, w * scale_factor, output])
                newtiles.append([y * scale_factor, h * scale_factor, newrow])

            newgrid = Grid(newtiles, grid.tile_w * scale_factor, grid.tile_h * scale_factor, grid.image_w * scale_factor, grid.image_h * scale_factor, grid.overlap * scale_factor)
            output = combine_grid(newgrid)
            return np.array(output) 
        
        model = load_model(filename)
        output_images = []
        save_paths = []
        for path in self.images:
            imgname, extension = os.path.splitext(os.path.basename(path))
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            try:
                output = upscale(model, img, scale_factor=upscale_factor)
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
