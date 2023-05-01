from artroom_helpers.gpu_detect import get_gpu_architecture
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import cv2
import os 
from PIL import Image 

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
