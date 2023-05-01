import os
from basicsr.utils.download_util import load_file_from_url
from artroom_helpers.upscale import ESRGAN, RealESRGAN, GFPGAN

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
                output_images, save_paths = GFPGAN(upscaler, upscale_factor, upscale_dest)
            elif "RealESRGAN" in upscaler:
                output_images, save_paths = RealESRGAN(upscaler, upscale_factor, upscale_dest)
            else:
                output_images, save_paths = ESRGAN(upscaler, upscale_factor, upscale_dest)

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