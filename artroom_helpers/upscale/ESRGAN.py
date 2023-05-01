import os
from artroom_helpers.gpu_detect import get_gpu_architecture
import torch 


def ESRGAN(self, upscaler, upscale_factor, upscale_dest):
    import artroom_helpers.upscale.esrgan_model_arch as arch

    def load_model(self, path: str):
        if "http" in path:
            filename = load_file_from_url(url=self.model_url, model_dir=self.model_path,
                                        file_name="%s.pth" % self.model_name,
                                        progress=True)
        else:
            filename = path
        if not os.path.exists(filename) or filename is None:
            print("Unable to load %s from %s" % (self.model_path, filename))
            return None

        state_dict = torch.load(filename, map_location='cpu')

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

    # restorer
    grid = images.split_grid(img, opts.ESRGAN_tile, opts.ESRGAN_tile, opts.ESRGAN_tile_overlap)
    newtiles = []
    scale_factor = 1

    for y, h, row in grid.tiles:
        newrow = []
        for tiledata in row:
            x, w, tile = tiledata

            output = upscale_without_tiling(model, tile)
            scale_factor = output.width // tile.width

            newrow.append([x * scale_factor, w * scale_factor, output])
        newtiles.append([y * scale_factor, h * scale_factor, newrow])

    newgrid = images.Grid(newtiles, grid.tile_w * scale_factor, grid.tile_h * scale_factor, grid.image_w * scale_factor, grid.image_h * scale_factor, grid.overlap * scale_factor)
    output = images.combine_grid(newgrid)


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
