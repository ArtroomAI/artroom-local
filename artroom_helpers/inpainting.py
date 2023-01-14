from artroom_helpers import patchmatch
from PIL import Image, ImageFilter, ImageOps
import numpy as np 
import math
import cv2

def mask_edge(mask: Image, edge_size: int, edge_blur: int) -> Image:
    npimg = np.asarray(mask, dtype=np.uint8)

    # Detect any partially transparent regions
    npgradient = np.uint8(255 * (1.0 - np.floor(np.abs(0.5 - np.float32(npimg) / 255.0) * 2.0)))

    # Detect hard edges
    npedge = cv2.Canny(npimg, threshold1=100, threshold2=200)

    # Combine
    npmask = npgradient + npedge

    # Expand
    npmask = cv2.dilate(npmask, np.ones((3,3), np.uint8), iterations = int(edge_size / 2))

    new_mask = Image.fromarray(npmask)

    if edge_blur > 0:
        new_mask = new_mask.filter(ImageFilter.BoxBlur(edge_blur))

    return ImageOps.invert(new_mask)

class Outpaint(object):
    def __init__(self, image, generate):
        self.image     = image
        self.generate  = generate

    def process(self, opt, old_opt, image_callback = None, prefix = None):
        image = self._create_outpaint_image(self.image, opt.out_direction)

        seed   = old_opt.seed
        prompt = old_opt.prompt

        def wrapped_callback(img,seed,**kwargs):
            image_callback(img,seed,use_prefix=prefix,**kwargs)


        return self.generate.prompt2image(
            prompt,
            seed           = seed,
            sampler        = self.generate.sampler,
            steps          = opt.steps,
            cfg_scale      = opt.cfg_scale,
            ddim_eta       = self.generate.ddim_eta,
            width          = opt.width,
            height         = opt.height,
            init_img       = image,
            strength       = 0.83,
            image_callback = wrapped_callback,
            prefix         = prefix,
        )

    def _create_outpaint_image(self, image, direction_args):
        assert len(direction_args) in [1, 2], 'Direction (-D) must have exactly one or two arguments.'

        if len(direction_args) == 1:
            direction = direction_args[0]
            pixels = None
        elif len(direction_args) == 2:
            direction = direction_args[0]
            pixels = int(direction_args[1])

        assert direction in ['top', 'left', 'bottom', 'right'], 'Direction (-D) must be one of "top", "left", "bottom", "right"'

        image = image.convert("RGBA")
        # we always extend top, but rotate to extend along the requested side
        if direction == 'left':
            image = image.transpose(Image.Transpose.ROTATE_270)
        elif direction == 'bottom':
            image = image.transpose(Image.Transpose.ROTATE_180)
        elif direction == 'right':
            image = image.transpose(Image.Transpose.ROTATE_90)

        pixels = image.height//2 if pixels is None else int(pixels)
        assert 0 < pixels < image.height, 'Direction (-D) pixels length must be in the range 0 - image.size'

        # the top part of the image is taken from the source image mirrored
        # coordinates (0,0) are the upper left corner of an image
        top = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM).convert("RGBA")
        top = top.crop((0, top.height - pixels, top.width, top.height))

        # setting all alpha of the top part to 0
        alpha = top.getchannel("A")
        alpha.paste(0, (0, 0, top.width, top.height))
        top.putalpha(alpha)

        # taking the bottom from the original image
        bottom = image.crop((0, 0, image.width, image.height - pixels))

        new_img = image.copy()
        new_img.paste(top, (0, 0))
        new_img.paste(bottom, (0, pixels))

        # create a 10% dither in the middle
        dither = min(image.height//10, pixels)
        for x in range(0, image.width, 2):
            for y in range(pixels - dither, pixels + dither):
                (r, g, b, a) = new_img.getpixel((x, y))
                new_img.putpixel((x, y), (r, g, b, 0))

        # let's rotate back again
        if direction == 'left':
            new_img = new_img.transpose(Image.Transpose.ROTATE_90)
        elif direction == 'bottom':
            new_img = new_img.transpose(Image.Transpose.ROTATE_180)
        elif direction == 'right':
            new_img = new_img.transpose(Image.Transpose.ROTATE_270)

        return new_img

def infill_patchmatch(im: Image.Image) -> Image:
    if im.mode != 'RGBA':
        print("Patchmatch failed, not RGBA")
        return im

    # Skip patchmatch if patchmatch isn't available
    if not patchmatch.patchmatch_available:
        print("patchmatch not available")
        return im
    
    # Patchmatch (note, we may want to expose patch_size? Increasing it significantly impacts performance though)
    im_patched_np = patchmatch.inpaint(im.convert('RGB'), ImageOps.invert(im.split()[-1]), patch_size = 3)
    im_patched = Image.fromarray(im_patched_np, mode = 'RGB')

    return im_patched

def get_tile_images(image: np.ndarray, width=8, height=8):
    _nrows, _ncols, depth = image.shape
    _strides = image.strides

    nrows, _m = divmod(_nrows, height)
    ncols, _n = divmod(_ncols, width)
    if _m != 0 or _n != 0:
        return None

    return np.lib.stride_tricks.as_strided(
        np.ravel(image),
        shape=(nrows, ncols, height, width, depth),
        strides=(height * _strides[0], width * _strides[1], *_strides),
        writeable=False
    )

def tile_fill_missing(im: Image.Image, tile_size: int = 16, seed: int = None) -> Image:
    # Only fill if there's an alpha layer
    if im.mode != 'RGBA':
        return im

    a = np.asarray(im, dtype=np.uint8)

    tile_size = (tile_size, tile_size)

    # Get the image as tiles of a specified size
    tiles = get_tile_images(a,*tile_size).copy()

    # Get the mask as tiles
    tiles_mask = tiles[:,:,:,:,3]

    # Find any mask tiles with any fully transparent pixels (we will be replacing these later)
    tmask_shape = tiles_mask.shape
    tiles_mask = tiles_mask.reshape(math.prod(tiles_mask.shape))
    n,ny = (math.prod(tmask_shape[0:2])), math.prod(tmask_shape[2:])
    tiles_mask = (tiles_mask > 0)
    tiles_mask = tiles_mask.reshape((n,ny)).all(axis = 1)

    # Get RGB tiles in single array and filter by the mask
    tshape = tiles.shape
    tiles_all = tiles.reshape((math.prod(tiles.shape[0:2]), * tiles.shape[2:]))
    filtered_tiles = tiles_all[tiles_mask]

    if len(filtered_tiles) == 0:
        return im

    # Find all invalid tiles and replace with a random valid tile
    replace_count = (tiles_mask == False).sum()
    rng = np.random.default_rng(seed = seed)
    tiles_all[np.logical_not(tiles_mask)] = filtered_tiles[rng.choice(filtered_tiles.shape[0], replace_count),:,:,:]

    # Convert back to an image
    tiles_all = tiles_all.reshape(tshape)
    tiles_all = tiles_all.swapaxes(1,2)
    st = tiles_all.reshape((math.prod(tiles_all.shape[0:2]), math.prod(tiles_all.shape[2:4]), tiles_all.shape[4]))
    si = Image.fromarray(st, mode='RGBA')

    return si
