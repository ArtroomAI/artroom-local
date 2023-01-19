import base64
import io
import cv2
from PIL import Image, ImageChops, ImageFilter
from PIL.Image import Image as ImageType
import numpy as np
from io import BytesIO
import re 

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def image_to_b64(image):
    image_file = io.BytesIO()
    image.save(image_file, format='JPEG')
    im_bytes = image_file.getvalue()  # im_bytes: image in binary format.
    imgb64 = base64.b64encode(im_bytes)
    return 'data:image/jpeg;base64,' + str(imgb64)[2:-1]

def b64_to_image(b64):
    image_data = re.sub('^data:image/.+;base64,', '', b64)
    return Image.open(BytesIO(base64.b64decode(image_data)))

def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )

def dataURL_to_bytes(dataURL: str) -> bytes:
    return base64.decodebytes(
        bytes(
            dataURL.split(",", 1)[1],
            "utf-8",
        )
    )

def dataURL_to_image(dataURL: str) -> ImageType:
    image = Image.open(
        io.BytesIO(
            base64.decodebytes(
                bytes(
                    dataURL.split(",", 1)[1],
                    "utf-8",
                )
            )
        )
    )
    return image

def repaste_and_color_correct(result: Image.Image, init_image: Image.Image, init_mask: Image.Image, mask_blur_radius: int = 8) -> Image.Image:
    if init_image is None or init_mask is None:
        return result
    
    # Get the original alpha channel of the mask if there is one.
    # Otherwise it is some other black/white image format ('1', 'L' or 'RGB')
    pil_init_mask = init_mask.getchannel('A') if init_mask.mode == 'RGBA' else init_mask.convert('L')
    pil_init_image = init_image.convert('RGBA') # Add an alpha channel if one doesn't exist

    # Build an image with only visible pixels from source to use as reference for color-matching.
    init_rgb_pixels = np.asarray(init_image.convert('RGB'), dtype=np.uint8)
    init_a_pixels = np.asarray(pil_init_image.getchannel('A'), dtype=np.uint8)
    init_mask_pixels = np.asarray(pil_init_mask, dtype=np.uint8)

    # Get numpy version of result
    np_image = np.asarray(result, dtype=np.uint8)

    # Mask and calculate mean and standard deviation
    mask_pixels = init_a_pixels * init_mask_pixels > 0
    np_init_rgb_pixels_masked = init_rgb_pixels[mask_pixels, :]
    np_image_masked = np_image[mask_pixels, :]

    if np_init_rgb_pixels_masked.size > 0:
        init_means = np_init_rgb_pixels_masked.mean(axis=0)
        init_std = np_init_rgb_pixels_masked.std(axis=0)
        gen_means = np_image_masked.mean(axis=0)
        gen_std = np_image_masked.std(axis=0)

        # Color correct
        np_matched_result = np_image.copy()
        np_matched_result[:,:,:] = (((np_matched_result[:,:,:].astype(np.float32) - gen_means[None,None,:]) / gen_std[None,None,:]) * init_std[None,None,:] + init_means[None,None,:]).clip(0, 255).astype(np.uint8)
        matched_result = Image.fromarray(np_matched_result, mode='RGB')
    else:
        matched_result = Image.fromarray(np_image, mode='RGB')

    # Blur the mask out (into init image) by specified amount
    if mask_blur_radius > 0:
        nm = np.asarray(pil_init_mask, dtype=np.uint8)
        nmd = cv2.erode(nm, kernel=np.ones((3,3), dtype=np.uint8), iterations=int(mask_blur_radius / 2))
        pmd = Image.fromarray(nmd, mode='L')
        blurred_init_mask = pmd.filter(ImageFilter.BoxBlur(mask_blur_radius))
    else:
        blurred_init_mask = pil_init_mask

    multiplied_blurred_init_mask = ImageChops.multiply(blurred_init_mask, pil_init_image.split()[-1])

    # Paste original on color-corrected generation (using blurred mask)
    matched_result.paste(init_image, (0,0), mask = multiplied_blurred_init_mask)
    return matched_result
