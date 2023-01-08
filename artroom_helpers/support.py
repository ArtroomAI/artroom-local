import base64
import os
import io
from PIL import Image, ImageChops
from PIL.Image import Image as ImageType
from typing import Union, Literal

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def image_to_b64(image):
    image_file = io.BytesIO()
    image.save(image_file, format='JPEG')
    im_bytes = image_file.getvalue()  # im_bytes: image in binary format.
    imgb64 = base64.b64encode(im_bytes)
    return 'data:image/jpeg;base64,' + str(imgb64)[2:-1]


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
"""
Saves a thumbnail of an image, returning its path.
"""

def save_thumbnail(
    image: ImageType,
    filename: str,
    path: str,
    size: int = 256,
) -> str:
    base_filename = os.path.splitext(filename)[0]
    thumbnail_path = os.path.join(path, base_filename + ".webp")

    if os.path.exists(thumbnail_path):
        return thumbnail_path

    thumbnail_width = size
    thumbnail_height = round(size * (image.height / image.width))

    image_copy = image.copy()
    image_copy.thumbnail(size=(thumbnail_width, thumbnail_height))

    image_copy.save(thumbnail_path, "WEBP")

    return thumbnail_path

# https://stackoverflow.com/questions/43864101/python-pil-check-if-image-is-transparent
def check_for_any_transparency(img: Union[ImageType, str]) -> bool:
    if type(img) is str:
        img = Image.open(str)

    if img.info.get("transparency", None) is not None:
        return True
    if img.mode == "P":
        transparent = img.info.get("transparency", -1)
        for _, index in img.getcolors():
            if index == transparent:
                return True
    elif img.mode == "RGBA":
        extrema = img.getextrema()
        if extrema[3][0] < 255:
            return True
    return False

def get_canvas_generation_mode(
    init_img: Union[ImageType, str], init_mask: Union[ImageType, str]
) -> Literal["txt2img", "outpainting", "inpainting", "img2img",]:
    if type(init_img) is str:
        init_img = Image.open(init_img)

    if type(init_mask) is str:
        init_mask = Image.open(init_mask)

    init_img = init_img.convert("RGBA")

    # Get alpha from init_img
    init_img_alpha = init_img.split()[-1]
    init_img_alpha_mask = init_img_alpha.convert("L")
    init_img_has_transparency = check_for_any_transparency(init_img)

    if init_img_has_transparency:
        init_img_is_fully_transparent = (
            True if init_img_alpha_mask.getbbox() is None else False
        )

    """
    Mask images are white in areas where no change should be made, black where changes
    should be made.
    """

    # Fit the mask to init_img's size and convert it to greyscale
    init_mask = init_mask.resize(init_img.size).convert("L")

    """
    PIL.Image.getbbox() returns the bounding box of non-zero areas of the image, so we first
    invert the mask image so that masked areas are white and other areas black == zero.
    getbbox() now tells us if the are any masked areas.
    """
    init_mask_bbox = ImageChops.invert(init_mask).getbbox()
    init_mask_exists = False if init_mask_bbox is None else True

    if init_img_has_transparency:
        if init_img_is_fully_transparent:
            return "txt2img"
        else:
            return "outpainting"
    else:
        if init_mask_exists:
            return "inpainting"
        else:
            return "img2img"
