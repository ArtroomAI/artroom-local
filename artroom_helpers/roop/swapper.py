import os.path

import cv2
import insightface
import base64
import io
import urllib

import numpy as np

from tqdm import tqdm
from PIL import Image


class FaceSwapper:
    def __init__(self, cache_path=os.path.expanduser('~') + "/.cache/roop_cache/"):
        model_path = "inswapper_128.onnx"
        model_link = "https://huggingface.co/henryruhs/roop/resolve/main/" + model_path
        os.makedirs(cache_path, exist_ok=True)
        model_path_full = os.path.join(cache_path, model_path)
        if "inswapper_128.onnx" not in os.listdir(cache_path):
            self.download(model_link, model_path_full)
        self.model = insightface.model_zoo.get_model(model_path_full, providers=["CPUExecutionProvider"])

    def download(self, url, path):
        request = urllib.request.urlopen(url)
        total = int(request.headers.get('Content-Length', 0))
        with tqdm(total=total, desc='Downloading', unit='B', unit_scale=True, unit_divisor=1024) as progress:
            urllib.request.urlretrieve(url, path,
                                       reporthook=lambda count, block_size, total_size: progress.update(block_size))

    def get_face_single(self, img_data: np.ndarray, face_index=0, det_size=(640, 640)):
        face_analyser = insightface.app.FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        face_analyser.prepare(ctx_id=0, det_size=det_size)
        face = face_analyser.get(img_data)

        if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
            det_size_half = (det_size[0] // 2, det_size[1] // 2)
            return self.get_face_single(img_data, face_index=face_index, det_size=det_size_half)

        try:
            return sorted(face, key=lambda x: x.bbox[0])[face_index]
        except IndexError:
            return None

    def swap_face(
            self,
            source_img: Image.Image,
            target_img: Image.Image,
            faces_index=None,
    ):
        if faces_index is None:
            faces_index = {0}
        result_image = target_img
        if isinstance(source_img, str):  # source_img is a base64 string
            if 'base64,' in source_img:  # check if the base64 string has a data URL scheme
                base64_data = source_img.split('base64,')[-1]
                img_bytes = base64.b64decode(base64_data)
            else:
                # if no data URL scheme, just decode
                img_bytes = base64.b64decode(source_img)
            source_img = Image.open(io.BytesIO(img_bytes))
        source_img = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
        target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
        source_face = self.get_face_single(source_img, face_index=0)
        if source_face is not None:
            result = target_img
            face_swapper = self.model

            for face_num in faces_index:
                target_face = self.get_face_single(target_img, face_index=face_num)
                if target_face is not None:
                    result = face_swapper.get(result, target_face, source_face)
                else:
                    print(f"No target face found for {face_num}")

            result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        else:
            print("No source face found")
        return result_image
