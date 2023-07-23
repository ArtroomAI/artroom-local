from contextlib import contextmanager
from copy import copy
from typing import Any

import torch

from adetailer import (
    __version__,
    get_models,
    ultralytics_predict,
)
from adetailer.args import BBOX_SORTBY, ADetailerArgs
from adetailer.common import PredictOutput
from adetailer.mask import filter_by_ratio, mask_preprocess, sort_bboxes
from sd_webui import safe
from sd_webui.shared import cmd_opts, opts

no_huggingface = getattr(cmd_opts, "ad_no_huggingface", False)
adetailer_dir = "."
model_mapping = get_models(adetailer_dir, huggingface=not no_huggingface)
SCRIPT_DEFAULT = "dynamic_prompting,dynamic_thresholding,wildcard_recursive,wildcards,lora_block_weight"


@contextmanager
def change_torch_load():
    orig = torch.load
    try:
        torch.load = safe.unsafe_torch_load
        yield
    finally:
        torch.load = orig


@contextmanager
def pause_total_tqdm():
    orig = opts.data.get("multiple_tqdm", True)
    try:
        opts.data["multiple_tqdm"] = False
        yield
    finally:
        opts.data["multiple_tqdm"] = orig


@contextmanager
def preseve_prompts(p):
    all_pt = copy(p.all_prompts)
    all_ng = copy(p.all_negative_prompts)
    try:
        yield
    finally:
        p.all_prompts = all_pt
        p.all_negative_prompts = all_ng


class AfterDetailerScript:
    def __init__(self, device=torch.device(0)):
        super().__init__()
        self.device = torch.device(device)

        self.controlnet_ext = None
        self.cn_script = None
        self.cn_latest_network = None
        self.predictor = ultralytics_predict
        self.models_choices = {
            0: 'face_yolov8n.pt',
            1: 'face_yolov8s.pt',
            2: 'hand_yolov8n.pt',
            3: 'person_yolov8n-seg.pt',
            4: 'person_yolov8s-seg.pt'
        }

    def __repr__(self):
        return f"{self.__class__.__name__}(version={__version__})"

    def get_ad_model(self, name: str):
        if name not in model_mapping:
            msg = f"[-] ADetailer: Model {name!r} not found. Available models: {list(model_mapping.keys())}"
            raise ValueError(msg)
        return model_mapping[name]

    def sort_bboxes(self, pred: PredictOutput) -> PredictOutput:
        sortby = opts.data.get("ad_bbox_sortby", BBOX_SORTBY[0])
        sortby_idx = BBOX_SORTBY.index(sortby)
        return sort_bboxes(pred, sortby_idx)

    def pred_preprocessing(self, pred: PredictOutput, args: ADetailerArgs):
        pred = filter_by_ratio(
            pred, low=args.ad_mask_min_ratio, high=args.ad_mask_max_ratio
        )
        pred = self.sort_bboxes(pred)
        return mask_preprocess(
            pred.masks,
            kernel=args.ad_dilate_erode,
            x_offset=args.ad_x_offset,
            y_offset=args.ad_y_offset,
            merge_invert=args.ad_mask_merge_invert,
        )

    @staticmethod
    def ensure_rgb_image(image: Any):
        if hasattr(image, "mode") and image.mode != "RGB":
            image = image.convert("RGB")
        return image


if __name__ == '__main__':
    processor = AfterDetailerScript()
