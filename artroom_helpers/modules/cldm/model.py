import os
import torch

from omegaconf import OmegaConf

from artroom_helpers.modules.cldm.cl_ldm.util import instantiate_from_config
from sd_modules.optimizedSD.ddpm import DiffusionWrapperv2


def get_state_dict(d):
    return d.get('state_dict', d)


def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict


def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    model.model = DiffusionWrapperv2({
        "target": "cldm.cldm.ControlledUnetModel",
        "params": {
            "image_size": 32,
            "in_channels": 4,
            "out_channels": 4,
            "model_channels": 320,
            "attention_resolutions": [4, 2, 1],
            "num_res_blocks": 2,
            "channel_mult": [1, 2, 4, 4],
            "num_heads": 8,
            "use_spatial_transformer": True,
            "transformer_depth": 1,
            "context_dim": 768,
            "use_checkpoint": True,
            "legacy": False
        }
    }, "crossattn")

    print(f'Loaded model config from [{config_path}]')
    return model
