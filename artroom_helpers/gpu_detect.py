import torch

def is_16xx_series():
    try:
        gpu_info = torch.cuda.get_device_name(0)
        if '1630' in gpu_info or '1650' in gpu_info or '1660' in gpu_info or '1600' in gpu_info:
            print(gpu_info + ' identified, forcing to full precision')
            return 1
        return 0
    except:
        return 2