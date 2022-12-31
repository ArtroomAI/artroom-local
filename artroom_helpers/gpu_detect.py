import torch

def is_16xx_series():
    if torch.cuda.is_available():
        gpu_info = torch.cuda.get_device_name(0)
        if '1630' in gpu_info or '1650' in gpu_info or '1660' in gpu_info or '1600' in gpu_info:
            print(gpu_info + ' identified, forcing to full precision')
            return '16XX'
        return 'NVIDIA'
    else:
        print("Cuda not available.")
        print("If you are using NVIDIA GPU please try updating your drivers")
        print("If you are using AMD, it is not yet supported.")
        return 'None'