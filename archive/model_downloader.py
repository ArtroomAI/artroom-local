import os
import shutil
from tqdm import tqdm
import json
import time
import requests
import shutil
import ctypes
import git

kernel32 = ctypes.windll.kernel32
kernel32.SetConsoleMode(kernel32.GetStdHandle(-10), 128)

try:
    #Set up settings
    userprofile = os.environ["USERPROFILE"]
    os.makedirs(f"{userprofile}/artroom/settings/",exist_ok=True)
    os.makedirs(f"{userprofile}/artroom/model_weights/",exist_ok=True)
    os.makedirs(f"{userprofile}/artroom/upscalers/",exist_ok=True)
    os.makedirs(f"{userprofile}/artroom/model_weights/upscalers",exist_ok=True)

    if not os.path.exists(f"{userprofile}/artroom/upscalers/GFPGAN"):
        git.Repo.clone_from("https://github.com/TencentARC/GFPGAN.git",f"{userprofile}/artroom/upscalers/GFPGAN")
    if not os.path.exists(f"{userprofile}/artroom/upscalers/Real-ESRGAN"):
        git.Repo.clone_from("https://github.com/xinntao/Real-ESRGAN.git",f"{userprofile}/artroom/upscalers/Real-ESRGAN")
    if not os.path.exists(f"{userprofile}/artroom/upscalers/CodeFormer"):
        git.Repo.clone_from("https://github.com/sczhou/CodeFormer.git",f"{userprofile}/artroom/upscalers/CodeFormer")
    if not os.path.exists(f"{userprofile}/artroom/settings/upscale_settings.json"):
        shutil.copy("upscale_settings.json",f"{userprofile}/artroom/settings/")
    else:
        os.remove(f"{userprofile}/artroom/settings/upscale_settings.json")
        shutil.copy("upscale_settings.json",f"{userprofile}/artroom/settings/")

    if not os.path.exists(f"{userprofile}/artroom/settings/error_mode.json"):
        shutil.copy("error_mode.json",f"{userprofile}/artroom/settings/")
    else:
        os.remove(f"{userprofile}/artroom/settings/error_mode.json")
        shutil.copy("error_mode.json",f"{userprofile}/artroom/settings/")

    if not os.path.exists(f"{userprofile}/artroom/settings/sd_settings.json"):
        shutil.copy("sd_settings.json",f"{userprofile}/artroom/settings/")
    else:
        original_json = json.load(open(f"{userprofile}/artroom/settings/sd_settings.json"))
        original_settings = original_json['Settings']
        original_config = original_json['Config']
        new_json = json.load(open("sd_settings.json"))
        new_settings = new_json['Settings']
        new_config = new_json['Config']
        for key in new_settings:
            if key not in original_settings:
                original_settings[key] = new_settings[key]
        for key in new_config:
            if key not in original_config:
                original_config[key] = new_config[key]
        update_original = {
            "Settings": original_settings,
            "Config": original_config
        }
        with open(f"{userprofile}/artroom/settings/sd_settings.json", "w") as outfile:
            json.dump(update_original, outfile, indent=4)

    try:
        queue_json = json.load(open(f'{userprofile}/artroom/settings/queue.json'))
    except:
        queue_json = {
            "Queue": []
        }
    with open(f'{userprofile}/artroom/settings/queue.json', 'w') as outfile:
        json.dump(queue_json, outfile, indent=4)
        
    upscale_models = {
        "RealESRGANx2": {
            "model_source": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            "model_dest": f"{userprofile}/artroom/upscalers/Real-ESRGAN/weights/"
        },
        "RealESRGAN": {
            "model_source": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth", 
            "model_dest": f"{userprofile}/artroom/upscalers/Real-ESRGAN/experiments/pretrained_models/"
        },
        "RealESRGAN-anime": {
            "model_source": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth", 
            "model_dest": f"{userprofile}/artroom/upscalers/Real-ESRGAN/experiments/pretrained_models/"
        },
        "GFPGANv1.3": {
            "model_source": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth", 
            "model_dest": f"{userprofile}/artroom/upscalers/GFPGAN/experiments/pretrained_models/"
        },
        "GFPGANv1.4": {
            "model_source": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth", 
            "model_dest": f"{userprofile}/artroom/upscalers/GFPGAN/experiments/pretrained_models/"
        },
        "RestoreFormer": {
            "model_source": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth", 
            "model_dest": f"{userprofile}/artroom/upscalers/GFPGAN/experiments/pretrained_models/"
        },
        "detection_Resnet50-lib": {
            "model_source": "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth", 
            "model_dest":f"{userprofile}/artroom/miniconda3/envs/artroom-ldm/lib/site-packages/facexlib/weights/"
        },  
        "parsing_parsenet-lib":{
            "model_source": "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
            "model_dest":f"{userprofile}/artroom/miniconda3/envs/artroom-ldm/lib/site-packages/facexlib/weights/"
        },
        "detection_Resnet50": {
            "model_source": "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth", 
            "model_dest": f"{userprofile}/artroom/upscalers/GFPGAN/gfpgan/weights/"
        },  
        "parsing_parsenet":{
            "model_source": "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
            "model_dest": f"{userprofile}/artroom/upscalers/GFPGAN/gfpgan/weights/"
        },
        "Codeformer": {
            "model_source": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth", 
            "model_dest": f"{userprofile}/artroom/upscalers/CodeFormer/weights/CodeFormer/"
        },
        "detection_Resnet50-CodeFormer": {
            "model_source": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth", 
            "model_dest": f"{userprofile}/artroom/upscalers/CodeFormer/weights/facelib/"
        },  
        "parsing_parsenet-CodeFormer":{
            "model_source": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth",
            "model_dest": f"{userprofile}/artroom/upscalers/CodeFormer/weights/facelib/"
        },
    }

    for model in upscale_models:
        url = upscale_models[model]["model_source"]
        model_dest = upscale_models[model]["model_dest"]
        model_name = os.path.basename(url)
        if os.path.exists(model_dest+model_name):
            continue
        if os.path.exists(f"{userprofile}/artroom/"+model_name):
            os.remove(f"{userprofile}/artroom/"+model_name)
        if os.path.exists(f"{userprofile}/artroom/model_weights/upscalers/{model_name}"):
            shutil.move(f"{userprofile}/artroom/model_weights/upscalers/{model_name}", model_dest)
        if not os.path.exists(model_dest+model_name):
            print(f"Downloading {model_name}...")
            response = requests.get(url, stream=True)
            total_size_in_bytes= int(response.headers.get('content-length', 0))
            block_size = 1024 #1 Kibibyte
            progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
            with open(f"{userprofile}/artroom/"+model_name, 'wb') as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
            progress_bar.close()
            if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                print("ERROR, something went wrong")
            shutil.move(f"{userprofile}/artroom/{model_name}", model_dest)

    model_dl_json = json.load(open("model_downloader.json"))

    possible_sources = model_dl_json["possible_sources"]
    sd_path = model_dl_json["sd_path"]
    model_path = model_dl_json["model_path"]
    model_name = model_dl_json["model_name"]
    userprofile = os.environ["USERPROFILE"]

    if "%UserProfile%" in model_path:
        model_path = model_path.replace("%UserProfile%",userprofile)

    sd_json = json.load(open(f"{userprofile}/artroom/settings/sd_settings.json"))
    sd_settings = sd_json['Settings']
    sd_config = sd_json['Config']

    custom_model_path = sd_config['model_ckpt']
    if "%UserProfile%" in custom_model_path:
        custom_model_path = custom_model_path.replace("%UserProfile%",userprofile)

    #Remove broken download if failed during install
    if os.path.exists(f"{userprofile}/artroom/"+"model.ckpt"):
        os.remove(f"{userprofile}/artroom/"+"model.ckpt")
    os.makedirs(model_path, exist_ok=True)  

    if os.path.exists(sd_path):
        if os.path.exists(model_path+"/model.ckpt") or os.path.exists(custom_model_path):
            print("Model found")
        else:
            print("Model not found")
            print("Downloading Model weights..... (if it freezes, try pressing enter)")
            # url = 'https://drive.google.com/uc?id=1W6nSS8at4kjO4ekAGJeFRwnRKIaLfDwS'
            # gdown.download(url, model_name, quiet=False) 
            # shutil.move(model_name,model_path)

            for url in possible_sources:
                try:
                    response = requests.get(url, stream=True)
                    total_size_in_bytes= int(response.headers.get('content-length', 0))
                    block_size = 1024 #1 Kibibyte
                    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
                    with open(f"{userprofile}/artroom/"+"model.ckpt", 'wb') as file:
                        for data in response.iter_content(block_size):
                            progress_bar.update(len(data))
                            file.write(data)
                    progress_bar.close()
                    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                        print("ERROR, something went wrong")              
                    shutil.move(f"{userprofile}/artroom/"+"model.ckpt",model_path)
                except:
                    pass
    os.chdir(f"{userprofile}/artroom/upscalers/CodeFormer")
    os.system("python basicsr/setup.py develop > artroomlogs.txt 2>&1'") 
except Exception as e:
    print(f"Failed downloading models {e}")
    with open("artroomlog.txt", "a") as logs:# append mode
        logs.write(f"Failed downloading models {e}")
    time.sleep(30)
