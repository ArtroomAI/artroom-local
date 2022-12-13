import subprocess
import json
import os
import time
import subprocess
from glob import glob
import shutil
import ctypes

kernel32 = ctypes.windll.kernel32
kernel32.SetConsoleMode(kernel32.GetStdHandle(-10), 128)

userprofile = os.environ["USERPROFILE"]
upscale_json = json.load(open(f"{userprofile}/artroom/settings/upscale_settings.json"))

upscale_images = upscale_json["upscale_images"]
upscaler = upscale_json["upscaler"]
upscale_factor = upscale_json["upscale_factor"]
upscale_dest = upscale_json["upscale_dest"]
upscale_strength = upscale_json["upscale_strength"]

#Filter non-images
upscale_images = [image for image in upscale_images if (".jpg" in image or ".png" in image or ".jpeg" in image)]

if upscale_dest == "":
    upscale_dest = os.path.dirname(upscale_images[0])

# images = glob(f"{upscale_images}/*.png") + glob(f"{upscale_images}/*.jpg") + glob(f"{upscale_images}/*.jpeg")
upscale_queue_path = f"{userprofile}/artroom/settings/upscale_queue"
if os.path.exists(upscale_queue_path):
    shutil.rmtree(upscale_queue_path)
    
os.makedirs(upscale_queue_path,exist_ok=True)
for image in upscale_images:    
    shutil.copy(image,upscale_queue_path)

if "GFPGAN" in upscaler or upscaler == "RestoreFormer":
    os.chdir(f"{userprofile}/artroom/upscalers/GFPGAN")
    #Potential error remover
    archs = glob("gfpgan/archs/*")
    for arch in archs:
        if "onnx" in arch:
            os.remove(arch)

    if "1.3" in upscaler:
        v = "1.3"
    elif "1.4" in upscaler:
        v = "1.4"
    else:
        v = upscaler
    script_command = [ f"{userprofile}\\artroom\\miniconda3\\condabin\\activate.bat","&&",
                    f"conda", "run","--no-capture-output", "-p", f"{userprofile}/artroom/miniconda3/envs/artroom-ldm",
                    "python", "inference_gfpgan.py","-i",f"{upscale_queue_path}","-o",f"{upscale_dest}/{upscaler}","-v",v,"-s",f"{upscale_factor}","--bg_upsampler","realesrgan","--suffix","_upscaled","-w",f"{upscale_strength}"
                    ]    
    print("Running....")
    print("If it freezes, please try pressing enter. Doesn't happen often but could happen once in a while")
    process = subprocess.run(script_command)
    print("Finished!")
    time.sleep(3)
elif upscaler == "RealESRGAN":
    os.chdir(f"{userprofile}/artroom/upscalers/Real-ESRGAN")
    script_command = [ f"{userprofile}\\artroom\\miniconda3\\condabin\\activate.bat","&&",
                    f"conda", "run","--no-capture-output", "-p", f"{userprofile}/artroom/miniconda3/envs/artroom-ldm",
                    "python", "inference_realesrgan.py","-i",f"{upscale_queue_path}","-o",f"{upscale_dest}/upscaled","-s",f"{upscale_factor}","--suffix","_upscaled","--tile","400"
                    ]    
    print("Running....")
    print("If it freezes, please try pressing enter. Doesn't happen often but could happen once in a while")
    process = subprocess.run(script_command)
    print("Finished!")
    time.sleep(3)
elif upscaler == "RealESRGAN-Anime":
    os.chdir(f"{userprofile}/artroom/upscalers/Real-ESRGAN")
    script_command = [ f"{userprofile}\\artroom\\miniconda3\\condabin\\activate.bat","&&",
                f"conda", "run","--no-capture-output", "-p", f"{userprofile}/artroom/miniconda3/envs/artroom-ldm",
                "python", "inference_realesrgan.py","-i",f"{upscale_queue_path}","-o",f"{upscale_dest}/upscaled-anime","-s",f"{upscale_factor}","--suffix","_upscaled","--tile","400",
                "--model", "RealESRGAN_x4plus_anime_6B"
                ]    
    print("Running....")
    print("If it freezes, please try pressing enter. Doesn't happen often but could happen once in a while")
    process = subprocess.run(script_command)
    print("Finished!")
    time.sleep(3)   
elif upscaler == "CodeFormer":
    os.chdir(f"{userprofile}/artroom/upscalers/CodeFormer/")
    script_command = [ f"{userprofile}\\artroom\\miniconda3\\condabin\\activate.bat","&&",
                f"conda", "run","--no-capture-output", "-p", f"{userprofile}/artroom/miniconda3/envs/artroom-ldm",
                "python", "inference_codeformer.py","--test_path",f"{upscale_queue_path}","--upscale",f"{upscale_factor}",
                "--bg_tile","400","--face_upsample", "--w", f"{upscale_strength}"
                ]    
    print("Running....")
    print("If it freezes, please try pressing enter. Doesn't happen often but could happen once in a while")
    process = subprocess.run(script_command)
    results = glob("results/*")
    for result in results:
        os.makedirs(f"{upscale_dest}/CodeFormer/",exist_ok=True)
        shutil.move(result,f"{upscale_dest}/CodeFormer/")
    print("Finished!")
    time.sleep(3)   
else:
    print("FAILURE")
    time.sleep(10)

#Clean up
shutil.rmtree(upscale_queue_path)