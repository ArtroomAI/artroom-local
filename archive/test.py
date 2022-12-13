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

    #Set up settings
userprofile = os.environ["USERPROFILE"]
os.makedirs(f"{userprofile}/artroom/settings/",exist_ok=True)
os.makedirs(f"{userprofile}/artroom/model_weights/",exist_ok=True)
os.makedirs(f"{userprofile}/artroom/upscalers/",exist_ok=True)
os.makedirs(f"{userprofile}/artroom/model_weights/upscalers",exist_ok=True)

if not os.path.exists(f"{userprofile}/artroom/upscalers/GFPGAN"):
    git.Repo.clone_from("https://github.com/TencentARC/GFPGAN.git",f"{userprofile}/artroom/upscalers/GFPGAN")

if not os.path.exists(f"{userprofile}/artroom/upscalers/GFPGAN"):
    git.Repo.clone_from("https://github.com/TencentARC/GFPGAN.git",f"{userprofile}/artroom/upscalers/GFPGAN")