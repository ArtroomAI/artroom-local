import json
import requests
import random 

server_ip = 'localhost'
run_job_url = f'http://{server_ip}:5300/xyplot'

x_key = "text_prompts"
x_values = ["8k", "best quality", "digital art", "highest detail", "highly detailed", "intricate detail", "masterpiece", "photo realistic", "trending on artstation", ]
y_key = "ckpt"
y_values = ["sd_xl_base_0.9.safetensors", "ProtoGen_X3.4-pruned-fp16.safetensors", "v1-5-pruned.ckpt"]

# X/Y candidates
clip_skip = 1
cfg_scale = 7.5
steps = 30
sampler = "euler_a" #"ddim", "dpmpp_2m", "dpmpp_2s_ancestral", "euler", "euler_a", "dpm_2", "dpm_a", "lms", "heun", "plms"
remove_background = "none" # "face", "u2net", "u2net_human_seg"
controlnet = "none" # "none", "canny", "pose", "depth", "hed", "normal", "scribble"
vae = "vae_rmada-cold.vae.pt" #"none"
loras = [] 
"""
Format for lora:
[
    { 
        "name" : "Character/JinxLol.safetensors",
        "weight" : 0.6,
        "trigger" : "jinxlol"
    },
    { 
        "name" : ...
        "weight" : ...,
        "trigger" : ...
    }
]
"""
ckpt = "sd_xl_base_0.9.safetensors"
width = 1024
height = 1024
strength = 0.65

# Probably Fixed Parameters 
n_iter = 1 # Probably makes the most sense to keep this at 1
seed = 5   # Probably makes sense to keep this fixed
text_prompts = "a cat ***"
negative_prompts = "mutated, deformed, amateur drawing, lowres, worst quality, low quality, jpeg artifacts, text, error, signature, watermark, username, blurry, censorship"
init_image = ""
mask_image = ""

# General Configs, probably just set once:
models_dir = "E:\\Model_Weights"
image_save_path = "E:\\Artroom\\ArtroomOuputs"
show_intermediates = False 
highres_fix = True 
long_save_path = False 
speed = "High"
save_grid = False
invert = False
use_removed_background = False 
use_preprocessed_controlnet = False

payloads = []

def update_payload(payload, key, value):
    if key == "text_prompts" or key == "negative_prompts":
        payload[key] = payload[key].replace("***", str(x_value)).replace("@@@", str(y_value))
    elif key == "lora":
        value["trigger"] + text_prompts
        payload["text_prompts"] = f'{value["trigger"]}, {payload["text_prompts"]}'
        payload[key] = payload[key] + [value]
    else:
        payload[key] = value
    return payload 

for y_value in y_values:
    for x_value in x_values:
        payload = {
            "clip_skip": clip_skip, 
            "cfg_scale": cfg_scale,
            "steps": steps,
            "sampler": sampler,
            "remove_background": remove_background,
            "controlnet": controlnet,
            "vae": vae,
            "loras": loras,
            "ckpt":ckpt,
            "width": width,
            "height": height,
            "strength": strength,
            "text_prompts": text_prompts,
            "negative_prompts": negative_prompts,
            "seed": seed,
            "init_image": init_image,
            "mask_image": mask_image,
            "invert": invert,
            "use_removed_background": use_removed_background,
            "use_preprocessed_controlnet": use_preprocessed_controlnet,
            "image_save_path": image_save_path,
            "n_iter": n_iter,
            "save_grid": save_grid,
            "speed": speed,
            "long_save_path": long_save_path,
            "highres_fix": highres_fix,
            "show_intermediates": show_intermediates,
            "models_dir": models_dir,
            "id": str(random.randint(1,100000)) # set a unique ID for each generate request
        }
        payload = update_payload(payload, x_key, x_value)
        payload = update_payload(payload, y_key, y_value)

        payloads.append(payload)

xypost = {
    "xyplots": payloads,
    "x_key": x_key,
    "x_values":  x_values,
    "y_key": y_key,
    "y_values": y_values
}
response = requests.post(run_job_url, json=xypost)