import json
import os
import random
import re
from glob import glob
import ctypes

kernel32 = ctypes.windll.kernel32
kernel32.SetConsoleMode(kernel32.GetStdHandle(-10), 128)

userprofile = os.environ['USERPROFILE']

sd_json = json.load(open(f'{userprofile}/artroom/settings/sd_settings.json'))
sd_settings = sd_json['Settings']
sd_config = sd_json['Config']

if sd_config['image_save_path'][-1] == '/' or sd_config['image_save_path'][-1] == '\\':
    sd_config['image_save_path'] = sd_config['image_save_path'][:-1]
outdir = sd_config['image_save_path'] + '/' + sd_settings['batch_name']

if '%UserProfile%' in outdir:
    outdir = outdir.replace('%UserProfile%',userprofile)
if '%UserProfile%' in sd_config['model_ckpt']:
    sd_config['model_ckpt'] = sd_config['model_ckpt'].replace('%UserProfile%',userprofile)

if sd_settings['use_random_seed']:
    seed = random.randint(1, 2703686851)
    sd_settings['seed'] = seed
else:
    seed = sd_settings['seed']

sampler_format_mapping = {
    'k_euler': 'euler',
    'k_euler_ancestral': 'euler_a',
    'k_dpm_2': 'dpm',
    'k_dpm_2_ancestral': 'dpm_a',
    'k_lms': 'lms',
    'k_heun': 'heun'
}

if sd_settings['sampler'] in sampler_format_mapping:
    sampler = sampler_format_mapping[sd_settings['sampler']]                
else:
    sampler = sd_settings['sampler']

if sd_settings['run_type'] == 'paintImg2Img':
    init_image = userprofile+'/artroom/settings/out.png'
else:
    init_image = sd_settings['init_image']

if init_image != '':
    sampler = 'ddim'

if sd_settings['aspect_ratio'] != 'Init Image' or len(init_image) == 0:
    W = int(sd_settings['width'])
    H = int(sd_settings['height'])
else:
    W = 0
    H = 0

if sd_config['speed'] == "Low":
    turbo = False
    superfast = False 
elif sd_config['speed'] == "Medium":
    turbo = True
    superfast = False
else:
    turbo = True 
    superfast = True 

settings = {
    'prompt': sd_settings['text_prompts'],
    'negative_prompt': sd_settings['negative_prompts'],
    'outdir': outdir,
    'skip_grid': not sd_config['save_grid'],
    'ddim_steps': int(sd_settings['steps']),
    'n_samples': int(sd_settings['n_samples']),
    'n_iter': int(sd_settings['n_iter']),
    'H': H,
    'W': W,
    'scale': float(sd_settings['cfg_scale']),
    'seed': int(seed),
    'precision' : 'full' if sd_config['use_full_precision'] else 'autocast',
    'ckpt': sd_config['model_ckpt'],
    'device': 'cpu' if sd_config['use_cpu'] else 'cuda',
    'sampler': sampler,
    'turbo': turbo,
    'superfast': superfast,
    'init_image': init_image,
    'strength': float(sd_settings['strength']) if len(init_image) > 0 else 1.0,
    'mask' : userprofile+'/artroom/settings/out.png' if sd_settings['run_type'] == 'inpainting' else '',
    'invert' : sd_settings['reverse_mask'],
    'neon_vram': sd_config['neon_vram'],
}
try:
    queue_json = json.load(open(f'{userprofile}/artroom/settings/queue.json'))
except:
    queue_json = {
        "Queue": [],
        "Running": False,
        "Keep_Warm": False,
        "Delay": 5
    }

if queue_json['Keep_Warm']:
    queue_json["Queue"].append(settings)
else:
    queue_json["Queue"].insert(0,settings)

with open(f'{userprofile}/artroom/settings/queue.json', 'w') as outfile:
    json.dump(queue_json, outfile, indent=4)

image_folder = os.path.join(outdir,re.sub(r'\W+', '','_'.join(sd_settings['text_prompts'].split())))[:150]
os.makedirs(image_folder,exist_ok=True)
os.makedirs(image_folder+"/settings",exist_ok=True)
sd_json = {'Settings': sd_settings, 'Config': sd_config}
sd_settings_count = len(glob(image_folder+'/settings/*.json'))
with open(f'{image_folder}/settings/sd_settings_{seed}_{sd_settings_count}.json', 'w') as outfile:
    json.dump(sd_json, outfile, indent=4)
with open(f'{userprofile}/artroom/settings/queue.json', 'w') as outfile:
    json.dump(queue_json, outfile, indent=4)     
print('Added to Queue!')