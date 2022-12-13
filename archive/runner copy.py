import subprocess
import json
import os
import random
import time
import subprocess
import sys
import re
from glob import glob
import ctypes

kernel32 = ctypes.windll.kernel32
kernel32.SetConsoleMode(kernel32.GetStdHandle(-10), 128)

userprofile = os.environ['USERPROFILE']
 
sd_json = json.load(open(f'{userprofile}/artroom/settings/sd_settings.json'))
sd_settings = sd_json['Settings']
sd_config = sd_json['Config']

os.chdir('stable-diffusion')

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
    # sd_settings['use_random_seed'] = False
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

prompt_file_path = f'{userprofile}/artroom/settings/'
with open(prompt_file_path+'prompt.txt', 'w') as f:
    f.write(sd_settings['text_prompts'])
with open(prompt_file_path+'negative_prompt.txt', 'w') as f:
    f.write(sd_settings['negative_prompts'])

script_command = [ f'{userprofile}\\artroom\\miniconda3\\condabin\\activate.bat','&&',
                    f'conda', 'run','--no-capture-output', '-p', f'{userprofile}/artroom/miniconda3/envs/artroom-ldm',
                    'python', 'optimizedSD/stable_diffusion.py',
                    # 'python', 'scripts/txt2img.py',
                    '--scale', str(sd_settings['cfg_scale']),
                    '--outdir', outdir,
                    '--n_samples', str(sd_settings['n_samples']),
                    '--ddim_steps', str(sd_settings['steps']),
                    '--seed', str(seed),
                    '--ckpt', sd_config['model_ckpt'],
                    '--n_iter', str(sd_settings['n_iter']),
                    '--from_file', prompt_file_path,
                    '--sampler', sampler,
                    '--skip_grid'
                    ]    

# if not sd_config['save_grid']:
#     script_command += ['--skip_grid']

if sd_config['use_turbo']:
    script_command += ['--turbo']
if sd_config['use_superfast']:
    script_command += ['--superfast']

if sd_config['use_full_precision']:
    script_command += ['--precision','full']

if sd_settings['aspect_ratio'] != 'Init Image' or len(init_image) == 0:
    script_command += ['--W', str(sd_settings['width'])]
    script_command += ['--H', str(sd_settings['height'])]

if init_image != '':
    script_command += ['--init_image',init_image]
    script_command += ['--strength',str(sd_settings['strength'])]

if sd_settings['run_type'] == 'inpainting':
    script_command += ['--mask', userprofile+'/artroom/settings/out.png']

if sd_settings['reverse_mask']:
    script_command += ['--invert']

print(script_command)
try:
    print('Running....')
    image_folder = os.path.join(outdir,re.sub(r'\W+', '','_'.join(sd_settings['text_prompts'].split())))[:150]
    # print(image_folder)
    os.makedirs(image_folder,exist_ok=True)
    os.makedirs(image_folder+"/settings",exist_ok=True)
    sd_json = {'Settings': sd_settings, 'Config': sd_config}
    sd_settings_count = len(glob(image_folder+'/*.json'))
    with open(f'{image_folder}/settings/sd_settings_{seed}_{sd_settings_count}.json', 'w') as outfile:
        json.dump(sd_json, outfile, indent=4)
    time.sleep(1)
    process = subprocess.run(script_command)
    print('Finished!')
    time.sleep(500)

except Exception as e:
    print(f'ERROR: {e}')
    time.sleep(600)
