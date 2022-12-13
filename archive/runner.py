import subprocess
import json
import os
import time
from glob import glob
import ctypes
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

kernel32 = ctypes.windll.kernel32
kernel32.SetConsoleMode(kernel32.GetStdHandle(-10), 128)

userprofile = os.environ['USERPROFILE']
os.chdir('stable-diffusion')
try:
    queue_json = json.load(open(f'{userprofile}/artroom/settings/queue.json'))
    if queue_json['Queue'][0]['neon_vram']:
        run_sd = 'optimizedSD_neon/stable_diffusion.py'
    else:
        run_sd = 'optimizedSD/stable_diffusion.py'
    script_command = [f'{userprofile}\\artroom\\miniconda3\\condabin\\activate.bat','&&',
                        f'conda', 'run','--no-capture-output', '-p', f'{userprofile}/artroom/miniconda3/envs/artroom-ldm',
                        'python', run_sd]
    queue_json["Running"] = True
    with open(f'{userprofile}/artroom/settings/queue.json', 'w') as outfile:
        json.dump(queue_json, outfile, indent=4)
    print('Running....')
    process = subprocess.run(script_command)       
    print('Finished!')
    time.sleep(5)
except Exception as e:
    queue_json["Running"] = False
    with open(f'{userprofile}/artroom/settings/queue.json', 'w') as outfile:
        json.dump(queue_json, outfile, indent=4)
    print(f'ERROR: {e}')
    time.sleep(60)
