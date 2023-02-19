from flask import Flask, request, jsonify, make_response
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename

from upscale import Upscaler
from stable_diffusion import StableDiffusion
from artroom_helpers import support
import logging
import os
from PIL import Image
import json
import shutil
import threading
import ctypes
import random
from uuid import uuid4
from glob import glob
import re

kernel32 = ctypes.windll.kernel32
kernel32.SetConsoleMode(kernel32.GetStdHandle(-10), 128)

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
print('Running in Debug Mode. Please keep CMD window open')

def return_output(status, status_message='', content=''):
    if not status_message and status == 'Failure':
        status_message = 'Unknown Error'
    return jsonify({'status': status, 'status_message': status_message, 'content': content})

def reset_settings_to_default(self):
    print('Failure, sd_settings not found. Resetting to default')
    if os.path.exists('sd_settings.json'):
        shutil.copy('sd_settings.json', f'{self.SD.artroom_path}/artroom/settings/')
        print('Successfully resetted to default')
    else:
        print('Resetting failed')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
max_ws_http_buffer_size = 50_000_000 # 50MB
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading', logger=False, engineio_logger=False, max_http_buffer_size=max_ws_http_buffer_size)

UP = Upscaler()
SD = StableDiffusion(socketio = socketio, Upscaler = UP)

def set_artroom_paths(artroom_path):
    UP.set_artroom_path(artroom_path)
    SD.set_artroom_path(artroom_path)

user_profile = os.environ['USERPROFILE']
artroom_install_log = f'{user_profile}/AppData/Local/artroom_install.log'
if os.path.exists(artroom_install_log):
    # artroom_path = f'{user_profile}/AppData/Local/artroom_install.log'
    f = open(artroom_install_log, 'r')
    artroom_path_raw = f.readline()
    f.close()
    artroom_path = artroom_path_raw[:-1]
else:
    artroom_path = os.environ['USERPROFILE']

threading.Thread(target=set_artroom_paths, args=[
    artroom_path], daemon=True).start()

@socketio.on('upscale')
def upscale(data):
    print('Running upscale...')
    if UP.running:
        print('Failure to upscale, upscale is already running')
        socketio.emit('upscale', { 'status': 'Failure', 'status_message': 'Upscale is already running' })
        return
    if len(data['upscale_images']) == 0:
        print('Failure to upscale, please select an image')
        socketio.emit('upscale', { 'status': 'Failure', 'status_message': 'Please select an image' })
        return

    if data['upscale_dest'] == '':
        data['upscale_dest'] = SD.image_save_path+'/upscale_outputs'

    UP.upscale(data['upscale_images'], data['upscaler'], data['upscale_factor'], data['upscale_dest'])
    socketio.emit('upscale', { 'status': 'Success', 'status_message': 'Your upscale has completed' })
    return 

def save_to_settings_folder(data):
    print("Saving settings...")
    if SD.long_save_path:
        image_folder = os.path.join(data['image_save_path'],data['batch_name'], re.sub(
            r'\W+', '', '_'.join(data['text_prompts'].split())))[:150]
        os.makedirs(image_folder, exist_ok=True)
        os.makedirs(image_folder+'/settings', exist_ok=True)
        sd_settings_count = len(glob(image_folder+'/settings/*.json'))
        with open(f'{image_folder}/settings/sd_settings_{data["seed"]}_{sd_settings_count}.json', 'w') as outfile:
            json.dump(data, outfile, indent=4)
    else:
        image_folder = os.path.join(
            data['image_save_path'],data['batch_name'])
        os.makedirs(image_folder, exist_ok=True)
        os.makedirs(image_folder+'/settings', exist_ok=True)
        sd_settings_count = len(glob(image_folder+'/settings/*.json'))
        prompt_name = re.sub(
            r'\W+', '', "_".join(data["text_prompts"].split()))[:100]
        with open(f'{image_folder}/settings/sd_settings_{prompt_name}_{data["seed"]}_{sd_settings_count}.json', 'w') as outfile:
            json.dump(data, outfile, indent=4)
    print("Settings saved")

def save_settings_cache(data):
    with open(f'{artroom_path}/artroom/settings/sd_settings.json', 'r') as infile:
        existing_data = json.load(infile)
    existing_data.update(data)
    with open(f'{artroom_path}/artroom/settings/sd_settings.json', 'w') as outfile:
        json.dump(existing_data, outfile, indent=4)

@socketio.on('generate')
def generate(data):
    if not SD.running:
        data['id'] = random.randint(1, 922337203685)
        sampler_format_mapping = {
            'k_euler': 'euler',
            'k_euler_ancestral': 'euler_a',
            'k_dpm_2': 'dpm',
            'k_dpm_2_ancestral': 'dpm_a',
            'k_lms': 'lms',
            'k_heun': 'heun'
        }

        if data['sampler'] in sampler_format_mapping:
            data['sampler'] = sampler_format_mapping[data['sampler']]

        if data['use_random_seed']:
            data['seed'] = random.randint(1, 4294967295)
        else:
            data['seed'] = int(data['seed'])
        data['steps'] = int(data['steps'])
        data['n_iter'] = int(data['n_iter'])
        data['cfg_scale'] = float(data['cfg_scale'])

        if len(data['init_image']) > 0:
            if 'strength' in data:
                data['strength'] = float(data['strength'])
            else:
                data['strength'] = 0.75
        else:
            data['strength'] = 0.75

        if '%UserProfile%' in data['image_save_path']:
            data['image_save_path'] = data['image_save_path'].replace(
                '%UserProfile%', os.environ['USERPROFILE'])
        data['image_save_path'] = data['image_save_path'].replace(os.sep, '/')

        if '%UserProfile%' in data['ckpt']:
            data['ckpt'] = data['ckpt'].replace(
                '%UserProfile%', os.environ['USERPROFILE'])
        if '%InstallPath%' in data['ckpt']:
            data['ckpt'] = data['ckpt'].replace(
                '%InstallPath%', artroom_path)
        data['ckpt'] = data['ckpt'].replace(os.sep, '/')

        if '%UserProfile%' in data['ckpt_dir']:
            data['ckpt_dir'] = data['ckpt_dir'].replace(
                '%UserProfile%', os.environ['USERPROFILE'])
        if '%InstallPath%' in data['ckpt_dir']:
            data['ckpt_dir'] = data['ckpt_dir'].replace(
                '%InstallPath%', artroom_path)
        data['ckpt_dir'] = data['ckpt_dir'].replace(os.sep, '/')

        if data['aspect_ratio'] == 'Init Image':
            # Load image sets it to be equal to init_image dimensions
            data['width'] = 0
            data['height'] = 0

        save_settings_cache(data)

        mask_b64 = data['mask_image']
        data['mask_image'] = ''
        init_image_str = data['init_image']
        print("Saving settings to folder...")
        save_to_settings_folder(data)
        ckpt_path = os.path.join(data['ckpt_dir'],data['ckpt']).replace(os.sep, '/')
        vae_path = os.path.join(data['ckpt_dir'],data['vae']).replace(os.sep, '/')
        # try:
        print("Starting gen...")
        print(data)
        SD.generate(
            text_prompts=data['text_prompts'],
            negative_prompts=data['negative_prompts'],
            batch_name=data['batch_name'],
            init_image_str=init_image_str,
            strength=data['strength'],
            mask_b64=mask_b64,
            invert=data['invert'],
            n_iter=int(data['n_iter']),
            steps=int(data['steps']),
            H=int(data['height']),
            W=int(data['width']),
            seed=int(data['seed']),
            sampler=data['sampler'],
            cfg_scale=float(data['cfg_scale']),
            palette_fix = data['palette_fix'],
            ckpt=ckpt_path,
            vae=vae_path,
            image_save_path=data['image_save_path'],
            speed=data['speed'],
            skip_grid=not data['save_grid'],
        )
        socketio.emit('job_done')

@socketio.on('/get_server_status')
def get_server_status():
    socketio.emit("get_server_status", {'server_running': SD.running }, broadcast=True)

@socketio.on('stop_queue')
def stop_queue():
    print('Stopping queue...')
    SD.interrupt()
    print('Queue stopped')
    socketio.emit("stop_queue", {'status': 'Success'}, broadcast=True)

@socketio.on('update_settings')
def update_settings(data):
    print('Updating Settings...')
    if not SD.artroom_path:
        print('Failure, artroom path not found')
        socketio.emit('update_settings', { 'status': 'Failure', 'status_message': 'Artroom Path not found' })
        return
    if not os.path.exists(f'{SD.artroom_path}/artroom/settings/sd_settings.json'):
        reset_settings_to_default()
        socketio.emit('update_settings', { 'status': 'Failure', 'status_message': 'sd_settings.json not found' })
        return
    if 'long_save_path' in data:
        SD.long_save_path = data['long_save_path']
    if 'highres_fix' in data:
        SD.highres_fix = data['highres_fix']
    sd_settings = json.load(
        open(f'{SD.artroom_path}/artroom/settings/sd_settings.json'))
    for key in data:
        value = data[key]
        if type(value) == str and '%UserProfile%' in value:
            value = value.replace(
                '%UserProfile%', os.environ['USERPROFILE']).replace(os.sep, '/')
        if type(value) == str and '%InstallPath%' in value:
            value = value.replace(
                '%InstallPath%', SD.artroom_path).replace(os.sep, '/')
        sd_settings[key] = value
    with open(f'{SD.artroom_path}/artroom/settings/sd_settings.json', 'w') as outfile:
        json.dump(sd_settings, outfile, indent=4)
    # SD.load_from_settings_json()
    print('Settings updated')
    return return_output('Success')

@socketio.on('update_settings_with_restart')
def update_settings(data):
    print('Updating Settings...')
    if not SD.artroom_path:
        print('Failure, artroom path not found')
        socketio.emit('update_settings_with_restart', { 'status': 'Failure', 'status_message': 'Artroom Path not found' })
        return
    if not os.path.exists(f'{SD.artroom_path}/artroom/settings/sd_settings.json'):
        reset_settings_to_default()
        socketio.emit('update_settings_with_restart', { 'status': 'Failure', 'status_message': 'sd_settings.json not found' })
        return
    if 'long_save_path' in data:
        SD.long_save_path = data['long_save_path']
    if 'highres_fix' in data:
        SD.highres_fix = data['highres_fix']
    sd_settings = json.load(
        open(f'{SD.artroom_path}/artroom/settings/sd_settings.json'))
    for key in data:
        value = data[key]
        if type(value) == str and '%UserProfile%' in value:
            value = value.replace(
                '%UserProfile%', os.environ['USERPROFILE']).replace(os.sep, '/')
        if type(value) == str and '%InstallPath%' in value:
            value = value.replace(
                '%InstallPath%', SD.artroom_path).replace(os.sep, '/')
        sd_settings[key] = value
    with open(f'{SD.artroom_path}/artroom/settings/sd_settings.json', 'w') as outfile:
        json.dump(sd_settings, outfile, indent=4)
    # SD.load_from_settings_json()
    print('Settings updated')
    socketio.emit('update_settings_with_restart', { 'status': 'Success' })

@app.route('/shutdown', methods=['GET'])
def shutdown():
    stop_queue()
    os._exit(0)

@socketio.on('connect')
def connected():
    '''event listener when client connects to the server'''
    print(request.sid)
    print('client has connected')
    socketio.emit('connect',{'data':f'id: {request.sid} is connected'})

@socketio.on('message')
def handle_message(data):
    '''event listener when client types a message'''
    print('data from the front end: ',str(data))
    socketio.emit('message',{'data':data,'id':request.sid},broadcast=True)

@socketio.on('disconnect')
def disconnected():
    '''event listener when client disconnects to the server'''
    print('user disconnected')
    socketio.emit('disconnect',f'user {request.sid} disconnected',broadcast=True)
    
if __name__ == '__main__':
    socketio.run(app, host='127.0.0.1', port=5300, allow_unsafe_werkzeug=True)

