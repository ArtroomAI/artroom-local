from flask import Flask, request, jsonify, make_response
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename

from upscale import Upscaler
from queue_manager import QueueManager
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
    QM.set_artroom_path(artroom_path)
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

QM = QueueManager(SD, artroom_path)
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

@socketio.on('/get_server_status')
def get_server_status():
    socketio.emit("get_server_status", {'server_running': SD.running }, broadcast=True)

@socketio.on('get_queue')
def get_queue():
    socketio.emit("get_queue", {'queue': QM.queue }, broadcast=True)

@socketio.on('start_queue')
def start_queue():
    print('Starting Queue...')
    if not QM.running:
        run_sd()
        socketio.emit("start_queue", {'status': 'Success' }, broadcast=True)
    else:
        print('Queue already running')
        socketio.emit("start_queue", {'status': 'Failure' }, broadcast=True)   

@socketio.on('pause_queue')
def pause_queue():
    print('Pausing queue...')
    if QM.running:
        QM.running = False
        print('Queue paused')
        socketio.emit("pause_queue", {'status': 'Success' }, broadcast=True)
    else:
        print('Failed to pause queue')
        socketio.emit("pause_queue", {'status': 'Failure' }, broadcast=True)

@socketio.on('stop_queue')
def stop_queue():
    print('Stopping queue...')
    QM.running = False
    SD.running = False
    print('Queue stopped')
    socketio.emit("stop_queue", {'status': 'Success'}, broadcast=True)

@socketio.on('remove_from_queue')
def remove_from_queue(data):
    print('Removing from queue...')
    QM.remove_from_queue(data['id'])
    print(f"{data['id']} removed from queue")
    socketio.emit("remove_from_queue", { 'status': 'Success', 'queue': QM.queue }, broadcast=True)

@socketio.on('add_to_queue')
def add_to_queue(data):
    print('Adding to queue...')
    if data['ckpt'] == '':
        print('Failure, model checkpoint cannot be blank')
        socketio.emit('add_to_queue', { 'status': 'Failure', 'status_message': 'Model Checkpoint cannot be blank. Please go to Settings and set a model ckpt.'})
        return

    QM.add_to_queue(data)

    # Cleans up printout so you don't print out the whole b64
    data_copy = dict(data)
    if len(data_copy['init_image']):
        data_copy['init_image'] = data_copy['init_image'][:100]+"..."
    if len(data_copy['mask_image']):
        data_copy['mask_image'] = data_copy['mask_image'][:100]+"..."
    print(f'Added to queue: {data_copy}')
    if not QM.running:
        run_sd()
    socketio.emit('add_to_queue', { 'status': 'Success', 'queue_size': len(QM.queue) })

@socketio.on('clear_queue')
def clear_queue():
    print('Clearing queue...')
    QM.clear_queue()
    print('Queue cleared')
    socketio.emit('clear_queue', { 'status': 'Success' })

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
    if 'delay' in data:
        QM.update_delay = data['delay']
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
    if 'delay' in data:
        QM.update_delay = data['delay']
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

@app.route('/get_settings', methods=['GET'])
def get_settings():
    if not SD.artroom_path:
        print('Failed to get settings, artroom path not found')
        return return_output('Failure', 'Artroom Path not found')
    if not os.path.exists(f'{SD.artroom_path}/artroom/settings/sd_settings.json'):
        reset_settings_to_default()
        return return_output('Failure', 'sd_settings.json not found')
    sd_settings = json.load(
        open(f'{SD.artroom_path}/artroom/settings/sd_settings.json'))
    return return_output('Success', content={'status': QM.queue, 'settings': sd_settings})


def run_sd():
    if not QM.running:
        print('Queue started!')
        QM.read_queue_json()
        QM.thread = threading.Thread(target=QM.run_queue, daemon=True)
        QM.thread.start()
        return return_output('Success', 'Starting Artroom')
    else:
        print('Queue already running')
        return return_output('Failure', 'Already running')

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

