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
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading', logger=False, engineio_logger=False)
SD = StableDiffusion(socketio)
UP = Upscaler()

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

@app.route('/upscale', methods=['POST'])
def upscale():
    print('Running upscale...')
    data = json.loads(request.data)
    if UP.running:
        print('Failure to upscale, upscale is already running')
        return return_output('Failure', 'Upscale is already running')
    if len(data['upscale_images']) == 0:
        print('Failure to upscale, please select an image')
        return return_output('Failure', 'Please select an image')

    if data['upscale_dest'] == '':
        data['upscale_dest'] = SD.image_save_path+'/upscale_outputs'

    upscale_output = UP.upscale(
        data['upscale_images'], data['upscaler'], data['upscale_factor'], data['upscale_dest'])
    print(upscale_output)
    return return_output(upscale_output['status'], upscale_output['status_message'])

@app.route('/upscale_canvas', methods=['POST'])
def upscale_canvas():
    print('Running upscale...')
    data = json.loads(request.data)
    if UP.running:
        print('Failure to upscale, upscale is already running')
        return return_output('Failure', 'Upscale is already running')
    
    os.makedirs(f'{SD.artroom_path}/artroom/intermediates/', exist_ok=True)
    canvas_temp_path = os.path.join(f'{SD.artroom_path}','artroom/intermediates/canvas.png')
    upscale_temp_path = os.path.join(f'{SD.artroom_path}','artroom/intermediates/')
    canvas_image = support.dataURL_to_image(data['upscale_image']).convert('RGBA')
    #Crops to only visible
    visible_image_bbox = canvas_image.getbbox()
    canvas_image = canvas_image.crop(visible_image_bbox)
    canvas_image.save(canvas_temp_path)

    upscale_output = UP.upscale(
        images = [canvas_temp_path],
        upscaler = data['upscaler'], 
        upscale_factor = data['upscale_factor'],
        upscale_dest = upscale_temp_path
        )
    SD.add_to_latest(upscale_output["content"]["output_images"][0], upscale_output["content"]["save_paths"][0])
    SD.latest_images_id = random.randint(1, 922337203685)
    print("Upscale Finished")
    return return_output(upscale_output['status'], upscale_output['status_message'])

@app.route('/get_images', methods=['GET'])
def get_images():
    path = request.args.get('path')
    id = int(request.args.get('id'))
    if id == SD.latest_images_id:
        return return_output('Hold', 'No new updates on images',
                            content={'latest_images_id': SD.latest_images_id, 'latest_images': []})
    try:
        if path == 'latest':
            image_data = SD.get_latest_images()            
        else:
            image = Image.open(path).convert('RGB')
            image_data = [{'b64': support.image_to_b64(image), 'path': path}]
        return return_output('Success', content={'latest_images_id': SD.latest_images_id, 'latest_images': image_data})
    except Exception as e:
        print(e)
        return return_output('Failure', 'Failed to get image', {'latest_images_id': -1, 'latest_images': []})


@app.route('/get_server_status', methods=['GET'])
def get_server_status():
    return return_output('Success', content={'server_running': SD.running})


@app.route('/get_progress', methods=['GET'])
def get_progress():
    percentage = -1
    if SD.stage == 'Generating':
        percentage = 0
    current_num, total_num, current_step, total_step = SD.get_steps()
    if total_step*total_num > 0:
        percentage = (current_num*total_step+current_step) / \
            (total_num*total_step)
    return return_output('Success', content={'current_name': current_num, 'total_num': total_num, 'current_step': current_step, 'total_step': total_step,
                                            'percentage': int(percentage*100), 'status': SD.stage})


@app.route('/update_settings', methods=['POST'])
def update_settings():
    print('Updating Settings...')
    data = json.loads(request.data)
    if not SD.artroom_path:
        print('Failure, artroom path not found')
        return return_output('Failure', 'Artroom Path not found')
    if not os.path.exists(f'{SD.artroom_path}/artroom/settings/sd_settings.json'):
        reset_settings_to_default()
        return return_output('Failure', 'sd_settings.json not found')
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


@app.route('/get_queue', methods=['GET'])
def get_queue():
    return return_output('Success', content={'queue': QM.queue})


@app.route('/start_queue', methods=['GET'])
def start_queue():
    print('Starting Queue...')
    if not QM.running:
        run_sd()
        return return_output('Success')
    else:
        print('Queue already running')
        return return_output('Failure')

@app.route('/pause_queue', methods=['GET'])
def pause_queue():
    print('Pausing queue...')
    if QM.running:
        QM.running = False
        print('Queue paused')
        return return_output('Success')
    else:
        print('Failed to pause queue')
        return return_output('Failure')

@app.route('/stop_queue', methods=['GET'])
def stop_queue():
    print('Stopping queue...')
    QM.running = False
    SD.running = False
    print('Queue stopped')
    return return_output('Success')

@app.route('/clear_queue', methods=['POST'])
def clear_queue():
    print('Clearing queue...')
    QM.clear_queue()
    print('Queue cleared')
    return return_output('Success')

@app.route('/remove_from_queue', methods=['POST'])
def remove_from_queue():
    print('Removing from queue...')
    data = json.loads(request.data)
    QM.remove_from_queue(data['id'])
    print(f'{data["id"]} removed from queue')
    return return_output('Success', content={'queue': QM.queue})


@app.route('/add_to_queue', methods=['POST'])
def add_to_queue():
    print('Adding to queue...')
    data = json.loads(request.data)
    if data['ckpt'] == '':
        print('Failure, model checkpoint cannot be blank')
        return return_output('Failure', 'Model Checkpoint cannot be blank. Please go to Settings and set a model ckpt.')

    QM.add_to_queue(data)

    # Cleans up printout so you don't print out the whole b64
    data_copy = dict(data)
    if len(data_copy['init_image']):
        data_copy['init_image'] = data_copy['init_image'][:100]+'...'
    if len(data_copy['mask_image']):
        data_copy['mask_image'] = data_copy['mask_image'][:100]+'...'
    print(f'Added to queue: {data_copy}')
    if not QM.running:
        run_sd()
    return return_output('Success', content={'queue': QM.queue})


@app.route('/start', methods=['POST'])
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
    socketio.run(app, host='127.0.0.1', port=5300)

