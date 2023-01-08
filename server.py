from flask import Flask, request, jsonify, make_response
from flask_socketio import SocketIO,emit
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
import traceback
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

class ArtroomServer:
    def __init__(self, SD):
        self.ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
        self.SD = SD 
        self.update_paths()

    def update_paths(self):
        self.result_url = self.SD.image_save_path
        self.thumbnail_image_url = os.path.join(self.SD.image_save_path, "thumbnails/")
        self.temp_image_url = os.path.join(self.SD.image_save_path, "temp-images/")
        self.intermediate_url = os.path.join(self.SD.image_save_path, "intermediates/")
        self.mask_image_url = os.path.join(self.SD.image_save_path, "mask-images/")
        self.init_image_url = os.path.join(self.SD.image_save_path, "init-images/")

        for path in [self.result_url, self.thumbnail_image_url, self.temp_image_url, self.intermediate_url, self.mask_image_url, self.init_image_url]:
            os.makedirs(path, exist_ok=True)

    def reset_settings_to_default(self):
        print('Failure, sd_settings not found. Resetting to default')
        if os.path.exists('sd_settings.json'):
            shutil.copy('sd_settings.json', f'{self.SD.artroom_path}/artroom/settings/')
            print('Successfully resetted to default')
        else:
            print('Resetting failed')

    def get_url_from_image_path(self, path):  
        self.update_paths()      
        """Given an absolute file path to an image, returns the URL that the client can use to load the image"""
        try:
            if "init-images" in path:
                return os.path.join(self.init_image_url, os.path.basename(path))
            elif "mask-images" in path:
                return os.path.join(self.mask_image_url, os.path.basename(path))
            elif "intermediates" in path:
                return os.path.join(self.intermediate_url, os.path.basename(path))
            elif "temp-images" in path:
                return os.path.join(self.temp_image_url, os.path.basename(path))
            elif "thumbnails" in path:
                return os.path.join(self.thumbnail_image_url, os.path.basename(path))
            else:
                return os.path.join(self.result_url, os.path.basename(path))
        except Exception as e:
            socketio.emit("error", {"message": (str(e))})
            print("\n")

            traceback.print_exc()
            print("\n")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)
SD = StableDiffusion()
UP = Upscaler()
AS = ArtroomServer(SD) 

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
    # returns (status,status_message)
    upscale_status = UP.upscale(
        data['upscale_images'], data['upscaler'], data['upscale_factor'], data['upscale_dest'])
    return return_output(upscale_status[0], upscale_status[1])


@app.route('/get_images', methods=['GET'])
def get_images():
    path = request.args.get('path')
    id = int(request.args.get('id'))
    if id == SD.latest_images_id:
        return return_output('Hold', 'No new updates on images',
                            content={'latest_images_id': SD.latest_images_id, 'latest_images': []})
    try:
        if path == 'latest':
            imageB64 = [support.image_to_b64(image)
                        for image in SD.get_latest_images()]
        else:
            image = Image.open(path).convert('RGB')
            imageB64 = support.image_to_b64(image)
        return return_output('Success', content={'latest_images_id': SD.latest_images_id, 'latest_images': imageB64})
    except:
        return return_output('Failure', 'Failed to get image', {'latest_images_id': -1, 'imageB64': ''})


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
        AS.reset_settings_to_default()
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
        AS.reset_settings_to_default()
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
    print(f"{data['id']} removed from queue")
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
        data_copy['init_image'] = data_copy['init_image'][:100]+"..."
    if len(data_copy['mask']):
        data_copy['mask'] = data_copy['mask'][:100]+"..."
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

@app.route('/invoke_upload', methods=['POST'])
def invoke_upload():
    try:
        #Comes from a fetch request, lets standardize
        data = json.loads(request.form["data"])
        filename = ""
        # check if the post request has the file part
        if "file" in request.files:
            file = request.files["file"]
            # If the user does not select a file, the browser submits an
            # empty file without a filename.
            if file.filename == "":
                return make_response("No file selected", 400)
            filename = file.filename
        elif "dataURL" in data:
            file = support.dataURL_to_bytes(data["dataURL"])
            if "filename" not in data or data["filename"] == "":
                return make_response("No filename provided", 400)
            filename = data["filename"]
        else:
            return make_response("No file or dataURL", 400)

        kind = data["kind"]

        if kind == "init":
            path = os.path.join(SD.image_save_path, "intermediates/")
        elif kind == "temp":
            path = os.path.join(SD.image_save_path, "temp-images/")
        elif kind == "result":
            path = SD.image_save_path
        elif kind == "mask":
            path = os.path.join(SD.image_save_path, "mask-images/")
        else:
            return make_response(f"Invalid upload kind: {kind}", 400)

        
        if not support.allowed_file(filename):
            return make_response(
                f'Invalid file type, must be one of: {", ".join(AS.ALLOWED_EXTENSIONS)}',
                400,
            )

        secured_filename = secure_filename(filename)

        uuid = uuid4().hex
        truncated_uuid = uuid[:8]

        split = os.path.splitext(secured_filename)
        name = f"{split[0]}.{truncated_uuid}{split[1]}"

        file_path = os.path.join(path, name)

        if "dataURL" in data:
            with open(file_path, "wb") as f:
                f.write(file)
        else:
            file.save(file_path)

        mtime = os.path.getmtime(file_path)

        pil_image = Image.open(file_path)

        if "cropVisible" in data and data["cropVisible"] == True:
            visible_image_bbox = pil_image.getbbox()
            pil_image = pil_image.crop(visible_image_bbox)
            pil_image.save(file_path)

        (width, height) = pil_image.size

        thumbnail_path = support.save_thumbnail(
            pil_image, os.path.basename(file_path), os.path.join(SD.image_save_path, "thumbnails/")
        )

        response = {
            "url": AS.get_url_from_image_path(file_path),
            "thumbnail": AS.get_url_from_image_path(thumbnail_path),
            "mtime": mtime,
            "width": width,
            "height": height,
        }

        return make_response(response, 200)

    except Exception as e:
        socketio.emit("error", {"message": (str(e))})
        print("\n")

        traceback.print_exc()
        print("\n")
        return make_response("Error uploading file", 500)

@app.route('/invoke_inpainting', methods=['POST'])
def invoke_inpainting():
    """
    generation_parameters["init_img"] is a base64 image
    generation_parameters["init_mask"] is a base64 image

    So we need to convert each into a PIL Image.
    """
    data = json.loads(request.data)

    original_bounding_box = data["bounding_box"].copy()

    initial_image = support.dataURL_to_image(
        data["init_img"]
    ).convert("RGBA")

    """
    The outpaint image and mask are pre-cropped by the UI, so the bounding box we pass
    to the generator should be:
        {
            "x": 0,
            "y": 0,
            "width": original_bounding_box["width"],
            "height": original_bounding_box["height"]
        }
    """

    data["bounding_box"]["x"] = 0
    data["bounding_box"]["y"] = 0

    # Convert mask dataURL to an image and convert to greyscale
    mask_image = support.dataURL_to_image(
        data["init_mask"]
    ).convert("L")

    actual_generation_mode = support.get_canvas_generation_mode(
        initial_image, mask_image
    )

    """
    Apply the mask to the init image, creating a "mask" image with
    transparency where inpainting should occur. This is the kind of
    mask that prompt2image() needs.
    """
    alpha_mask = initial_image.copy()
    alpha_mask.putalpha(mask_image)

    data["init_img"] = initial_image
    data["init_mask"] = alpha_mask

    # Remove the unneeded parameters for whichever mode we are doing
    if actual_generation_mode == "inpainting":
        data.pop("seam_size", None)
        data.pop("seam_blur", None)
        data.pop("seam_strength", None)
        data.pop("seam_steps", None)
        data.pop("tile_size", None)
        data.pop("force_outpaint", None)
    elif actual_generation_mode == "img2img":
        data["height"] = original_bounding_box["height"]
        data["width"] = original_bounding_box["width"]
        data.pop("init_mask", None)
        data.pop("seam_size", None)
        data.pop("seam_blur", None)
        data.pop("seam_strength", None)
        data.pop("seam_steps", None)
        data.pop("tile_size", None)
        data.pop("force_outpaint", None)
        data.pop("infill_method", None)
    
    print(data)


@app.route('/shutdown', methods=['GET'])
def shutdown():
    stop_queue()
    os._exit(0)

@socketio.on("connect")
def connected():
    """event listener when client connects to the server"""
    print(request.sid)
    print("client has connected")
    emit("connect",{"data":f"id: {request.sid} is connected"})

@socketio.on('data')
def handle_message(data):
    """event listener when client types a message"""
    print("data from the front end: ",str(data))
    emit("data",{'data':data,'id':request.sid},broadcast=True)

@socketio.on("disconnect")
def disconnected():
    """event listener when client disconnects to the server"""
    print("user disconnected")
    emit("disconnect",f"user {request.sid} disconnected",broadcast=True)
    
if __name__ == '__main__':
    socketio.run(app, host='127.0.0.1', port=5300)

