try:

    from upscale import Upscaler
    from queue_manager import QueueManager
    from stable_diffusion import StableDiffusion
    import logging
    import os
    from flask import Flask, request, jsonify
    from PIL import Image
    import json
    import base64
    import shutil
    from io import BytesIO
    import threading
    import re
    import ctypes
    import time

    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-10), 128)


    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    print('Running in Debug Mode. Please keep CMD window open')


    def return_output(status, status_message='', content=''):
        if not status_message and status == 'Failure':
            status_message = 'Unknown Error'
        return jsonify({'status': status, 'status_message': status_message, 'content': content})


    def image_to_b64(image):
        image_file = BytesIO()
        image.save(image_file, format='JPEG')
        im_bytes = image_file.getvalue()  # im_bytes: image in binary format.
        imgb64 = base64.b64encode(im_bytes)
        return 'data:image/jpeg;base64,' + str(imgb64)[2:-1]


    def b64_to_image(b64):
        image_data = re.sub('^data:image/.+;base64,', '', b64)
        return Image.open(BytesIO(base64.b64decode(image_data))).convert('RGB')


    def reset_settings_to_default():
        print('Failure, sd_settings not found. Resetting to default')
        if os.path.exists('sd_settings.json'):
            shutil.copy('sd_settings.json', f'{SD.artroom_path}/artroom/settings/')
            print('Successfully resetted to default')
        else:
            print('Resetting failed')


    app = Flask(__name__)
    SD = StableDiffusion()
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
                imageB64 = [image_to_b64(image)
                            for image in SD.get_latest_images()]
            else:
                image = Image.open(path).convert('RGB')
                imageB64 = image_to_b64(image)
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


    @app.route('/shutdown', methods=['GET'])
    def shutdown():
        stop_queue()
        os._exit(0)
        
    if __name__ == '__main__':
        app.run(host='127.0.0.1', port=5300)

except Exception as e:
    import time
    print(e)
    time.sleep(200)