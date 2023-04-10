try:
    import numpy as np
    import json
    import ctypes
    import logging
    import os
    import re
    import math

    from upscale import Upscaler
    from stable_diffusion import StableDiffusion
    from artroom_helpers import support
    from artroom_helpers.generation.preprocess import mask_from_face
    from model_merger import ModelMerger

    from flask import Flask, request, jsonify, make_response
    from flask_socketio import SocketIO
    from werkzeug.utils import secure_filename
    from PIL import Image, ImageDraw, ImageOps
    from scipy.spatial import ConvexHull
    from uuid import uuid4
    from glob import glob

    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-10), 128)

    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    print('Running in Debug Mode. Please keep CMD window open')


    def return_output(status, status_message='', content=''):
        if not status_message and status == 'Failure':
            status_message = 'Unknown Error'
        return jsonify({'status': status, 'status_message': status_message, 'content': content})


    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'secret!'
    max_ws_http_buffer_size = 50_000_000  # 50MB
    socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading', logger=False, engineio_logger=False,
                        max_http_buffer_size=max_ws_http_buffer_size)

    UP = Upscaler()
    SD = StableDiffusion(socketio=socketio, Upscaler=UP)

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

    os.makedirs(os.path.join(artroom_path, "model_weights/Loras"), exist_ok=True)
    os.makedirs(os.path.join(artroom_path, "model_weights/Vaes"), exist_ok=True)
    os.makedirs(os.path.join(artroom_path, "model_weights/ControlNet"), exist_ok=True)


    @socketio.on('upscale')
    def upscale(data):
        print('Running upscale...')
        if UP.running:
            print('Failure to upscale, upscale is already running')
            socketio.emit('upscale', {'status': 'Failure', 'status_message': 'Upscale is already running'})
            return
        if len(data['upscale_images']) == 0:
            print('Failure to upscale, please select an image')
            socketio.emit('upscale', {'status': 'Failure', 'status_message': 'Please select an image'})
            return

        if data['upscale_dest'] == '':
            data['upscale_dest'] = data['image_save_path'] + '/upscale_outputs'

        UP.upscale(data['models_dir'], data['upscale_images'], data['upscaler'], data['upscale_factor'],
                   data['upscale_dest'])
        socketio.emit('upscale', {'status': 'Success', 'status_message': 'Your upscale has completed'})
        return


    @socketio.on('merge_models')
    def merge_models(data):
        ModelMerger(data).run()


    def save_to_settings_folder(data):
        print("Saving settings...")
        try:
            if data['long_save_path']:
                image_folder = os.path.join(data['image_save_path'], re.sub(
                    r'\W+', '', '_'.join(data['text_prompts'].split())))[:150]
                os.makedirs(image_folder, exist_ok=True)
                os.makedirs(os.path.join(image_folder,'/settings'), exist_ok=True)
                sd_settings_count = len(glob(image_folder + '/settings/*.json'))
                with open(f'{image_folder}/settings/sd_settings_{data["seed"]}_{sd_settings_count}.json', 'w') as outfile:
                    json.dump(data, outfile, indent=4)
            else:
                image_folder = os.path.join(data['image_save_path'], 'settings').replace(os.sep, '/')
                os.makedirs(image_folder, exist_ok=True)
                sd_settings_count = len(glob(f'{image_folder}/*.json'))
                prompt_name = re.sub(
                    r'\W+', '', "_".join(data["text_prompts"].split()))[:100]
                with open(f'{image_folder}/sd_settings_{prompt_name}_{data["seed"]}_{sd_settings_count}.json',
                        'w') as outfile:
                    json.dump(data, outfile, indent=4)
            print("Settings saved")
        except Exception as e:
            print(f"Settings failed to save! {e}")

    @socketio.on('preview_controlnet')
    def preview_controlnet(data):
        from artroom_helpers.process_controlnet_images import apply_pose, apply_depth, apply_canny, apply_normal, \
            apply_scribble, HWC3, apply_hed, init_cnet_stuff, deinit_cnet_stuff
        try:
            print(f'Previewing controlnet {data["controlnet"]}')
            image = support.b64_to_image(data['initImage'])
            w, h = image.size
            # resize to integer multiple of 32
            w, h = map(lambda x: x - x % 64, (w, h))
            image = image.resize((w, h), resample=Image.LANCZOS)
            image = HWC3(np.array(image))
            init_cnet_stuff(data["controlnet"])
            match data["controlnet"]:
                case "canny":
                    image = apply_canny(image)
                case "pose":
                    image = apply_pose(image)
                case "depth":
                    image = apply_depth(image)
                case "normal":
                    image = apply_normal(image)
                case "scribble":
                    image = apply_scribble(image)
                case "hed":
                    image = apply_hed(image)
            output = support.image_to_b64(Image.fromarray(image))
            socketio.emit('get_controlnet_preview', {'controlnetPreview': output})
            print(f"Preview finished")
        except Exception as e:
            print(f"ControlNet preview failed {e}")
            return
        return


    @socketio.on('preview_remove_background')  # TODO Implement this in own tab for bulk jobs
    def preview_remove_background(data):
        print('Removing background...')
        try:
            if data['remove_background'] == 'face':
                import face_alignment
                image = support.b64_to_image(data["initImage"]).convert("RGB")
                mask = mask_from_face(image, image.width, image.height)

                output = ImageOps.fit(image, mask.size, centering=(0.5, 0.5))
                output.putalpha(mask)
            else:
                from artroom_helpers.rembg import rembg
                image = support.b64_to_image(data["initImage"]).convert("RGBA")
                output = rembg.remove(image, session_model=data['remove_background'])
            socketio.emit('get_remove_background_preview', {'removeBackgroundPreview': support.image_to_b64(output)})
            print(f"Background Removed using {data['remove_background']}")
        except Exception as e:
            print(f"Failed to remove background {e}")
            return
        return


    @socketio.on('generate')
    def generate(data):
        if not SD.running:
            try:
                SD.running = True
                mask_b64 = data['mask_image']
                data['mask_image'] = data['mask_image'][:100] + "..."
                init_image_str = data['init_image']
                data['init_image'] = data['init_image'][:100] + "..."

                print("Saving settings to folder...")
                save_to_settings_folder(data)
                ckpt_path = os.path.join(data['models_dir'], data['ckpt']).replace(os.sep, '/')
                vae_path = os.path.join(data['models_dir'], 'Vaes', data['vae']).replace(os.sep, '/')
                lora_paths = []
                if len(data['lora']) > 0:
                    for lora in data['lora']:
                        lora_paths.append({
                            'path': os.path.join(data['models_dir'], 'Loras', lora['name']).replace(os.sep, '/'),
                            'weight': lora['weight']
                        })
            except Exception as e:
                print(f"Failed to add to queue {e}")
                SD.running = False
                socketio.emit('job_done')
                return
            #try:
            print("Starting gen...")
            SD.generate(
                text_prompts=data['text_prompts'],
                negative_prompts=data['negative_prompts'],
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
                clip_skip=max(int(data['clip_skip']),1),
                palette_fix=data['palette_fix'],
                ckpt=ckpt_path,
                vae=vae_path,
                loras=lora_paths,
                image_save_path=data['image_save_path'],
                speed=data['speed'],
                skip_grid=not data['save_grid'],
                long_save_path=data['long_save_path'],
                highres_fix=data['highres_fix'],
                show_intermediates=data['show_intermediates'],
                controlnet=data['controlnet'],
                use_preprocessed_controlnet=data['use_preprocessed_controlnet'],
                remove_background=data['remove_background'],
                use_removed_background=data['use_removed_background'],
                models_dir=data['models_dir'],
            )
            socketio.emit('job_done')

            # except Exception as e:
            #     print(f"Generation failed! {e}")
            #     SD.clean_up()
            #     socketio.emit('job_done')


    @socketio.on('/get_server_status')
    def get_server_status():
        socketio.emit("get_server_status", {'server_running': SD.running}, broadcast=True)


    @socketio.on('stop_queue')
    def stop_queue():
        SD.interrupt()
        socketio.emit("stop_queue", {'status': 'Success'}, broadcast=True)

    @app.route('/generate', methods=['POST'])
    def generate():
        data = json.loads(request.data)
        if not SD.running:
            try:
                SD.running = True
                mask_b64 = data['mask_image']
                data['mask_image'] = data['mask_image'][:100] + "..."
                init_image_str = data['init_image']
                data['init_image'] = data['init_image'][:100] + "..."

                print("Saving settings to folder...")
                save_to_settings_folder(data)
                ckpt_path = os.path.join(data['models_dir'], data['ckpt']).replace(os.sep, '/')
                vae_path = os.path.join(data['models_dir'], 'Vaes', data['vae']).replace(os.sep, '/')
                lora_paths = []
                if len(data['lora']) > 0:
                    for lora in data['lora']:
                        lora_paths.append({
                            'path': os.path.join(data['models_dir'], 'Loras', lora['name']).replace(os.sep, '/'),
                            'weight': lora['weight']
                        })
            except Exception as e:
                print(f"Failed to add to queue {e}")
                SD.running = False
                socketio.emit('job_done')
                return
            try:
                print("Starting gen...")
                SD.generate(
                    text_prompts=data['text_prompts'],
                    negative_prompts=data['negative_prompts'],
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
                    clip_skip=max(int(data['clip_skip']),1),
                    palette_fix=data['palette_fix'],
                    ckpt=ckpt_path,
                    vae=vae_path,
                    loras=lora_paths,
                    image_save_path=data['image_save_path'],
                    speed=data['speed'],
                    skip_grid=not data['save_grid'],
                    long_save_path=data['long_save_path'],
                    highres_fix=data['highres_fix'],
                    show_intermediates=data['show_intermediates'],
                    controlnet=data['controlnet'],
                    use_preprocessed_controlnet=data['use_preprocessed_controlnet'],
                    remove_background=data['remove_background'],
                    use_removed_background=data['use_removed_background'],
                    models_dir=data['models_dir'],
                )
                socketio.emit('job_done')

            except Exception as e:
                print(f"Generation failed! {e}")
                SD.clean_up()
                socketio.emit('job_done')
        return "Finished"

    @app.route('/xyplot', methods=['POST'])
    def xyplot():
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        xyplot_data = json.loads(request.data)
        x_key = xyplot_data["x_key"]
        x_values = xyplot_data["x_values"]
        y_key = xyplot_data["y_key"]
        y_values = xyplot_data["y_values"]
        if x_key == "lora":
            x_values = [os.path.basename(x["name"]) + ", weight: " + str(x["weight"]) for x in x_values]

        if y_key == "lora":
            y_values = [os.path.basename(y["name"]) + ", weight: " + str(y["weight"]) for y in y_values]

        x_path = '_'.join(list(map(str, x_values)))
        y_path = '_'.join(list(map(str, y_values)))
        xy_path = f"xyplot_{x_key}_{x_path}_{y_key}_{y_path}"[:100]
        # Replace invalid characters with an empty string
        invalid_pattern = re.compile(r'[<>:"/\\|?*\x00-\x1f\s]')
        xy_path = invalid_pattern.sub('', xy_path)
        image_save_path = os.path.join(xyplot_data["xyplots"][0]['image_save_path'], xy_path)

        for data in xyplot_data["xyplots"]:
            if not SD.running:
                try:
                    SD.running = True
                    mask_b64 = data['mask_image']
                    data['mask_image'] = data['mask_image'][:100] + "..."
                    init_image_str = data['init_image']
                    data['init_image'] = data['init_image'][:100] + "..."

                    print("Saving settings to folder...")
                    save_to_settings_folder(data)
                    ckpt_path = os.path.join(data['models_dir'], data['ckpt']).replace(os.sep, '/')
                    vae_path = os.path.join(data['models_dir'], 'Vaes', data['vae']).replace(os.sep, '/')
                    lora_paths = []
                    if len(data['lora']) > 0:
                        for lora in data['lora']:
                            lora_paths.append({
                                'path': os.path.join(data['models_dir'], 'Loras', lora['name']).replace(os.sep, '/'),
                                'weight': lora['weight']
                            })
                except Exception as e:
                    print(f"Failed to add to queue {e}")
                    SD.running = False
                    socketio.emit('job_done')
                    return
                try:
                    print("Starting gen...")
                    SD.generate(
                        text_prompts=data['text_prompts'],
                        negative_prompts=data['negative_prompts'],
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
                        clip_skip=max(int(data['clip_skip']),1),
                        palette_fix=data['palette_fix'],
                        ckpt=ckpt_path,
                        vae=vae_path,
                        loras=lora_paths,
                        image_save_path=image_save_path,
                        speed=data['speed'],
                        skip_grid=not data['save_grid'],
                        long_save_path=data['long_save_path'],
                        highres_fix=data['highres_fix'],
                        show_intermediates=data['show_intermediates'],
                        controlnet=data['controlnet'],
                        use_preprocessed_controlnet=data['use_preprocessed_controlnet'],
                        remove_background=data['remove_background'],
                        use_removed_background=data['use_removed_background'],
                        models_dir=data['models_dir'],
                    )
                    socketio.emit('job_done')

                except Exception as e:
                    print(f"Generation failed! {e}")
                    SD.clean_up()
                    socketio.emit('job_done')
      
        # Load the images into a list
        image_files = sorted(os.listdir(image_save_path))
        images = [plt.imread(os.path.join(image_save_path, f)) for f in image_files]

        # Calculate the number of rows and columns for the grid
        n_cols = len(x_values)
        n_rows = len(y_values)

        # Calculate the figure size based on the image size and grid layout
        fig_width = n_cols * 5
        fig_height = n_rows * 5

        # Create figure and subplots
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), sharex='col', sharey='row')

        # Add the images to the subplots
        for i, ax in enumerate(axs.flat):
            ax.imshow(images[i])
            ax.set_xticks([])
            ax.set_yticks([])
            row = i // n_cols
            col = i % n_cols
            ax.set_xlabel(f"{x_key}: {x_values[col]}", fontsize=10)
            ax.set_ylabel(f"{y_key}: {y_values[row]}", fontsize=10, rotation=90, labelpad=10)

        # Save the figure
        fig.tight_layout()
        plt.savefig(os.path.join(image_save_path, "xyplot.png"))

        return "Finished"

    @app.route('/shutdown', methods=['GET'])
    def shutdown():
        stop_queue()
        os._exit(0)


    @socketio.on('connect')
    def connected():
        '''event listener when client connects to the server'''
        print(request.sid)
        print('client has connected')
        socketio.emit('connect', {'data': f'id: {request.sid} is connected'})


    @socketio.on('message')
    def handle_message(data):
        '''event listener when client types a message'''
        print('data from the front end: ', str(data))
        socketio.emit('message', {'data': data, 'id': request.sid}, broadcast=True)


    @socketio.on('disconnect')
    def disconnected():
        '''event listener when client disconnects to the server'''
        print('user disconnected')
        socketio.emit('disconnect', f'user {request.sid} disconnected', broadcast=True)


    if __name__ == '__main__':
        socketio.run(app, host='127.0.0.1', port=5300, allow_unsafe_werkzeug=True)
except Exception as e:
    import time
    print("Runtime failed")
    print(e)
    time.sleep(120)
