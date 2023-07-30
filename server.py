try:
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    import logging

    logging.getLogger("xformers").addFilter(
        lambda record: 'A matching Triton is not available' not in record.getMessage())
    import numpy as np
    import json
    import ctypes
    import shutil
    import os
    import re
    import sys
    import platform
    sys.path.append(os.curdir)
    sys.path.append("lora_training/")
    sys.path.append("backend/ComfyUI/")
    from backend.ComfyUI import comfy
    from upscale import Upscaler
    from stable_diffusion import StableDiffusion, Mute, Model
    from artroom_helpers import support
    from artroom_helpers.generation.preprocess import mask_from_face
    from artroom_helpers.toast_status import toast_status
    from model_merger import ModelMerger

    from flask import Flask, request, jsonify
    from flask_socketio import SocketIO
    from PIL import Image, ImageOps
    from glob import glob

    if sys.platform.startswith('win32'):
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


    class InterceptedStdout:
        def __init__(self, original_stdout, callback):
            self.original_stdout = original_stdout
            self.callback = callback

        def write(self, text):
            processed_text = self.callback(text)
            self.original_stdout.write(processed_text)

        def flush(self):
            self.original_stdout.flush()


    progress_pattern = re.compile(r'\[(.*?)\]')
    current_num_pattern = re.compile(r'\d+(?=/)')


    def your_callback_function(text: str):
        socketio.emit('messagesss', text)

        if 'OutOfMemoryError' in text:
            socketio.emit('status', toast_status(title="Cuda Out of Memory Error - try generating smaller images",
                                                 status="error"))

        try:
            active_model = SD.active_model
            if active_model is not None:
                match = progress_pattern.search(text)
                if match:
                    parts = match.group(1).split(',')
                    time_spent = parts[0].split('<')[0].strip()
                    eta = parts[0].split('<')[1].strip()
                    iterations_per_sec = parts[1].strip()

                    current_step_match = current_num_pattern.search(text)
                    current_step = int(current_step_match.group()) if current_step_match else 0

                    socketio.emit('get_progress', {
                        'current_step': current_step,
                        'total_steps': active_model.steps,
                        'current_num': active_model.current_num - 1,
                        'total_num': active_model.total_num,
                        'time_spent': time_spent,
                        'eta': eta,
                        'iterations_per_sec': iterations_per_sec
                    })
        except Exception as e:
            pass

        return text


    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stderr = InterceptedStdout(original_stderr, your_callback_function)
    sys.stdout = InterceptedStdout(original_stdout, your_callback_function)


    @socketio.on('upscale')
    def upscale(data):
        print('Running upscale...', data)
        if UP.running:
            print('Failure to upscale, upscale is already running')
            socketio.emit('status', toast_status(
                title='Upscale Failed', description='Upscale is already running',
                status='error', duration=5000, isClosable=False))
            return
        if len(data['upscale_images']) == 0:
            print('Failure to upscale, please select an image')
            socketio.emit('status', toast_status(
                title='Upscale Failed', description='Please select an image',
                status='error', duration=5000, isClosable=False))
            return

        if 'upscale_dest' not in data or data['upscale_dest'] == '':
            data['upscale_dest'] = os.path.dirname(data['upscale_images'][0]) + '/upscale_outputs'

        response = UP.upscale(
            data['models_dir'],
            data['upscale_images'],
            data['upscaler'],
            data['upscale_factor'],
            data['upscale_dest'])
        
        socketio.emit('upscale_completed', {'upscale_dest': data['upscale_dest']})

        socketio.emit('status', toast_status(
            title='Upscale Completed', description='Your upscale has completed',
            status='success', duration=2000, isClosable=False))
        return


    @socketio.on('merge_models')
    def merge_models(data):
        ModelMerger(data).run()


    @socketio.on('train')
    def train(data):
        print("Starting Training...")

        def check_SDXL(path):
            # Clear up SD so you don't run out of vram during this step
            SD.running = False
            SD.active_model = Model(ckpt='', models_dir='', socketio=socketio)

            with Mute():
                _, clip, _, _ = comfy.sd.load_checkpoint_guess_config(
                    path,
                    output_vae=False,
                    output_clip=True,
                    embedding_directory='')
                if hasattr(clip.cond_stage_model, "clip_l"):
                    return True
                return False

        print("Checking for SDXL...")
        SDXL = check_SDXL(data['model'])
        if SDXL:
            print('Training SDXL model')
            train_file = 'SDXL_train_network.py'
        else:
            print('Training non-SDXL model')
            train_file = 'train_network.py'

        import subprocess
        python_path = sys.executable
        base_path = os.path.dirname(os.path.dirname(sys.executable))  # First one takes you to artroom_backend
        accelerate_path = os.path.join(os.path.dirname(python_path), "Scripts" if os.name == "nt" else "bin",
                                       "accelerate.exe")
        print("Setting up training...")

        try:
            setup_command_libraries = f"{python_path} -m pip install ./lora_training"
            _ = subprocess.run(setup_command_libraries, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except:
            print("Installation of lora setup failed, please rerun Artroom as admin and try again. We will fix this requirement soon!")
            return 
        
        print('Checking lora dependencies...')
        try:
            setup_command_dependencies = f"{python_path} -m pip install -r requirements_lora.txt --user --no-warn-script-location"
            _ = subprocess.run(setup_command_dependencies, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as e:
            print(f"Installing lora dependencies failed. Please reach out on discord or support@artroom.ai for help. Error code: {e}")

        training_images_path = os.path.join(base_path, "training_data", data["name"])
        # Clean slate of images
        if os.path.isdir(training_images_path):
            shutil.rmtree(training_images_path)
        os.makedirs(training_images_path, exist_ok=True)

        for file in data["images"]:
            trigger_path = os.path.join(training_images_path, f"{data['numRepeats']}_{data['trigger_word']}")
            os.makedirs(trigger_path, exist_ok=True)
            shutil.copy(file, os.path.join(trigger_path, os.path.basename(file)))

        command = (
            f'"{python_path}" '
            f'"{accelerate_path}" launch '
            '--num_cpu_threads_per_process=16 '
            f'lora_training/{train_file} '
            f'--train_data_dir="{training_images_path}" '
            f'--output_name="{data["name"]}" '
            f'--pretrained_model_name_or_path="{data["model"]}" '
            f'--resolution={data["resolution"]} '
            f'--network_alpha={data["networkAlpha"]} '
            f'--max_train_steps={data["maxTrainSteps"]} '
            f'--clip_skip=1 '
            f'--text_encoder_lr={data["textEncoderLr"]} '
            f'--unet_lr={data["unetLr"]} '
            f'--network_dim={data["networkDim"]} '
            f'--lr_scheduler_num_cycles={data["lrSchedulerNumCycles"]} '
            f'--learning_rate={data["learningRate"]} '
            f'--lr_scheduler={data["lrScheduler"]} '
            f'--train_batch_size={data["trainBatchSize"]} '
            f'--save_every_n_epochs={data["saveEveryNEpochs"]} '
            f'--optimizer_type={data["optimizerType"]} '
            f'--bucket_reso_steps={data["bucketResoSteps"]} '
            f'--min_bucket_reso={data["minBucketReso"]} '
            f'--max_bucket_reso={data["maxBucketReso"]} '
            f'--output_dir="{os.path.join(data["modelsDir"], "Lora")}" '
            '--save_model_as=safetensors '
            '--caption_extension=.txt '
            '--mixed_precision=fp16 '
            '--save_precision=fp16 '
            '--enable_bucket '
            '--network_module=networks.lora '
            '--cache_latents '
            '--xformers '
            '--bucket_no_upscale '
            '--full_fp16 '
        )
        if SDXL:
            command += '--no_half_vae '
        if "Prodigy" in data["optimizerType"]:
            command+= '--optimizer_args decouple=True weight_decay=0.5 betas=0.9,0.99 use_bias_correction=False '
        try:
            subprocess.run(command, shell=True, check=True)
            print("Lora Training has completed!")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred: {e}")


    def save_to_settings_folder(data):
        print("Saving settings...")
        try:
            if data['long_save_path']:
                image_folder = os.path.join(data['image_save_path'], re.sub(
                    r'\W+', '', '_'.join(data['text_prompts'].split())))[:150]
                os.makedirs(image_folder, exist_ok=True)
                os.makedirs(os.path.join(image_folder, 'settings'), exist_ok=True)
                sd_settings_count = len(glob(image_folder + '/settings/*.json'))
                with open(f'{image_folder}/settings/sd_settings_{data["seed"]}_{sd_settings_count}.json',
                          'w') as outfile:
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
        from artroom_helpers.process_controlnet_images import preview_controlnet, HWC3
        try:
            print(f'Previewing controlnet {data["controlnet"]}')
            image = support.b64_to_image(data['initImage'])
            w, h = image.size
            # resize to integer multiple of 32
            w, h = map(lambda x: x - x % 64, (w, h))
            image = image.resize((w, h), resample=Image.LANCZOS)
            image = HWC3(np.array(image))
            image = preview_controlnet(image, data["controlnet"], data["models_dir"])
            print("DONE PREVIEW")
            image = HWC3(np.array(image))
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
                vae_path = os.path.join(data['models_dir'], 'Vae', data['vae']).replace(os.sep, '/')
                lora_paths = []
                if len(data['loras']) > 0:
                    for lora in data['loras']:
                        lora_paths.append({
                            'path': os.path.join(data['models_dir'], 'Lora', lora['name']).replace(os.sep, '/'),
                            'name': lora['name'],
                            'weight': lora['weight']
                        })
            except Exception as e:
                print(f"Failed to add to queue {e}")
                SD.running = False
                socketio.emit('job_done')
                return
            try:
                print("Starting gen...")
                print(data)
                SD.generate(
                    image_save_path=data['image_save_path'],
                    long_save_path=data['long_save_path'],
                    highres_fix=data['highres_fix'],
                    show_intermediates=data['show_intermediates'],
                    use_preprocessed_controlnet=data['use_preprocessed_controlnet'],
                    remove_background=data['remove_background'],
                    use_removed_background=data['use_removed_background'],
                    models_dir=data['models_dir'],
                    text_prompts=data['text_prompts'],
                    negative_prompts=data['negative_prompts'],
                    init_image_str=init_image_str,
                    mask_str=mask_b64,
                    invert=data['invert'],
                    steps=int(data['steps']),
                    H=int(data['height']),
                    W=int(data['width']),
                    strength=data['strength'],
                    cfg_scale=float(data['cfg_scale']),
                    seed=int(data['seed']),
                    sampler=data['sampler'],
                    n_iter=int(data['n_iter']),
                    batch_size=1,
                    ckpt=ckpt_path,
                    vae=vae_path,
                    loras=lora_paths,
                    controlnet=data['controlnet'],
                    background_removal_type="none",
                    clip_skip=max(int(data['clip_skip']), 1),
                    generation_mode=data.get('generation_mode'),
                    highres_steps=data.get('highres_steps'),
                    highres_strength=data.get('highres_strength')
                )
            except Exception as e:
                print(f"Generation failed! {e}")
                SD.running = False
            socketio.emit('job_done')


    @socketio.on('stop_queue')
    def stop_queue():
        SD.interrupt()
        socketio.emit("status",
                      toast_status(title="Queue interrupted. Generations will stop after this one.", status="info",
                                   duration=2000))


    @app.route('/xyplot', methods=['POST'])
    def xyplot():
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        xyplot_data = json.loads(request.data)
        x_key = xyplot_data["x_key"]
        x_values = xyplot_data["x_values"]
        y_key = xyplot_data["y_key"]
        y_values = xyplot_data["y_values"]
        if x_key == "loras":
            x_values = [os.path.basename(x["name"]) + ", weight: " + str(x["weight"]) for x in x_values]

        if y_key == "loras":
            y_values = [os.path.basename(y["name"]) + ", weight: " + str(y["weight"]) for y in y_values]

        x_path = '_'.join(list(map(str, x_values)))
        y_path = '_'.join(list(map(str, y_values)))
        # xy_path = f"xyplot_{x_key}_{x_path}_{y_key}_{y_path}"[:100]
        xy_path = f"xyplot_{y_path}_{y_key}"[:100]
        # Replace invalid characters with an empty string
        invalid_pattern = re.compile(r'[<>:"/\\|?*\x00-\x1f\s]')
        xy_path = invalid_pattern.sub('', xy_path)
        image_save_path = os.path.join(xyplot_data["xyplots"][0]['image_save_path'], xy_path)

        # Check if the folder exists and create it if it doesn't
        base_count = 0
        while os.path.exists(f"{image_save_path}_{base_count:03}"):
            base_count += 1
        image_save_path = f"{image_save_path}_{base_count:03}"
        os.makedirs(image_save_path, exist_ok=True)

        for data in xyplot_data["xyplots"]:
            try:
                SD.running = True
                mask_b64 = data['mask_image']
                data['mask_image'] = data['mask_image'][:100] + "..."
                init_image_str = data['init_image']
                data['init_image'] = data['init_image'][:100] + "..."

                print("Saving settings to folder...")
                save_to_settings_folder(data)
                ckpt_path = os.path.join(data['models_dir'], data['ckpt']).replace(os.sep, '/')
                vae_path = os.path.join(data['models_dir'], 'Vae', data['vae']).replace(os.sep, '/')
                lora_paths = []
                if len(data['loras']) > 0:
                    for lora in data['loras']:
                        lora_paths.append({
                            'path': os.path.join(data['models_dir'], 'Lora', lora['name']).replace(os.sep, '/'),
                            'name': lora['name'],
                            'weight': lora['weight']
                        })
            except Exception as e:
                print(f"Failed to add to queue {e}")
                SD.running = False
                socketio.emit('job_done')
                return
            try:
                print("Starting gen...")
                print(data)
                SD.generate(
                    image_save_path=data['image_save_path'],
                    long_save_path=data['long_save_path'],
                    highres_fix=data['highres_fix'],
                    show_intermediates=data['show_intermediates'],
                    use_preprocessed_controlnet=data['use_preprocessed_controlnet'],
                    remove_background=data['remove_background'],
                    use_removed_background=data['use_removed_background'],
                    models_dir=data['models_dir'],
                    text_prompts=data['text_prompts'],
                    negative_prompts=data['negative_prompts'],
                    init_image_str=init_image_str,
                    mask_str=mask_b64,
                    invert=data['invert'],
                    steps=int(data['steps']),
                    H=int(data['height']),
                    W=int(data['width']),
                    strength=data['strength'],
                    cfg_scale=float(data['cfg_scale']),
                    seed=int(data['seed']),
                    sampler=data['sampler'],
                    n_iter=int(data['n_iter']),
                    batch_size=1,
                    ckpt=ckpt_path,
                    vae=vae_path,
                    loras=lora_paths,
                    controlnet=data['controlnet'],
                    background_removal_type="none",
                    clip_skip=max(int(data['clip_skip']), 1),
                    generation_mode=data.get('generation_mode'),
                    highres_steps=data.get('highres_steps'),
                    highres_strength=data.get('highres_strength')
                )
            except Exception as e:
                print(f"Generation failed! {e}")
                SD.running = False
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
            ax.set_xlabel(f"{x_key}: {x_values[col]}", fontsize=15)
            ax.set_ylabel(f"{y_key}: {y_values[row]}", fontsize=15, rotation=90, labelpad=10)

        # Save the figure
        fig.tight_layout()
        plt.savefig(os.path.join(image_save_path, f"{xy_path}.png"))


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
        socketio.emit('message', {'data': data, 'id': request.sid})


    @socketio.on('disconnect')
    def disconnected():
        '''event listener when client disconnects to the server'''
        print('user disconnected')
        socketio.emit('disconnect', f'user {request.sid} disconnected')


    if __name__ == '__main__':
        socketio.run(app, host='127.0.0.1', port=5300, allow_unsafe_werkzeug=True, debug=False)
except Exception as e:
    import time

    print("Runtime failed")
    print(e)
    if "No module" in str(e):
        print("Please go to Settings and Update Packages")

    time.sleep(120)
