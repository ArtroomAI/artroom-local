import json
import threading
import time
import random
from glob import glob
import os
import re
from artroom_helpers.gpu_detect import get_gpu_architecture


def return_error(status, status_message='', content=''):
    if not status_message and status == 'Failure':
        status_message = 'Unknown Error'
    return {'status': status, 'status_message': status_message, 'content': content}


class QueueManager:
    def __init__(self, SD, artroom_path=None):
        self.SD = SD
        self.artroom_path = artroom_path
        self.queue = []
        self.running = {}
        self.delay = 5
        self.threads = None

        self.error = None

        self.read_queue_json()

    def resume_all(self):
        for idx in self.running.keys():
            self.running[idx] = True

    def pause_all(self):
        for idx in self.running.keys():
            self.running[idx] = False

    def any_is_running(self):
        return True in self.running.values()

    def parse_errors(self, error):
        self.error = return_error('Error')
        return self.error

    def clear_queue(self):
        self.queue = []

    def remove_from_queue(self, id):
        for i, item in enumerate(self.queue):
            if item['id'] == id:
                self.queue.pop(i)

    def update_delay(self, new_delay):
        if new_delay < 1:
            self.delay = 1
        else:
            self.delay = new_delay

    def write_queue_json(self):
        print('Writing to queue...')
        queue_json = {'queue': self.queue}
        with open(f'{self.artroom_path}/artroom/settings/queue.json', 'w') as outfile:
            json.dump(queue_json, outfile, indent=4)
        print('Wrote to Queue')

    def read_queue_json(self):
        if os.path.exists(f'{self.artroom_path}/artroom/settings/queue.json'):
            queue_json = json.load(
                open(f'{self.artroom_path}/artroom/settings/queue.json'))
            self.queue = queue_json['queue']
        else:
            self.queue = []

        if os.path.exists(f'{self.artroom_path}/artroom/settings/s.json'):
            queue_json = json.load(
                open(f'{self.artroom_path}/artroom/settings/queue.json'))
            self.queue = queue_json['queue']

    def set_artroom_path(self, path):
        self.artroom_path = path

    def add_to_queue(self, data):
        print(f'Adding to queue...')
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
                '%InstallPath%', self.artroom_path)
        data['ckpt'] = data['ckpt'].replace(os.sep, '/')

        if '%UserProfile%' in data['ckpt_dir']:
            data['ckpt_dir'] = data['ckpt_dir'].replace(
                '%UserProfile%', os.environ['USERPROFILE'])
        if '%InstallPath%' in data['ckpt_dir']:
            data['ckpt_dir'] = data['ckpt_dir'].replace(
                '%InstallPath%', self.artroom_path)
        data['ckpt_dir'] = data['ckpt_dir'].replace(os.sep, '/')

        if data['aspect_ratio'] == 'Init Image':
            # Load image sets it to be equal to init_image dimensions
            data['width'] = 0
            data['height'] = 0

        self.queue.append(data)
        self.save_settings_cache(data)
        self.write_queue_json()
        print(f'Queue written, length of queue: {len(self.queue)}')

    def save_to_settings_folder(self, data):
        print("Saving settings...")
        if self.SD.long_save_path:
            image_folder = os.path.join(data['image_save_path'], data['batch_name'], re.sub(
                r'\W+', '', '_'.join(data['text_prompts'].split())))[:150]
            os.makedirs(image_folder, exist_ok=True)
            os.makedirs(image_folder + '/settings', exist_ok=True)
            sd_settings_count = len(glob(image_folder + '/settings/*.json'))
            with open(f'{image_folder}/settings/sd_settings_{data["seed"]}_{sd_settings_count}.json', 'w') as outfile:
                json.dump(data, outfile, indent=4)
        else:
            image_folder = os.path.join(
                data['image_save_path'], data['batch_name'])
            os.makedirs(image_folder, exist_ok=True)
            os.makedirs(image_folder + '/settings', exist_ok=True)
            sd_settings_count = len(glob(image_folder + '/settings/*.json'))
            prompt_name = re.sub(
                r'\W+', '', "_".join(data["text_prompts"].split()))[:100]
            with open(f'{image_folder}/settings/sd_settings_{prompt_name}_{data["seed"]}_{sd_settings_count}.json',
                      'w') as outfile:
                json.dump(data, outfile, indent=4)
        print("Settings saved")

    def save_settings_cache(self, data):
        with open(f'{self.artroom_path}/artroom/settings/sd_settings.json', 'r') as infile:
            existing_data = json.load(infile)
        existing_data.update(data)
        with open(f'{self.artroom_path}/artroom/settings/sd_settings.json', 'w') as outfile:
            json.dump(existing_data, outfile, indent=4)

    def generate(self, next_gen):
        mask_b64 = next_gen['mask_image']
        next_gen['mask_image'] = ''
        init_image_str = next_gen['init_image']
        print("Saving settings to folder...")
        self.save_to_settings_folder(next_gen)
        ckpt_path = os.path.join(next_gen['ckpt_dir'], next_gen['ckpt']).replace(os.sep, '/')
        vae_path = os.path.join(next_gen['ckpt_dir'], next_gen['vae']).replace(os.sep, '/')
        try:
            print("Starting gen...")
            self.SD.generate(
                text_prompts=next_gen['text_prompts'],
                negative_prompts=next_gen['negative_prompts'],
                batch_name=next_gen['batch_name'],
                init_image_str=init_image_str,
                strength=next_gen['strength'],
                mask_b64=mask_b64,
                invert=next_gen['invert'],
                n_iter=int(next_gen['n_iter']),
                steps=int(next_gen['steps']),
                H=int(next_gen['height']),
                W=int(next_gen['width']),
                seed=int(next_gen['seed']),
                sampler=next_gen['sampler'],
                cfg_scale=float(next_gen['cfg_scale']),
                ckpt=ckpt_path,
                vae=vae_path,
                image_save_path=next_gen['image_save_path'],
                speed=next_gen['speed'],
                skip_grid=not next_gen['save_grid'],
            )
        except Exception as e:
            print(f'Failure: {e}')
            self.parse_errors(e)
            self.running = {}
            self.SD.running = False

            self.SD.lock_models(lock=False)  # unlock

    def run_queue(self):
        self.running[str(threading.get_native_id())] = True
        while self.running[str(threading.get_native_id())]:
            if len(self.queue) > 0 and not self.SD.stage == "Loading Model":
                queue_item = self.queue.pop()
                self.SD.lock_models()
                try:
                    self.generate(queue_item)
                except Exception as e:
                    print(f"Failed to generate: {e}")
                try:
                    self.remove_from_queue(queue_item['id'])
                except Exception as e:
                    print(f"Failed to remove: {e}")
                self.write_queue_json()
            else:
                pass
            time.sleep(self.delay)
            if len(self.queue) == 0:
                self.running[str(threading.get_native_id())] = False
        if not self.any_is_running():
            print("All threads are done")
            self.SD.lock_models(lock=False)  # unlock
