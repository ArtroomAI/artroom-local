import os
import json
import zipfile
from artroom_helpers.support import download_model_from_google

# Open the JSON file
with open('startup_files.json', 'r') as f:
    # Load JSON data from file
    data = json.load(f)

os.makedirs('models', exist_ok=True)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

# Download models
for file in data['file']:
    destination = 'models/'
    filename = os.path.join(file['destination'], file['name'])
    if not os.path.exists(os.path.join(destination, filename)):
        if file['name'] == 'sd_v_21.ckpt' or file['name'] == 'sd_xl_base_0.9.safetensors' or file['name'] == 'v1-5-pruned.ckpt':
            download_model_from_google(filename)
    else:
        print(f"{filename} already exists")