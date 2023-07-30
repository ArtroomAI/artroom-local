import requests 
import re
import os 
import tqdm 

def download_model(url: str, output_path: str) -> None:
    """
    Download a model from a given URL and save it to a specified output path.
    """
    # Create the directory for the output file if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Download the file from the URL
    with requests.get(url, stream=True) as response:
        response.raise_for_status()

        # Get the total file size in bytes
        file_size = int(response.headers.get('Content-Length', 0))

        # Create a tqdm progress bar
        with tqdm(total=file_size, unit='iB', unit_scale=True) as progress_bar:

            # Write the content to the output file
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        progress_bar.update(len(chunk))

def download_from_civitai(download_url: str, destination: str):
    response = requests.get(download_url)

    # Get the file name from the Content-Disposition header
    content_disposition = response.headers.get("Content-Disposition")

    if content_disposition:
        # Use regular expressions to extract the file name
        match = re.search('filename="(.+)"', content_disposition)
        if match:
            filename = match.group(1)

    download_destination = os.path.join(destination, filename)
    if not os.path.exists(download_destination):
        download_model(download_url, download_destination)