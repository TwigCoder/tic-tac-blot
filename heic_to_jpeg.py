# libcamera uses the JPEG format, and the iPhone uses HEIC, because it has to be special. 
# No way am I taking a snapshot of each image using the raspberry pi camera for training.
# This script will convert HEIC images to JPEG because I desire batch conversion.

import os
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

def convert_heic_to_jpeg(heic_folder='data/images_heic', jpeg_folder='data/images', quality=100):
    heic_files = [f for f in os.listdir(heic_folder)]

    if not heic_files:
        return "NO FILES FOUND!"
    
    for i, f in enumerate(heic_files):
        try:
            input_path = os.path.join(heic_folder, f)
            output_f = os.path.splitext(f)[0] + '.jpeg'
            output_path = os.path.join(jpeg_folder, output_f)

            with Image.open(input_path) as img:
                img = img.convert('RGB')
            
            img.save(output_path, 'JPEG', quality=quality)

        except Exception as e:
            print(f'ERROR WHILE CONVERTING: {e}')

convert_heic_to_jpeg()
