import numpy as np
from PIL import Image
import cv2
import os
def load(image_dir, subject_id, i):
    image_path = f"{image_dir}/{subject_id}/{i:02d}"

    # check if image exists as png or jpg
    if not os.path.exists(f"{image_path}.png"):
        image_path = f"{image_path}.jpg"
    else:
        image_path = f"{image_path}.png"
    
    image = Image.open(image_path).convert("RGB")

    # read annotation
    label_path = f"{image_dir}/{subject_id}/{i:02d}-inst.png"
    label_image = Image.open(label_path).convert("L")

    label_image = np.array(label_image)


    return image, label_image