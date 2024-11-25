import numpy as np
from PIL import Image
import cv2
max_i = 10

def load(image_dir, subject_id, i):
    i = min(i, max_i)

    image_path = f"{image_dir}/{subject_id}/{i+1}.jpg"
    image = Image.open(image_path).convert("RGB")

    # read annotation
    label_path = f"{image_dir}/{subject_id}/{i+1}.png"
    label_image = Image.open(label_path).convert("L")

    label_image = np.array(label_image)
    # values bigger than 128 are set to 1 smaller to 0
    label_image = np.where(label_image > 128, 1, 0)

    return image, label_image
