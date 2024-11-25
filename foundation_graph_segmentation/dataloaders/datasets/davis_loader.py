import numpy as np
from PIL import Image
import cv2
import os
image_path_pre = "JPEGImages/480p"
label_path_pre = "Annotations/480p"



color_mapping = [
    [0, 0, 0],
    [0,0,128],
    [0,128,0],
    [0,128,128],
    [128,0,0],
    [128,0,128],
    [128,128,0],
    [128,128,128],
    [0,0,64],
]

# switch r and b

color_mapping = np.array(color_mapping)
color_mapping = color_mapping[:, [2, 1, 0]]



def get_all_colors(dataset_dir, subject_id):

    colors = []

    # get image count
    image_count = len(os.listdir(f"{dataset_dir}/{image_path_pre}/{subject_id}"))

    for i in range(image_count):
        label_path = f"{dataset_dir}/{label_path_pre}/{subject_id}/{i:05}.png"
        label_image = np.array(Image.open(label_path))

        pixels = label_image.reshape(-1, label_image.shape[-1])

        # get unique values
        unique_values = np.unique(label_image)

        # if not part of colors, add to colors
        for value in unique_values:
            if value not in colors:
                colors.append(value)
        
    # sort colors so black is first
    colors.sort()

    return colors
        


def load(dataset_dir, subject_id, i, all_colors):

    image_path = f"{dataset_dir}/{image_path_pre}/{subject_id}/{i:05}.jpg"

    image = Image.open(image_path).convert("RGB")

    # read annotation
    label_path = f"{dataset_dir}/{label_path_pre}/{subject_id}/{i:05}.png"
    label_image = np.array(Image.open(label_path).convert("RGB"))

    # convert rgb images to unit8 id images
    label_mask = np.zeros(label_image.shape[:2], dtype=np.uint8)
    pixels = label_image.reshape(-1, label_image.shape[-1])

    for i, value in enumerate(color_mapping):

        label_mask[np.all(label_image == value, axis=-1)] = i

    return image, label_mask

# def main():
#     # print all colors 
#     dataset_dir = "/data/dataset/davis_data"
#     subject_dir = "/data/dataset/davis_data/Annotations/480p/"
    
#     import cv2

#     # get all colors in the dataset
#     all_colors = {}
#     unique_values1 = []
#     for subject_id in os.listdir(subject_dir):
#         label_path = f"{dataset_dir}/{label_path_pre}/{subject_id}/{0:05}.png"
        
        
#         label_image = np.array(Image.open(label_path).convert("RGB"))

#         pixels = label_image.reshape(-1, label_image.shape[-1])

#         # get unique values (rgb)
#         unique_values = np.unique(pixels, axis=0)

#         # if there are more than 3 unique values, say hello
#         if len(unique_values) > 4:
#             print(f"Subject {subject_id} has more than 3 unique values")



            

# main()


#     print(all_colors)


    
# main()