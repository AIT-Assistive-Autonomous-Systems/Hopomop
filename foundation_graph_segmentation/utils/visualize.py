import numpy as np
import colorcet as cc
import seaborn as sns
import json

from matplotlib import pyplot as plt

def visualize_segmentation(data, output_path):
    images = data["images"]
    masks = data["masks"]
    masks_gt = data["masks_gt"]
    points = data["points"]
    predictions = data["predictions"]

    color = sns.color_palette(cc.glasbey_light, 20)

    for i, (image, mask) in enumerate(zip(images, masks)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)

        new_mask = np.zeros_like(image)

    

        for col_i, col in enumerate(color):
            new_mask[mask == col_i] = np.array(col) * 255
            
        new_mask[mask == 0] = [0, 0, 0]
            

        plt.imshow(new_mask, alpha=0.5)
        plt.axis("off")
        plt.title(f"Image {i}")
        plt.savefig(f"{output_path}/image_{i}.png")