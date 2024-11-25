import numpy as np
import os
from PIL import Image
from enum import Enum
TRUCK_IDS = [
    ('visual_truck_base_base', 1),
    ('visual_truck_base_cab', 2),
    ('visual_truck_base_loading_platform', 3),
    ('visual_truck_base_mounting_base', 4),
    ('visual_truck_base_wheel_b_l', 5),
    ('visual_truck_base_wheel_b_r', 6),
    ('visual_truck_base_wheel_c_l', 7),
    ('visual_truck_base_wheel_c_r', 8),
    ('visual_truck_base_wheel_f_l', 9),
    ('visual_truck_base_wheel_f_r', 10),
    ('visual_truck_crane_arm', 11),
    ('visual_truck_crane_boom', 12),
    ('visual_truck_crane_control_box', 13),
    ('visual_truck_crane_double_joint_link', 14),
    ('visual_truck_crane_inner_jaw', 15),
    ('visual_truck_crane_inner_telescope', 16),
    ('visual_truck_crane_outer_jaw', 17),
    ('visual_truck_crane_outer_telescope', 18),
    ('visual_truck_crane_rotator_upper_part', 19),
    ('visual_truck_crane_slewing_column', 20),
    ('visual_truck_crane_tool_center_point', 21)
]



HIGH = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], [18], [19], [20], [21]]
HIGH_MERGE_MIRRORED = [[1], [2], [3], [4], [5, 6], [7, 8], [9, 10], [11], [12], [13], [14], [15, 17], [16], [18], [19], [20], [21]]
MEDIUM = [[1], [2], [3], [4], [5, 6], [7, 8], [9, 10], [11], [12], [13], [15, 17], [16], [18], [19, 14, 21], [20]]
LOW = [[1,3,4], [2], [5, 6, 7, 8, 9, 10], [11,16,18], [12], [13,20], [15, 17, 19, 14, 21]]
TRUCK_CRANE = [[1,3,4,2, 5, 6, 7, 8, 9, 10], [11,16,18,12,13,20,15, 17, 19, 14, 21]]
TRUCK = [[1,3,4,2, 5, 6, 7, 8, 9, 10,11,16,18,12,13,20,15, 17, 19, 14, 21]]
LOADING_PLATFORM = [[1]]

GRANULARITY_MERGE_LIST = [HIGH, HIGH_MERGE_MIRRORED, MEDIUM, LOW, TRUCK_CRANE, TRUCK, LOADING_PLATFORM]

class AnnotationGranularity(Enum):
    HIGH = 0
    HIGH_MERGE_MIRRORED = 1
    MEDIUM = 2
    LOW = 3
    TRUCK_CRANE = 4
    TRUCK = 5
    LOADING_PLATFORM = 6

NAMES = {
    AnnotationGranularity.TRUCK: ["Background", "Truck"],
    AnnotationGranularity.TRUCK_CRANE: ["Background", "Truck", "Crane"],
    AnnotationGranularity.LOW: ["Background", "Base", "Cab", "Wheel", "Crane Telescopic", "Crane Boom", "Crane Base", "Gripper"],
}

class AnnotatedImageLoader():
    def __init__(self, directory: str, granularity: AnnotationGranularity = AnnotationGranularity.HIGH):
        self.directory = directory

    def merge_annotated_masks(self, annotation: Image, granularity: AnnotationGranularity = AnnotationGranularity.HIGH):
        # take list according to granularity
        indices_to_merge = GRANULARITY_MERGE_LIST[granularity.value]
        
        annotation_merged = np.zeros_like(annotation) #np.array(annotation)
        
        # overwrite all the indices with the new index
        for idx, indices in enumerate(indices_to_merge):
            for i in indices:

                annotation_merged[np.array(annotation) == i] = idx + 1
                #annotation_merged[annotation_merged == i] = idx + 1

        return Image.fromarray(annotation_merged)

    def get_names(self, granularity: AnnotationGranularity = AnnotationGranularity.HIGH):
        return NAMES[granularity]

    def load(self, idx, granularity: AnnotationGranularity = AnnotationGranularity.HIGH, load_annotaion=True):
        # prepare paths
        path_to_image = os.path.join(self.directory, f'{idx:02d}.jpg')
        image = Image.open(path_to_image).convert('RGB')

        if load_annotaion:
            path_to_annotation = os.path.join(self.directory, f'{idx:02d}-inst.png')
            annotation = Image.open(path_to_annotation)
            # remove alpha channel
            annotation = annotation.convert('L')
        else:
            annotation = Image.fromarray(np.zeros((image.size[1], image.size[0]), dtype=np.uint8))
        
        if granularity == None:
            annotation_merged = annotation
        else:            
            # merge the annotation according to the granularity
            annotation_merged = self.merge_annotated_masks(annotation, granularity)

        # get all the colors in the annotation
        colors = annotation_merged.getcolors()
        colors = [c[1] for c in colors]

        # create masks for each color
        masks = {}
        for color in colors:
            annotation_mask = np.array(annotation_merged)
            annotation_mask = annotation_mask == color
            masks[color] = annotation_mask
        

        return image, np.array(annotation_merged), masks, path_to_image

    


