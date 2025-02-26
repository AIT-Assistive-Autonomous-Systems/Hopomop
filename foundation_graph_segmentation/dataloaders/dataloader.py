import numpy as np
import cv2
from PIL import Image
import sys
import gc

# torch related imports
import torch
from torch.utils.data import Dataset
from torch_geometric.utils.convert import from_networkx
import networkx as nx
from sklearn.metrics.pairwise import euclidean_distances
from dataloaders.datasets import ffs_loader
from dataloaders.datasets import davis_loader
from dataloaders.datasets import truck_loader
from dataloaders.datasets import simple_loader
import albumentations as A
sys.path.append("foundation_graph_segmentation")

from interest_point_detectors.superpoint.utils import read_image


def create_adjacency_matrix(superpixel_centers, clip_seg_feature_summed, k=16):
    """Create adjacency matrix for superpixels based on k nearest neighbors.

    Args:
    - superpixel_centers: (N, 2) array of superpixel centers
    - k: Number of nearest neighbors
    """

    # Compute pairwise distances between superpixel centers
    pairwise_distances = euclidean_distances(superpixel_centers, superpixel_centers)

    # Find k nearest neighbors for each superpixel
    nearest_neighbors = np.argsort(pairwise_distances, axis=1)[
        :, 1 : k + 1
    ]  # Exclude self, hence starting from index 1

    # Create adjacency matrix
    num_superpixels = len(superpixel_centers)
    adjacency_matrix = np.zeros((num_superpixels, num_superpixels), dtype=int)

    for i in range(num_superpixels):

        # Get distance weight
        # distance_weight = 1 - pairwise_distances[i, nearest_neighbors[i]] / np.max(
        #     pairwise_distances[i, nearest_neighbors[i]]
        # )
   
        adjacency_matrix[i, nearest_neighbors[i]] = 1
    return adjacency_matrix


def apply_clip_seg(
    clip_seg_model, clip_seg_processor, clip_seg_prompts, image, shape, device
):
    """Apply CLIPSeg model to image and return feature map.

    Args:
    - clip_seg_model: CLIPSeg model
    - clip_seg_processor: CLIPSeg processor
    - clip_seg_prompts: List of prompts for CLIPSeg
    - image: PIL image
    - shape: Shape of the image
    - device: Device to run model on
    """

    # clip seg input
    clip_seg_input = clip_seg_processor(
        clip_seg_prompts,
        images=[image] * len(clip_seg_prompts),
        return_tensors="pt",
        padding=True,
    )


    # inputs to device
    for key in clip_seg_input:
        clip_seg_input[key] = clip_seg_input[key].to(device)

    # call the model
    clip_seg_feature_out = clip_seg_model(**clip_seg_input).logits

    # sigmoid the output logits
    clip_seg_feature_out = torch.sigmoid(clip_seg_feature_out)

    # convert to numpy
    clip_seg_feature_out_numpy = clip_seg_feature_out.detach().cpu().numpy()

    # sum over all features
    if len(clip_seg_feature_out_numpy.shape) == 3:
        clip_seg_feature_summed = np.zeros(
            (clip_seg_feature_out_numpy.shape[1], clip_seg_feature_out_numpy.shape[2])
        )

        for i, feature in enumerate(clip_seg_feature_out_numpy):
            clip_seg_feature_summed += feature
    else:
        # if only one feature map, we can skip the summation
        clip_seg_feature_summed = clip_seg_feature_out_numpy

    # reverse shape
    reversed_shape = (shape[1], shape[0])

    # scale to image size
    clip_seg_feature_summed = cv2.resize(
        clip_seg_feature_summed, reversed_shape, interpolation=cv2.INTER_CUBIC
    )

    return clip_seg_feature_summed


class SegDataset(Dataset):
    def __init__(
        self,
        index_list,
        image_dir,
        dataset_type,
        subject_id,
        min_points=512,
        graph_neighbours=30,
        super_glue_model=None,
        clip_seg_model=None,
        clip_seg_processor=None,
        clip_seg_prompts=None,
        use_clip_seg_features=False,
        use_position_feature=False,
        random_points=False,
        augment=False,
        train=False,
        device="cuda",
    ):
        self.image_count = len(index_list)
        self.image_dir = image_dir
        self.dataset_type = dataset_type
        self.subject_id = subject_id

        self.labels = []
        self.data = []
        self.class_count = {}
        self.pixel_count = {}

        self.super_glue_model = super_glue_model
        self.clip_seg_model = clip_seg_model
        self.clip_seg_processor = clip_seg_processor
        self.clip_seg_prompts = clip_seg_prompts

        self.graph_neighbours = graph_neighbours
        self.min_points = min_points

        self.images = []
        self.annotations = []
        self.masks = []
        self.clip_seg_features = []

        self.use_clip_seg_feature = use_clip_seg_features
        self.use_position_feature = use_position_feature

        self.random_points = random_points
        self.augment = augment
        # feature count
        self.feature_count = 256
        if self.use_clip_seg_feature:
            self.feature_count += 1
        if self.use_position_feature:
            self.feature_count += 2

        self.device = device
        self.train = train

        if self.dataset_type == "truck":
            truck_data_loader = truck_loader.AnnotatedImageLoader(image_dir)
            # string to enum subject id
            granularity = truck_loader.AnnotationGranularity[subject_id]

        for i in index_list:
            # read imag
            if self.dataset_type == "davis":
                all_colors = davis_loader.get_all_colors(image_dir, subject_id)
                image, label_image = davis_loader.load(image_dir, subject_id, i, all_colors)
            if self.dataset_type == "ffs":
                image, label_image = ffs_loader.load(image_dir, subject_id, i)
            if self.dataset_type == "simple":
                image, label_image = simple_loader.load(image_dir, subject_id, i)
            if self.dataset_type == "truck":
                image, label_image, _, _ = truck_data_loader.load(i, granularity)

            # count the pixel count of each color in annotation
            unique, counts = np.unique(label_image, return_counts=True)

            for unique, counts in zip(unique, counts):
                if self.pixel_count.get(unique) is None:
                    self.pixel_count[unique] = 0
                self.pixel_count[unique] += counts

            masks = {}
            # create masks for each color
            for color in np.unique(label_image):
                mask = np.array(label_image) == color
                masks[color] = mask
 
            self.images.append((np.array(image.convert("RGB"))))
            self.annotations.append(label_image)
            self.masks.append(masks)

        # superglue model disables this, we need to enable it
        torch.set_grad_enabled(True)

    def __len__(self):
        return len(self.images)

    def apply_augmentations(self, image, label_masks):
        """Apply augmentations to image and masks.

        Args:
        - image: Image
        - label_masks: Dictionary of masks
        
        Returns:
        - Augmented image and masks
        """

        # define augmentations
        image_transform = A.Compose(
            [
                # Crop and resize back to original size
                A.RandomResizedCrop(height=image.shape[0], width=image.shape[1], scale=(0.5, 1.0)),
                # #Brightness and contrast
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
                A.RandomGamma(gamma_limit=(80, 120)),
                A.GaussNoise(var_limit=(5.0, 20.0)),
            ]
        )

        new_masks = {}

        # masks to list
        masks = list(label_masks.values())
        # all masks to uint8
        masks = [mask.astype(np.uint8) for mask in masks]
        
        # apply image + mask augmentations
        transforms = image_transform(image=image, masks=masks)
        image = transforms["image"]
        masks = transforms["masks"]

        # convert masks back to bool
        for i, color in enumerate(label_masks.keys()):
            new_masks[color] = masks[i].astype(bool)

        return image, new_masks



    def __getitem__(self, idx):

        image = self.images[idx]
        masks = self.masks[idx]

        if self.train and self.augment:
            #augment images
            image, masks = self.apply_augmentations(image, masks)
        
          # read images for superpoint
        image0, inp0, scales0 = read_image(
            image=image,
            device=self.device,
            resize=(image.shape[1], image.shape[0]),
            rotation=0,
            resize_float=False,
        )

        # call superpoint model
        pred0 = self.super_glue_model({"image": inp0})
        
        # get points
        points = pred0["keypoints"][0].cpu().numpy().astype(int)

        # sort points by values
        scores = pred0["scores"][0].detach().cpu().numpy()

        # random choice
        if self.random_points:
            choice = np.random.choice(points.shape[0], self.min_points, replace=True)
        else:
            # only take the points with the highest scores
            choice = np.argsort(scores)[::-1].copy()

        # fill up with random choice to get min_points, this is important for dataset batching
        if len(choice) < self.min_points:
            choice = np.concatenate(
                (choice, np.random.choice(points.shape[0], self.min_points - len(choice)))
            )
        
        # choose the points
        points = points[choice]
        
        # get features for points
        features = pred0["descriptors"][0].T
        features = features[choice]

        clip_seg_feature_summed = apply_clip_seg(
            self.clip_seg_model,
            self.clip_seg_processor,
            self.clip_seg_prompts,
            image,
            image.shape[:2],
            self.device,
            )


        clip_seg_feature_summed = clip_seg_feature_summed / np.max(
            clip_seg_feature_summed
        )

        # apply clipseg features
        if self.use_clip_seg_feature:
            # add clipseg pixel value for every point position to feature vector
            features = torch.cat(
                (
                    features,
                    torch.tensor(clip_seg_feature_summed[points[:, 1], points[:, 0]])
                    .to(self.device)
                    .float()
                    .unsqueeze(1),
                ),
                dim=1,
            )

        # apply position features
        if self.use_position_feature:
            # normalized y
            normalized_y = points[:, 1] / image0.shape[0]
            # normalized x
            normalized_x = points[:, 0] / image0.shape[1]

            # append to features
            features = torch.cat(
                (features, torch.tensor(normalized_x).to(self.device).float().unsqueeze(1)),
                dim=1,
            )

            features = torch.cat(
                (features, torch.tensor(normalized_y).to(self.device).float().unsqueeze(1)),
                dim=1,
            )

        # create labels
        point_labels = np.zeros((points.shape[0], 1))

        # create adjacency matrix depending on the points and neighbor count
        matrix = create_adjacency_matrix(points, clip_seg_feature_summed,k=self.graph_neighbours)

        adj_matrix = nx.DiGraph(matrix)

        # extract edges from matrix
        edges = from_networkx(adj_matrix)

        # for each color in the masks we find the points that are in the mask
        for color, mask in masks.items():

            if self.class_count.get(color) is None:
                self.class_count[color] = 0

            mask = np.array(mask, dtype=np.uint8)
            mask = mask.astype(bool)


            points_in_mask = mask[points[:, 1], points[:, 0]]
            self.class_count[color] += np.sum(points_in_mask)

            point_labels[points_in_mask] = color

        # convert to tensor
        labels = torch.tensor(point_labels).to(self.device).long().squeeze(1)



        points = torch.tensor(points).to(self.device).long()
        edges = edges.edge_index.to(self.device).long()

        # clear gpu memory
        torch.cuda.empty_cache()
        del pred0
        del adj_matrix
        del matrix
        gc.collect()

        # check if requires grad is set
        return (
            (points, edges, features, labels),
            image0,
            self.annotations[idx],
        )

    def get_feature_count(self):
        return self.feature_count

    def get_label_count(self):
        return len(self.pixel_count.keys())
    
    def get_pixel_count(self):
        return self.pixel_count
