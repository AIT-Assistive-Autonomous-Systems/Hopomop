import argparse
from interest_point_detectors.superpoint.model_loader import superpoint_model_loader
from dataloaders.dataloader import SegDataset
from model.graph_classifier import GNNClassifier, SAGEClassifier, GATClassifier
from model.segmentor import Segmentor
from utils.visualize import visualize_segmentation


from transformers import AutoProcessor, CLIPSegForImageSegmentation
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from torch_geometric.data import Data
import torchmetrics

import cv2
from matplotlib import pyplot as plt

import seaborn as sns
import colorcet as cc
import utils.metrics as seg_metrics

import yaml

train_config =  {
    "nms_radius": 4,
    "min_points": 2048,
    "graph_neighbours": 8,
    "hidden_dim": 1024,
    "integration_dim": 256,
    "super_glue_threshold": 0.0002,
    "model_type": "SAGE",
    "sam_promt_type": "point",
    "box_threshold": 1,
    "point_threshold": 1,
    "sam_point_samples": 15,
    "dropouts": 0.1,
    "dropouts_edge": 0.5,
    "epochs": 100,
    "step_size": 50,
}

def train(config):
    device = "cuda"

    super_glue_threshold = config["super_glue_threshold"]
    nms_radius = config["nms_radius"]

    # load superglue model
    super_glue_model = superpoint_model_loader(keypoint_threshold=super_glue_threshold, nms_radius=nms_radius)

    # load clipseg model
    clip_seg_processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    clip_seg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)

    clip_seg_prompts = [config["clip_seg_prompt"]]
    # 10 random indices 0-99

    indices = np.random.randint(0, 40, 10)

    train_dataset = SegDataset(
    index_list = config["image_indices"],
    image_dir = config["data_dir"],
    dataset_type=config["dataset_type"],
    subject_id = config["subject_id"],
    super_glue_model = super_glue_model, 
    graph_neighbours = config["graph_neighbours"],
    min_points = config["min_points"],
    clip_seg_model = clip_seg_model, 
    clip_seg_processor = clip_seg_processor, 
    clip_seg_prompts = clip_seg_prompts,
    device = device,
    use_clip_seg_features=config["use_clip_seg_features"],
    use_position_feature=config["use_position_features"],)

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    feature_dim = train_dataset.get_feature_count()
    hidden_dim = config["hidden_dim"]
    integration_dim = config["integration_dim"]
    output_dim = train_dataset.get_label_count()

    dropout_val = config["dropouts"]
    dropout_val_edge = config["dropouts_edge"]



    if config["model_type"] == "GNN":
        model = GNNClassifier(feature_dim, hidden_dim, output_dim, integration_dim, dropout_val, dropout_val_edge).to(device)
    elif config["model_type"] ==  "SAGE":
        model = SAGEClassifier(feature_dim, hidden_dim, output_dim, integration_dim, dropout_val, dropout_val_edge).to(device)
    elif config["model_type"] ==  "GAT":
        model = GATClassifier(feature_dim, hidden_dim, output_dim, integration_dim, dropout_val, dropout_val_edge).to(device)

    # load model checkpoint
    model.load_state_dict(torch.load(config["checkpoint_path"]))
    model = model.to(device)
    
    model.eval()

    segmentor = Segmentor(config, device)

    metric_bal_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=output_dim, average="macro").to(device)
    metric_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=output_dim, average="macro").to(device)
    eval_data = {
        "images": [],
        "points": [],
        "predictions": [],
        "masks": [],
        "masks_gt": [],
        "class_metrics": [],
        "seg_metrics": []
    }

    for samples in train_dataloader:
        print("Processing image in class", config["subject_id"])
        features = samples[0][2]
        edge_index = samples[0][1]   
        gt = samples[0][3][0]
        data = Data(x=features, edge_index=edge_index)
    
        output = model(data)
        # apply softmax
        output_cat = torch.nn.functional.softmax(output, dim=1)
        # get the class with the highest probability
        output_cat = torch.argmax(output_cat, dim=1)

        # to numpy
        output_cat = output_cat.cpu().detach().numpy()
        points = samples[0][0].squeeze(0).cpu().detach().numpy()
        
        labels = gt.cpu().detach().numpy()
        scores = output.cpu().detach().numpy()
        features_np = features.cpu().detach().numpy()

        mask, new_points, new_classes = segmentor.segment(samples[1][0], points, output_cat, labels, scores, features_np)
        
        metric_val_f1 = metric_f1(output, gt)
        metric_val_bal_acc = metric_bal_acc(output, gt)

        eval_data["images"].append(samples[1][0].cpu().detach().numpy())
        eval_data["points"].append(new_points)
        eval_data["predictions"].append(new_classes)
        eval_data["masks"].append(mask)
        eval_data["masks_gt"].append(samples[2][0].cpu().detach().numpy())
        eval_data["class_metrics"].append((metric_val_f1, metric_val_bal_acc))
        eval_data["class_count"] = output_dim
        
    return eval_data


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        type=str,
        default="/workspaces/foundation-graph-segmentation/foundation_graph_segmentation/config/parameters.yaml",
        help="Config file for training",
    )

    args = parser.parse_args()

    train_config = {}
    with open(args.config_file, "r") as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)

    eval_data = train(train_config)

    visualize_segmentation(eval_data, f"../results/output_{train_config['subject_id']}")
    



   