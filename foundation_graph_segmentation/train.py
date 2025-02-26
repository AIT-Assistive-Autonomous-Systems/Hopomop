import argparse
from interest_point_detectors.superpoint.model_loader import (
    superpoint_model_loader,
)
from dataloaders.dataloader import SegDataset
from model.graph_classifier import (
    GNNClassifier,
    SAGEClassifier,
    GATClassifier,
)
from transformers import AutoProcessor, CLIPSegForImageSegmentation
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from torch_geometric.data import Data
import torchmetrics

import yaml


def train(config):
    device = "cuda"

    super_glue_threshold = config["super_glue_threshold"]
    nms_radius = config["nms_radius"]

    checkpoint_path = config["checkpoint_path"]

    # load superglue model
    super_glue_model = superpoint_model_loader(
        keypoint_threshold=super_glue_threshold, nms_radius=nms_radius
    )

    # load clipseg model
    clip_seg_processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    clip_seg_model = CLIPSegForImageSegmentation.from_pretrained(
        "CIDAS/clipseg-rd64-refined"
    ).to(device)

    clip_seg_prompts = [config["clip_seg_prompt"]]

    train_dataset = SegDataset(
        index_list=config["image_indices"],
        image_dir=config["data_dir"],
        dataset_type=config["dataset_type"],
        subject_id=config["subject_id"],
        super_glue_model=super_glue_model,
        graph_neighbours=config["graph_neighbours"],
        min_points=config["min_points"],
        clip_seg_model=clip_seg_model,
        clip_seg_processor=clip_seg_processor,
        clip_seg_prompts=clip_seg_prompts,
        use_clip_seg_features=config["use_clip_seg_features"],
        use_position_feature=config["use_position_features"],
        random_points=False,
        train=True,
        augment = config["augment"],
        device=device,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    feature_dim = train_dataset.get_feature_count()
    hidden_dim = config["hidden_dim"]
    integration_dim = config["integration_dim"]
    output_dim = train_dataset.get_label_count()

    learning_rate = config["learning_rate"]
    num_epochs = config["epochs"]
    step_size = config["step_size"]



    gamma = 0.5

    dropout_val = config["dropouts"]
    dropout_val_edge = config["dropouts_edge"]

    loss_type = config["loss_type"]

    pixel_count = train_dataset.get_pixel_count()

    class_count = np.array([pixel_count[i] for i in range(output_dim)])
    class_count = class_count / np.sum(class_count)

    weights = torch.tensor(1 - class_count).to(device).float()
    weights[0] = 0.05


    if train_dataset.get_label_count() == 2:
        loss_type = "NLLLoss"
    else:
        loss_type = "NLLLoss"

    if loss_type == "NLLLoss":
        criterion = nn.NLLLoss(weight=weights)
    elif loss_type == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss(weight=weights)

    if config["model_type"] == "GNN":
        model = GNNClassifier(
            feature_dim,
            hidden_dim,
            output_dim,
            integration_dim,
            dropout_val,
            dropout_val_edge,
        ).to(device)
    elif config["model_type"] == "SAGE":
        model = SAGEClassifier(
            feature_dim,
            hidden_dim,
            output_dim,
            integration_dim,
            dropout_val,
            dropout_val_edge,
        ).to(device)
    elif config["model_type"] == "GAT":
        model = GATClassifier(
            feature_dim,
            hidden_dim,
            output_dim,
            integration_dim,
            dropout_val,
            dropout_val_edge,
        ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler that reduces the learning rate by 10x every 100 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    model.train()
    # metric_list

    metric_bal_acc = torchmetrics.classification.Accuracy(
        task="multiclass", num_classes=output_dim, average="macro"
    ).to(device)
    metric_f1 = torchmetrics.classification.F1Score(
        task="multiclass", num_classes=output_dim, average="macro"
    ).to(device)
    import cv2
    max_f1 = -1
    for epoch in range(num_epochs):
        samples = next(iter(train_dataloader))


        features = samples[0][2]
        edge_index = samples[0][1]

        data = Data(x=features, edge_index=edge_index)

        optimizer.zero_grad()

        gt = samples[0][3][0].to(device)

        # forward pass
        output_cat = model(data)

        f1 = metric_f1(output_cat, gt)
        if f1 > max_f1 and f1 != 1.0:
            max_f1 = f1
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model saved with F1: {f1:.3f}")
            # exit on 0.95, but not on 1 because of images with only background
            if f1 > 0.99:
                break

        if loss_type == "NLLLoss":
            output_cat = torch.nn.functional.log_softmax(output_cat, dim=1)
            loss = criterion(output_cat, gt)
        else:
            output_cat = torch.nn.functional.softmax(output_cat, dim=1)
            loss = criterion(output_cat, gt)

        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()

        print(
            f"Epoch {epoch}, Loss: {loss}, Bal Acc: {metric_bal_acc(output_cat, gt):.3f}, F1: {metric_f1(output_cat, gt):.3f}"
        )


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

    train(train_config)



