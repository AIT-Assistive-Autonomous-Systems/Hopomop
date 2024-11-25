import torch
from torch.nn import Dropout
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.utils import dropout_edge
import torch.nn.functional as F

import timm


class GNNClassifier(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        output_dim=3,
        integration_dim=512,
        dropout_val=0.2,
        dropout_val_edge=0.6,
    ):
        super(GNNClassifier, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, integration_dim)

        self.dropout = Dropout(dropout_val)

        self.dropout_val_edge = dropout_val_edge

        self.integration_linear = torch.nn.Linear(integration_dim, integration_dim)
        self.integration_linear2 = torch.nn.Linear(integration_dim, integration_dim)
        self.classifier = torch.nn.Linear(integration_dim, output_dim)

    def forward(self, data):
        #x, edge_index, batch = data.x[0], data.edge_index[0], data.batch

        edge_index, _ = dropout_edge(
            edge_index, p=self.dropout_val_edge, training=self.training
        )

        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)

        edge_index, _ = dropout_edge(
            edge_index, p=self.dropout_val_edge, training=self.training
        )

        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)

        edge_index, _ = dropout_edge(
            edge_index, p=self.dropout_val_edge, training=self.training
        )
        x = self.conv3(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.integration_linear(x)
        x = torch.relu(x)
        x = self.classifier(x)

        #x = torch.log_softmax(x, dim=1)

        return x


class SAGEClassifier(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        output_dim=3,
        integration_dim=512,
        dropout_val=0.2,
        dropout_val_edge=0.6,
    ):
        super(SAGEClassifier, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim, normalize=True)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, normalize=True)
        self.conv3 = SAGEConv(hidden_dim, integration_dim, normalize=True)

        self.dropout = Dropout(dropout_val)

        self.dropout_val_edge = dropout_val_edge

        self.integration_linear = torch.nn.Linear(integration_dim, integration_dim)
        self.integration_linear2 = torch.nn.Linear(integration_dim, integration_dim)
        self.classifier = torch.nn.Linear(integration_dim, output_dim)

    def forward(self, data):

        x, edge_index, batch = data.x[0], data.edge_index[0], data.batch

        edge_index, _ = dropout_edge(
            edge_index, p=self.dropout_val_edge, training=self.training
        )

        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)

        edge_index, _ = dropout_edge(
            edge_index, p=self.dropout_val_edge, training=self.training
        )

        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)

        edge_index, _ = dropout_edge(
            edge_index, p=self.dropout_val_edge, training=self.training
        )
        x = self.conv3(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.integration_linear(x)
        x = torch.relu(x)
        x = self.classifier(x)

        #x = torch.log_softmax(x, dim=1)
        return x


class GATClassifier(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        output_dim=3,
        integration_dim=512,
        dropout_val=0.2,
        dropout_val_edge=0.6,
    ):
        super(GATClassifier, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=8)
        self.conv2 = GATConv(hidden_dim * 8, hidden_dim, heads=8)
        self.conv3 = GATConv(hidden_dim * 8, integration_dim)

        self.dropout = Dropout(dropout_val)

        self.dropout_val_edge = dropout_val_edge

        self.integration_linear = torch.nn.Linear(integration_dim, integration_dim)
        self.integration_linear2 = torch.nn.Linear(integration_dim, integration_dim)
        self.classifier = torch.nn.Linear(integration_dim, output_dim)

    def forward(self, data):

        x, edge_index, batch = data.x[0], data.edge_index[0], data.batch

        edge_index, _ = dropout_edge(
            edge_index, p=self.dropout_val_edge, training=self.training
        )

        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)

        edge_index, _ = dropout_edge(
            edge_index, p=self.dropout_val_edge, training=self.training
        )

        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)

        edge_index, _ = dropout_edge(
            edge_index, p=self.dropout_val_edge, training=self.training
        )
        x = self.conv3(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.integration_linear(x)
        x = torch.relu(x)
        x = self.classifier(x)

        #x = torch.log_softmax(x, dim=2)

        return x
