import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, global_mean_pool, global_max_pool


class EdgeGNN(nn.Module):
    """
    Edge-conditioned Graph Neural Network for HER overpotential prediction.

    Uses NNConv, which uses edge features to build dynamic weight matrices.
    This means the model can actually learn from composition-weighted edges.
    """

    def __init__(self, node_dim=10, edge_dim=3, hidden_dim=64, dropout=0.15):
        super().__init__()
        self.dropout = dropout

        # --- Layer 1: NNConv ---
        # Edge network: takes edge feature (3) -> produces (node_dim * hidden_dim) matrix
        edge_net1 = nn.Sequential(
            nn.Linear(edge_dim, 32),
            nn.ReLU(),
            nn.Linear(32, node_dim * hidden_dim),
        )
        self.conv1 = NNConv(node_dim, hidden_dim, edge_net1, aggr="mean")

        # --- Layer 2: NNConv ---
        edge_net2 = nn.Sequential(
            nn.Linear(edge_dim, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_dim * hidden_dim),
        )
        self.conv2 = NNConv(hidden_dim, hidden_dim, edge_net2, aggr="mean")

        # --- Layer 3: NNConv ---
        edge_net3 = nn.Sequential(
            nn.Linear(edge_dim, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_dim * hidden_dim),
        )
        self.conv3 = NNConv(hidden_dim, hidden_dim, edge_net3, aggr="mean")

        # --- Batch norm for stability ---
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # --- MLP Head: concat of mean + max pool ---
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        # Layer 1
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 3
        x = self.conv3(x, edge_index, edge_attr)
        x = self.bn3(x)
        x = F.relu(x)

        # Pool: concatenate mean and max
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        # MLP head
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x.squeeze(-1)


if __name__ == "__main__":
    from torch_geometric.data import Data

    print("Testing EdgeGNN...")

    # Fake graph: 5 nodes, 10 features; 20 edges, 3 features
    x = torch.randn(5, 10)
    edge_index = torch.tensor([
        [0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4],
        [1,2,3,4,0,0,1,2,3,4,0,1,2,3,4,1,0,4,3,2],
    ], dtype=torch.long)
    edge_attr = torch.randn(20, 3)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.batch = torch.zeros(5, dtype=torch.long)

    model = EdgeGNN(node_dim=10, edge_dim=3, hidden_dim=64)
    out = model(data)

    print(f"Input:  x={x.shape}, edge_attr={edge_attr.shape}")
    print(f"Output: {out.shape}")
    print(f"Predicted: {out.item():.2f} mV")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print("\nEdgeGNN works!")