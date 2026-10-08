import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool


class CatalystGNN(nn.Module):
    """
    Graph Neural Network for predicting HER overpotential of HEA catalysts.

    Architecture:
    - 3 GNN message-passing layers
    - Global mean pooling
    - 2-layer MLP head
    - Single scalar output (overpotential in mV)
    """

    def __init__(self, node_feat_dim=9, edge_feat_dim=2, hidden_dim=64, dropout=0.2):
        super().__init__()

        self.dropout = dropout

        # --- GNN Layers ---
        # Layer 1: node_feat_dim -> hidden_dim
        # GINConv with a small MLP inside
        mlp1 = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv1 = GINConv(mlp1)

        # Layer 2: hidden_dim -> hidden_dim
        mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv2 = GINConv(mlp2)

        # Layer 3: hidden_dim -> hidden_dim
        mlp3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv3 = GINConv(mlp3)

        # --- MLP Head ---
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 3
        x = self.conv3(x, edge_index)
        x = F.relu(x)

        # Global mean pooling -> single vector per graph
        x = global_mean_pool(x, batch)

        # MLP Head
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)  # (num_graphs, 1)

        return x.squeeze(-1)  # (num_graphs,)


if __name__ == "__main__":
    # Quick self-test
    from torch_geometric.data import Data

    print("Testing CatalystGNN...")

    # Create a fake graph: 5 nodes, 9 features, 20 edges
    x = torch.randn(5, 9)
    edge_index = torch.tensor([
        [0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4],
        [1,2,3,4,0,0,1,2,3,4,0,1,2,3,4,1,0,4,3,2]
    ], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)
    data.batch = torch.zeros(5, dtype=torch.long)

    model = CatalystGNN(node_feat_dim=9, hidden_dim=64)
    out = model(data)

    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Predicted overpotential: {out.item():.2f} mV")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print("\nModel works!")