import torch
import torch.nn as nn
import torchvision.models as models
import torch_geometric.nn as pyg_nn

# Define the HybridCNNGAT model class
class HybridCNNGAT(nn.Module):
    def __init__(self):
        super(HybridCNNGAT, self).__init__()
        self.cnn = models.efficientnet_b4(weights="DEFAULT")
        
        for param in self.cnn.features[:-3].parameters():
            param.requires_grad = True
        
        self.cnn.classifier = nn.Identity()
        self.fc1 = nn.Linear(1792, 512)  # EfficientNet-B4 output size

        # GAT layers
        self.gat1 = pyg_nn.GATConv(512, 256, heads=4, concat=True, dropout=0.4)
        self.gat2 = pyg_nn.GATConv(256 * 4, 128, heads=4, concat=True, dropout=0.4)
        self.fc2 = nn.Linear(128 * 4, 3)  # Final output (3 classes)

        # Reset with fixed seed
        torch.manual_seed(42)
        self.gat1.reset_parameters()
        torch.manual_seed(42)
        self.gat2.reset_parameters()

    def forward(self, x, edge_index):
        cnn_features = self.cnn(x)
        x = torch.relu(self.fc1(cnn_features))
        x = self.gat1(x, edge_index)
        x = self.gat2(x, edge_index)
        return self.fc2(x)

# Helper to build edge_index
def create_edge_index(num_nodes):
    if num_nodes == 1:
        return torch.tensor([[0], [0]], dtype=torch.long)  # Self-loop for single image

    edge_index = []
    for i in range(num_nodes):
        for j in range(i + 1, min(i + 3, num_nodes)):
            edge_index.append([i, j])
            edge_index.append([j, i])
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()
