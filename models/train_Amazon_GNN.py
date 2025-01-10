import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import coo_matrix
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

# Parse the dataset
def load_interactions(file_path):
    interactions = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = list(map(int, line.strip().split()))
            user_id = parts[0]
            item_ids = parts[1:]
            interactions[user_id] = item_ids
    return interactions

# Build graph data from interactions
def build_graph(interactions):
    edges = []
    for user, items in interactions.items():
        for item in items:
            edges.append([user, item])
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index

# Load train and test data
train_data = load_interactions("data/train.txt")
test_data = load_interactions("data/test.txt")


# Create edge indices for train and test data
train_edge_index = build_graph(train_data)
test_edge_index = build_graph(test_data)

# Define GNN model
class GNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, 2)  # Binary classification (connected or not)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# Generate node features (dummy features for this example)
num_nodes = max(train_edge_index.max().item(), test_edge_index.max().item()) + 1
x = torch.eye(num_nodes)  # Identity matrix as features

# Define training and evaluation function
def train(model, data, optimizer, criterion, epochs=3):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(data.x, data.edge_index)
        loss = criterion(output[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        pred = logits.argmax(dim=1)
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        accuracy = correct / data.test_mask.sum()
        print(f"Accuracy: {accuracy:.4f}")

# Prepare data object for PyTorch Geometric
data = Data(
    x=x, 
    edge_index=train_edge_index, 
    train_mask=torch.rand(num_nodes) < 0.8,  # Random train/test split
    test_mask=torch.rand(num_nodes) >= 0.8,
    y=torch.randint(0, 2, (num_nodes,))  # Random binary labels
)

# Initialize and train the model
model = GNN(num_features=num_nodes, hidden_channels=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

train(model, data, optimizer, criterion)

# Evaluate the model
evaluate(model, data)