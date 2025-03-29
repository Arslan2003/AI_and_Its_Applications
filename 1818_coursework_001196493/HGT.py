import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# To be finished...

def create_graph(csv_file):
    df = pd.read_csv(csv_file)

    # Create a NetworkX graph
    G = nx.Graph()

    # Add nodes and edges
    for _, row in df.iterrows():
        G.add_edge(row['from_id'], row['to_id'], **row.to_dict())

    return G

# Load the CSV file
G = create_graph("HI-Small_Trans_formatted.csv")

# Visualize the graph
nx.draw(G, with_labels=True)
plt.show()


import torch.nn.functional as F

from torch_geometric.nn import HGTConv
from torch_geometric.transforms import RandomNodeSplit
# import pandas as pd
# import torch
# from torch_geometric.data import Data
# import networkx as nx
# import matplotlib.pyplot as plt
#
# def create_graph(csv_file):
#     df = pd.read_csv(csv_file)
#
#     # Create edge index and edge features
#     edge_index = torch.tensor([[sender_id, receiver_id] for sender_id, receiver_id in zip(df['from_id'], df['to_id'])], dtype=torch.long).T
#     edge_attr = torch.tensor(df[['Timestamp', 'Amount Sent', 'Sent Currency', 'Amount Received', 'Received Currency', 'Payment Format', 'Is Laundering']].values, dtype=torch.float)
#
#     # Create labels (adjust as needed)
#     labels = df['Is Laundering'].values  # Assign labels based on edge-level information
#
#     # Create PyG Data object
#     data = Data(edge_index=edge_index, edge_attr=edge_attr, y=labels)
#
#     return data
#
# def visualize_graph(data):
#     G = nx.Graph()
#
#     # Add nodes
#     for node_id in range(data.num_nodes):
#         G.add_node(node_id)
#
#     # Add edges
#     for edge in zip(data.edge_index[0], data.edge_index[1]):
#         G.add_edge(edge[0].item(), edge[1].item())
#
#     # Visualize the graph
#     nx.draw(G, with_labels=True)
#     plt.show()
#
# visualize_graph(create_graph('HI-Small_Trans_formatted.csv'))

# class HGTModel(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_heads):
#         super().__init__()
#         self.conv1 = HGTConv(in_channels, hidden_channels, num_heads)
#         self.conv2 = HGTConv(hidden_channels, out_channels, num_heads)
#
#     def forward(self, x, edge_index, edge_attr):
#         x = self.conv1(x, edge_index, edge_attr).relu()
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.conv2(x, edge_index, edge_attr)
#         return F.log_softmax(x, dim=1)
#
# def train(model, data, optimizer, device):
#     model.train()
#     optimizer.zero_grad()
#     out = model(None, data.edge_index.to(device), data.edge_attr.to(device)).to(device)  # Pass None for x as we're not using node features
#     loss = F.nll_loss(out[data.train_mask].to(device), data.y[data.train_mask].to(device))
#     loss.backward()
#     optimizer.step()
#
# def test(model, data, device):
#     model.eval()
#     out = model(None, data.edge_index.to(device), data.edge_attr.to(device)).to(device)
#     pred = out.argmax(dim=1)
#     correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
#     acc = correct / data.test_mask.sum().item()
#     return acc
#
# if __name__ == "__main__":
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     # Load data
#     data = create_graph('HI-Small_Trans_formatted.csv')
#
#     # Create model, optimizer, and loss function
#     model = HGTModel(in_channels=data.edge_attr.shape[1], hidden_channels=64, out_channels=2, num_heads=8).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#
#     # Train the model
#     for epoch in range(20):
#         train(model, data, optimizer, device)
#         acc = test(model, data, device)
#         print(f'Epoch: {epoch+1}, Accuracy: {acc:.4f}')