import torch
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
import torch_geometric.nn as geom_nn
import torch.nn.functional as F

def build_hetero_graph(df):
    """
    Constructs a heterogeneous graph from transaction records.
    Card holders (cards) and Merchants are proxied from the dataset's PCA components
    to build a bipartite relation: card -> transacts_with -> merchant.
    """
    df = df.copy()
    
    # Construct node proxies from V-features
    df['card_id'] = (df['V1'].round(1).astype(str) + "_" + df['V2'].round(1).astype(str)).astype('category').cat.codes
    df['merchant_id'] = (df['V3'].round(1).astype(str) + "_" + df['V4'].round(1).astype(str)).astype('category').cat.codes

    num_cards = df['card_id'].nunique()
    num_merchants = df['merchant_id'].nunique()

    # Initialize PyTorch Geometric HeteroData object
    data = HeteroData()

    # Node index tensors
    data['card'].node_id = torch.arange(num_cards)
    data['merchant'].node_id = torch.arange(num_merchants)

    # Initialize dummy node features (card demographics / merchant category codes in production)
    data['card'].x = torch.ones((num_cards, 16))
    data['merchant'].x = torch.ones((num_merchants, 16))

    # Bipartite Edge Index: card -> transacts_with -> merchant
    edge_index = torch.tensor([df['card_id'].values, df['merchant_id'].values], dtype=torch.long)
    data['card', 'transacts_with', 'merchant'].edge_index = edge_index

    # Edge Features (Amount and other PCA components as transaction descriptors)
    edge_attr = torch.tensor(df[['V5', 'V6', 'Amount_scaled' if 'Amount_scaled' in df.columns else 'Amount']].values, dtype=torch.float)
    data['card', 'transacts_with', 'merchant'].edge_attr = edge_attr

    # Binary Classification Target Labels on the Edges (Class = 1 for fraud, 0 for legitimate)
    data['card', 'transacts_with', 'merchant'].edge_label = torch.tensor(df['Class'].values, dtype=torch.long)

    return data

class HeteroFraudGNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        
        # Heterogeneous Graph Convolution Layer 1: message passing from card to merchant
        self.conv1 = geom_nn.HeteroConv({
            ('card', 'transacts_with', 'merchant'): geom_nn.SAGEConv((-1, -1), hidden_channels)
        }, agg='sum')
        
        # Heterogeneous Graph Convolution Layer 2: message passing on updated node embeddings
        self.conv2 = geom_nn.HeteroConv({
            ('card', 'transacts_with', 'merchant'): geom_nn.SAGEConv((-1, -1), hidden_channels)
        }, agg='sum')

        # Linear Classifier to map aggregated edge embeddings to binary predictions
        self.classifier = torch.nn.Linear(hidden_channels * 2, 2)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        # Layer 1
        h_dict = self.conv1(x_dict, edge_index_dict)
        h_dict = {key: x.relu() for key, x in h_dict.items()}
        
        # Layer 2
        h_dict = self.conv2(h_dict, edge_index_dict)
        h_dict = {key: x.relu() for key, x in h_dict.items()}
        
        # Concatenate sender (card) and receiver (merchant) embeddings for the specific transaction edges
        u_src, v_dst = edge_label_index[0], edge_label_index[1]
        edge_feats = torch.cat([h_dict['card'][u_src], h_dict['merchant'][v_dst]], dim=-1)
        
        return self.classifier(edge_feats)

if __name__ == "__main__":
    print("GNN module structure compiled successfully!")
