import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolutionLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj_matrix):
        # x: batch_size * seq_len * feat_dim
        # adj_matrix: batch_size * seq_len * seq_len
        output = self.linear(x)
        output = torch.bmm(adj_matrix, output)
        return output


class GraphNeuralNetwork(nn.Module):
    def __init__(self, dim, num_layer, final_nonlinearity=False):
        super(GraphNeuralNetwork, self).__init__()
        self.layers = []
        for i in range(num_layer):
            gc_layer = GraphConvolutionLayer(dim, dim)
            self.layers.append(gc_layer)
        self.layers = nn.ModuleList(self.layers)
        self.final_nonlinearity = final_nonlinearity

    def forward(self, x, adj_matrix):
        embedding = x
        for layer in self.layers[:-1]:
            embedding = layer(embedding, adj_matrix)
            embedding = F.relu(embedding)
        output = self.layers[-1](embedding, adj_matrix)
        if self.final_nonlinearity:
            output = F.relu(embedding)
        return output
