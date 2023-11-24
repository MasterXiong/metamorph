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
    def __init__(self, input_dim, hidden_dims, output_dim, final_nonlinearity=False):
        super(GraphNeuralNetwork, self).__init__()
        self.layers = []
        self.dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(self.dims) - 1):
            gc_layer = GraphConvolutionLayer(self.dims[i], self.dims[i + 1])
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
