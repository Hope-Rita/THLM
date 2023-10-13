import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from tqdm import tqdm

from HetSANN.RelationGraphConv import RelationGraphConv
from HetSANN.HeteroConv import HeteroGraphConv


class HetSANNLayer(nn.Module):
    def __init__(self, graph: dgl.DGLHeteroGraph, input_dim: int, hidden_dim: int, n_heads: int = 8,
                 dropout: float = 0.2, residual: bool = True):
        """

        :param graph: a heterogeneous graph
        :param input_dim: int, input dimension
        :param hidden_dim: int, hidden dimension
        :param n_heads: int, number of attention heads
        :param dropout: float, dropout rate
        :param residual: boolean, residual connections or not
        """
        super(HetSANNLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.residual = residual

        # node transformation parameters of each type
        self.node_transformation_weight = nn.ParameterDict({
            ntype: nn.Parameter(torch.randn(input_dim, n_heads * hidden_dim))
            for ntype in graph.ntypes
        })

        # relation attention parameters of each type
        self.relations_attention_weight = nn.ParameterDict({
            etype: nn.Parameter(torch.randn(n_heads, 2 * hidden_dim))
            for etype in graph.etypes
        })

        # hetero conv modules, each RelationGraphConv deals with a single type of relation
        # hence, there are two RelationGraphConv modules in a SingleViewLayer
        self.hetero_conv = HeteroGraphConv({
            etype: RelationGraphConv(in_feats=(input_dim, input_dim), out_feats=hidden_dim,
                                     num_heads=n_heads, dropout=dropout, negative_slope=0.2)
            for etype in graph.etypes
        })

        if self.residual:
            # residual connection
            self.res_fc = nn.ModuleDict()
            for ntype in graph.ntypes:
                if graph.number_of_dst_nodes(ntype) > 0:
                    self.res_fc[ntype] = nn.Linear(input_dim, n_heads * hidden_dim, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        for weight in self.node_transformation_weight:
            nn.init.xavier_normal_(self.node_transformation_weight[weight], gain=gain)
        for weight in self.relations_attention_weight:
            nn.init.xavier_normal_(self.relations_attention_weight[weight], gain=gain)
        if self.residual:
            for ntype in self.res_fc:
                nn.init.xavier_normal_(self.res_fc[ntype].weight, gain=gain)

    def forward(self, graph: dgl.DGLHeteroGraph, node_features: dict):
        """

        :param graph: dgl.DGLHeteroGraph
        :param node_features: dict, {ntype: node_feature},
        :return: output_features: dict, {relation_type: target_node_features}
        """
        # dictionary of input source features and destination features
        input_src = node_features

        if graph.is_block:
            input_dst = {}
            for ntype in node_features:
                input_dst[ntype] = node_features[ntype][:graph.number_of_dst_nodes(ntype)]
        else:
            input_dst = node_features

        # output_features, dict {(stype, etype, dtype): features}
        relation_features = self.hetero_conv(graph, input_src, input_dst, self.node_transformation_weight,
                                           self.relations_attention_weight)

        output_features = {}
        # aggregate different types of relations of the target node with sum aggregation (dgl implementation which is the same as the original paper)
        for ntype in input_dst:
            if graph.number_of_dst_nodes(ntype) != 0:
                dst_node_features = [relation_features[(stype, etype, dtype)] for stype, etype, dtype in relation_features if dtype == ntype]
                # Tensor, shape -> (relations_num, num_ntype, n_heads * hidden_dim)
                dst_node_features = torch.stack(dst_node_features, dim=0)
                # Tensor, shape -> (num_ntype, n_heads * hidden_dim)
                output_features[ntype] = torch.mean(dst_node_features, dim=0)

        # residual connection for the target node
        if self.residual:
            for ntype in output_features:
                output_features[ntype] = output_features[ntype] + self.res_fc[ntype](input_dst[ntype])

        # node features after relation crossing layer, {ntype: node_features}
        return output_features


class HetSANN(nn.Module):
    def __init__(self, graph: dgl.DGLHeteroGraph, input_dim_dict: dict, hidden_dim: int, num_layers: int, n_heads: int = 4,
                 dropout: float = 0.2, residual: bool = True):
        """

        :param graph: a heterogeneous graph
        :param input_dim_dict: input dim dictionary
        :param hidden_dim: int, hidden dimension
        :param num_layers: int, number of stacked layers
        :param n_heads: int, number of attention heads
        :param dropout: float, dropout rate
        :param residual: boolean, residual connections or not
        """
        super(HetSANN, self).__init__()

        self.input_dim_dict = input_dim_dict
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.residual = residual

        # align the dimension of different types of nodes
        self.projection_layer = nn.ModuleDict({
            ntype: nn.Linear(input_dim_dict[ntype], hidden_dim * n_heads) for ntype in input_dim_dict
        })

        # each layer takes in the heterogeneous graph as input
        self.layers = nn.ModuleList()

        # for each relation_layer
        for layer_num in range(self.num_layers):
            self.layers.append(HetSANNLayer(graph, hidden_dim * n_heads, hidden_dim, n_heads, dropout, residual))

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        for ntype in self.projection_layer:
            nn.init.xavier_normal_(self.projection_layer[ntype].weight, gain=gain)

    def forward(self, blocks: list, node_features: dict):
        """

        :param blocks: list of sampled dgl.DGLHeteroGraph
        :param node_features: node features, dict, {"type": features}
        :return:
        """
        # feature projection
        for ntype in node_features:
            node_features[ntype] = self.projection_layer[ntype](node_features[ntype])

        # graph convolution
        for block, layer in zip(blocks, self.layers):
            node_features = layer(block, node_features)

        return node_features

    def inference(self, graph: dgl.DGLHeteroGraph, node_features: dict, device: str):
        """
        mini-batch inference of final representation over all node types. Outer loop: Interate the layers, Inner loop: Interate the batches

        :param graph: The whole relational graphs
        :param node_features: features of all the nodes in the whole graph, dict, {"type": features}
        :param device: device str
        """
        with torch.no_grad():
            # interate over each layer
            for index, layer in enumerate(self.layers):
                # Tensor, features of all types of nodes, store on cpu
                y = {
                    ntype: torch.zeros(
                        graph.number_of_nodes(ntype), self.hidden_dim * self.n_heads) for ntype in graph.ntypes}
                # full sample for each type of nodes
                sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
                dataloader = dgl.dataloading.NodeDataLoader(
                    graph,
                    {ntype: torch.arange(graph.number_of_nodes(ntype)) for ntype in graph.ntypes},
                    sampler,
                    batch_size=1280,
                    shuffle=True,
                    drop_last=False,
                    num_workers=4)

                tqdm_dataloader = tqdm(dataloader, ncols=120)
                for batch, (input_nodes, output_nodes, blocks) in enumerate(tqdm_dataloader):
                    block = blocks[0].to(device)

                    input_features = {ntype: node_features[ntype][input_nodes[ntype]].to(device) for ntype in input_nodes.keys()}

                    if index == 0:
                        # feature projection for the first layer in the full batch inference
                        for ntype in input_features:
                            input_features[ntype] = self.projection_layer[ntype](input_features[ntype])

                    h = layer(block, input_features)

                    for k in h.keys():
                        y[k][output_nodes[k]] = h[k].cpu()

                    tqdm_dataloader.set_description(f'inference for the {batch}-th batch in model {index}-th layer')

                # update the features of all the nodes (after the graph convolution) in the whole graph
                node_features = y

        return y
