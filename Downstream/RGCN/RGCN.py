import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import dgl.nn.pytorch as dglnn
import dgl


class RelGraphConvLayer(nn.Module):
    def __init__(self, graph: dgl.DGLHeteroGraph, in_feat, out_feat, num_bases, dropout=0.0, activation=None, self_loop=False):
        super(RelGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        # number of basis relations
        self.num_bases = num_bases
        self.etypes = graph.etypes
        self.activation = activation
        self.self_loop = self_loop
        self.relation_conv = dglnn.HeteroGraphConv({
            # weight in GraphConv is set to False, trainable parameters are initialized in self.weight
            etype: dglnn.GraphConv(in_feat, out_feat, norm='right', weight=False, bias=False) for etype in self.etypes
        })

        self.use_basis = 0 < num_bases < len(self.etypes)
        if self.use_basis:
            # model, its forward() return the linear combination of bases, shape (num_etypes, in_feat, out_feat)
            self.basis = dglnn.WeightBasis((in_feat, out_feat), num_bases, len(self.etypes))
        else:
            # shape -> (rel_names, in_feat, out_feat)
            self.weight = nn.Parameter(torch.randn(len(self.etypes), in_feat, out_feat))

        self.h_bias = nn.Parameter(torch.randn(out_feat))

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.randn(in_feat, out_feat))

        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.weight, gain=gain)
        nn.init.zeros_(self.h_bias)
        if self.self_loop:
            nn.init.xavier_uniform_(self.loop_weight, gain=gain)

    def forward(self, graph: dgl.DGLHeteroGraph, node_features: dict):
        """

        :param graph: dgl.DGLHeteroGraph
        :param node_features: dict, {"type": features}
        :return:
            dict[str, torch.Tensor]
                New node features for each node type.
        """

        graph = graph.local_var()

        # dictionary of input source features and destination features
        input_src = node_features
        if graph.is_block:
            input_dst = {}
            for ntype in node_features:
                input_dst[ntype] = node_features[ntype][:graph.number_of_dst_nodes(ntype)]
        else:
            input_dst = node_features

        # shape -> (num_etypes, in_feat, out_feat)
        weight = self.basis() if self.use_basis else self.weight
        weight_dict = {self.etypes[i]: {'weight': w.squeeze(dim=0)} for i, w in enumerate(torch.split(weight, split_size_or_sections=1, dim=0))}

        output_features = self.relation_conv(graph, (input_src, input_dst), mod_kwargs=weight_dict)

        dst_node_features = {}
        for dtype in output_features:
            h = output_features[dtype]
            if self.self_loop:
                h = h + torch.matmul(input_dst[dtype], self.loop_weight)
            h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            dst_node_features[dtype] = self.dropout(h)
        return dst_node_features


class RGCN(nn.Module):
    def __init__(self, graph: dgl.DGLHeteroGraph, input_dim_dict: dict, hidden_sizes: list, num_bases: int, dropout=0.0, use_self_loop=False):
        """

        :param graph: graph, dgl.DGLHeteroGraph
        :param input_dim_dict:
        :param hidden_sizes:
        :param num_bases:
        :param dropout:
        :param use_self_loop:
        """
        super(RGCN, self).__init__()
        self.input_dim_dict = input_dim_dict
        self.hidden_sizes = hidden_sizes
        self.num_bases = num_bases

        self.dropout = dropout
        self.use_self_loop = use_self_loop

        # align the dimension of different types of nodes
        self.projection_layer = nn.ModuleDict({
            ntype: nn.Linear(input_dim_dict[ntype], hidden_sizes[0]) for ntype in input_dim_dict
        })

        self.layers = nn.ModuleList()

        self.layers.append(
            RelGraphConvLayer(graph, self.hidden_sizes[0], self.hidden_sizes[0], self.num_bases,
                              dropout=self.dropout,
                              activation=F.relu, self_loop=self.use_self_loop))

        for index in range(1, len(self.hidden_sizes)):
            self.layers.append(RelGraphConvLayer(graph, self.hidden_sizes[index-1], self.hidden_sizes[index], self.num_bases, dropout=self.dropout,
                                                 activation=F.relu, self_loop=self.use_self_loop))

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

        :param graph: The whole graph
        :param node_features: features of all the nodes in the whole graph, dict, {"type": features}
        :param device: device str
        """
        with torch.no_grad():

            # interate over each layer
            for index, layer in enumerate(self.layers):
                # Tensor, features of all types of nodes, store on cpu
                y = {
                    ntype: torch.zeros(
                        graph.number_of_nodes(ntype), self.hidden_sizes[index]) for ntype in graph.ntypes}
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
                        if k in output_nodes:
                            y[k][output_nodes[k]] = h[k].cpu()

                    tqdm_dataloader.set_description(f'inference for the {batch}-th batch in model {index}-th layer')

                # update the features of all the nodes (after the graph convolution) in the whole graph
                node_features = y

        return y
