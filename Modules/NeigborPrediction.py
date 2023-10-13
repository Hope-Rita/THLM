import copy

import dgl.sampling
import torch.nn as nn
import torch
import numpy as np

from R_HGNN.R_HGNN import R_HGNN


class NeighborPrediction(nn.Module):
    """

    """

    def __init__(self, node_num, hidden_dim, dropout):
        super(NeighborPrediction, self).__init__()
        self.node_num = node_num
        self.hidden_dim = hidden_dim
        self.nodeEmb = nn.ModuleDict()
        self.dropout = nn.Dropout(dropout)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def random_init(self, ntype, sparse=False):
        """Initialize the weight and bias embeddings

        Initialize label weight embedding with N(0, 0.02) while keeping PAD
        column to be 0. Initialize label bias embedding with 0.
        """
        mat = 0.02 * np.random.randn(self.node_num, self.nodeEmb[ntype].weight.shape[1])
        self.init_from(mat, ntype, sparse=sparse)

    def init_from(self, mat, ntype, sparse=False):
        """Initialize the weight and bias embeddings with given matrix

        Args:
            mat (ndarray): matrix used for initialize, shape = (nr_labels, hidden_size + 1)
        """
        if not isinstance(mat, np.ndarray):
            raise ValueError("Expect ndarray to initialize label embedding")
        if mat.shape[0] != self.node_num:
            raise ValueError("nr_labels mismatch!")

        # split weight and bias
        self.nodeEmb[ntype] = nn.Embedding.from_pretrained(
            torch.FloatTensor(mat),
            freeze=False,
            sparse=sparse,
        )


class GNNNeighborPred(NeighborPrediction):
    def __init__(self, node_num, hidden_dim=96, relation_hidden_dim=8, nlayers=2, nheads=8,
                 dropout=0.1, node_neighbors_min_num=5, residual=True, graph=None, masktr=0.2):
        super(GNNNeighborPred, self).__init__(node_num, hidden_dim, dropout)
        self.nlayers = nlayers
        self.maskrt = masktr
        sample_nodes_num = []
        for layer in range(nlayers):
            sample_nodes_num.append({etype: node_neighbors_min_num for etype in graph.canonical_etypes})
        self.hidden_dim = hidden_dim * nheads
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.transform = nn.ModuleDict()
        for ntype, nnode in node_num.items():
            self.transform[ntype] = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
            nn.init.xavier_uniform_(self.transform[ntype].weight)

        self.sampler = dgl.dataloading.MultiLayerNeighborSampler(sample_nodes_num)

        self.r_hgnn = R_HGNN(graph=graph,
                             input_dim_dict={ntype: self.hidden_dim for ntype in graph.ntypes},
                             hidden_dim=hidden_dim, relation_input_dim=relation_hidden_dim,
                             relation_hidden_dim=relation_hidden_dim,
                             num_layers=nlayers, n_heads=nheads, dropout=dropout,
                             residual=residual)

    def forward(self, graph, id_pairs_dict, nodeEmbed, word_emb, aver_feats, attn_mask, rawEmb=False):
        device = aver_feats.device
        word_emb_dict = {ntype: self.transform[ntype](aver_feats) for ntype in id_pairs_dict.keys()}

        neigh_pred_list = {}

        node_local, node_index = {}, {}
        for id_type, id_pairs in id_pairs_dict.items():
            unique_nodes, index_ = id_pairs.unique(return_inverse=True)  # shape: batch_sz * M
            node_local[id_type] = unique_nodes.cpu()
            node_index[id_type] = index_

        input_ns, output_ns, blocks = self.sampler.sample(graph, node_local)

        blocks = [b.to(device) for b in blocks]

        input_features = {}

        for stype, etype, dtype in blocks[0].canonical_etypes:
            ids = blocks[0].srcnodes[dtype].data[dgl.NID].clone().cpu()
            select_ids = np.unique(np.random.randint(low=0, high=len(ids),
                                                         size=int(len(ids) * self.maskrt))).tolist()
            ids[select_ids] = -1
            input_features[(stype, etype, dtype)] = nodeEmbed[dtype].weight[ids].to(device)

        nodes_representation, _ = self.r_hgnn(blocks, copy.deepcopy(input_features))

        for id_type, id_pairs in id_pairs_dict.items():
            nor_nodeEmb = nodes_representation[id_type]
            nodeEmb = nor_nodeEmb[node_index[id_type]].to(device)
            # print(nodeEmb.shape, word_emb_dict[id_type].shape)
            neigh_pred = torch.matmul(nodeEmb, word_emb_dict[id_type].unsqueeze(dim=-1)).squeeze(dim=-1)
            neigh_pred_list[id_type] = neigh_pred

        return neigh_pred_list

def get_neighborPrediction(graph, pred_type, node_num, hidden_dim, dropout, preEmbedding, mskrate):
    if pred_type == "gcnEmb":
        return GNNNeighborPred(node_num, graph=graph, masktr=mskrate)
    else:
        raise ValueError("Prediction Module {} not supported".format(pred_type))
